from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.passwords import hash_password
from app.core.tenant_context import TenantContextError, get_current_tenant
from app.models.appointment import AppointmentStatus
from app.models.message import Message
from app.rag.embeddings import EMBEDDING_DIMENSIONS
from app.repositories.appointment import AppointmentRepository
from app.repositories.base import (
    CrossTenantAccessError,
    MissingTenantColumnError,
    TenantScopedRepository,
)
from app.repositories.channel import ChannelRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.doctor import DoctorRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.lead import LeadRepository
from app.repositories.message import MessageRepository
from app.repositories.operator import OperatorRepository
from app.repositories.user import UserRepository
from tests.conftest import Seed


def _new_channel_kwargs(seed: Seed) -> dict[str, Any]:
    return {"type": "instagram", "credentials": "new-token", "external_id": "ig-new-channel"}


def _new_user_kwargs(seed: Seed) -> dict[str, Any]:
    return {"channel_id": seed.a.channel.id, "external_id": "ext-new-user"}


def _new_conversation_kwargs(seed: Seed) -> dict[str, Any]:
    # Closed, not open: the seeded user already has an open conversation and
    # uq_conversations_open_per_user allows only one at a time. These cases
    # are about tenant stamping and cross-tenant reads, so any valid row
    # serves — a second *open* one would be testing the wrong thing, and is
    # covered directly in test_conversation_service.py instead.
    return {"user_id": seed.a.user.id, "status": "closed"}


def _new_knowledge_base_kwargs(seed: Seed) -> dict[str, Any]:
    return {
        "question": "New question?",
        "answer": "New answer.",
        "embedding": [0.0] * EMBEDDING_DIMENSIONS,
    }


def _new_operator_kwargs(seed: Seed) -> dict[str, Any]:
    return {
        "name": "Dr. New",
        "role": "doctor",
        "username": "dr.new-unique",
        "password_hash": hash_password("new-secret"),
    }


def _new_appointment_kwargs(seed: Seed) -> dict[str, Any]:
    return {
        "user_id": seed.a.user.id,
        "doctor_name": "Dr. New",
        # An hour off the seeded booking: uq_appointments_tenant_id_scheduled_at
        # now covers confirmed as well as scheduled, so a second row at the
        # same instant would collide with the seed's own appointment.
        "scheduled_at": datetime.now(UTC) + timedelta(hours=1),
        "status": AppointmentStatus.SCHEDULED,
    }


def _new_doctor_kwargs(seed: Seed) -> dict[str, Any]:
    return {"name": "Dr. New", "specialty": "Ortodont", "working_hours": "10:00 - 17:00"}


def _new_lead_kwargs(seed: Seed) -> dict[str, Any]:
    return {"patient_name": "Yangi bemor", "phone": "+998 90 111 22 33"}


TENANT_SCOPED_CASES: list[
    tuple[str, type[TenantScopedRepository[Any]], Callable[[Seed], dict[str, Any]]]
] = [
    ("channel", ChannelRepository, _new_channel_kwargs),
    ("user", UserRepository, _new_user_kwargs),
    ("conversation", ConversationRepository, _new_conversation_kwargs),
    ("knowledge_base", KnowledgeBaseRepository, _new_knowledge_base_kwargs),
    ("operator", OperatorRepository, _new_operator_kwargs),
    ("appointment", AppointmentRepository, _new_appointment_kwargs),
    ("doctor", DoctorRepository, _new_doctor_kwargs),
    ("lead", LeadRepository, _new_lead_kwargs),
]
CASE_IDS = [case[0] for case in TENANT_SCOPED_CASES]


# --- 1. TenantScopedRepository isolation, across all 6 tenant-scoped models ---


@pytest.mark.parametrize("attr, repo_cls, new_kwargs", TENANT_SCOPED_CASES, ids=CASE_IDS)
async def test_get_hides_other_tenants_row(
    attr: str,
    repo_cls: type[TenantScopedRepository[Any]],
    new_kwargs: Callable[[Seed], dict[str, Any]],
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    other_row = getattr(seed.b, attr)
    with as_tenant(seed.tenant_a.id):
        result = await repo_cls(db_session).get(other_row.id)
    assert result is None


@pytest.mark.parametrize("attr, repo_cls, new_kwargs", TENANT_SCOPED_CASES, ids=CASE_IDS)
async def test_list_only_returns_own_tenant(
    attr: str,
    repo_cls: type[TenantScopedRepository[Any]],
    new_kwargs: Callable[[Seed], dict[str, Any]],
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        rows = await repo_cls(db_session).list()
    ids_seen = {row.id for row in rows}
    assert getattr(seed.a, attr).id in ids_seen
    assert getattr(seed.b, attr).id not in ids_seen


@pytest.mark.parametrize("attr, repo_cls, new_kwargs", TENANT_SCOPED_CASES, ids=CASE_IDS)
async def test_create_stamps_current_tenant(
    attr: str,
    repo_cls: type[TenantScopedRepository[Any]],
    new_kwargs: Callable[[Seed], dict[str, Any]],
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id):
        obj = await repo_cls(db_session).create(**new_kwargs(seed))
    assert obj.tenant_id == seed.tenant_a.id


@pytest.mark.parametrize("attr, repo_cls, new_kwargs", TENANT_SCOPED_CASES, ids=CASE_IDS)
async def test_create_with_foreign_tenant_id_raises(
    attr: str,
    repo_cls: type[TenantScopedRepository[Any]],
    new_kwargs: Callable[[Seed], dict[str, Any]],
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    with as_tenant(seed.tenant_a.id), pytest.raises(CrossTenantAccessError):
        await repo_cls(db_session).create(tenant_id=seed.tenant_b.id, **new_kwargs(seed))


# --- 2. MessageRepository isolation (scoped indirectly via conversation) ---


async def test_message_get_hides_message_on_other_tenant_conversation(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    with as_tenant(seed.tenant_a.id):
        result = await MessageRepository(db_session).get(seed.b.message.id)
    assert result is None


async def test_message_create_on_other_tenant_conversation_raises(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    with as_tenant(seed.tenant_a.id), pytest.raises(CrossTenantAccessError):
        await MessageRepository(db_session).create(
            conversation_id=seed.b.conversation.id,
            sender="bot",
            content="leaked?",
            channel="instagram",
        )


async def test_message_list_for_other_tenant_conversation_is_empty(
    db_session: AsyncSession, seed: Seed, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> None:
    with as_tenant(seed.tenant_a.id):
        result = await MessageRepository(db_session).list_for_conversation(seed.b.conversation.id)
    assert list(result) == []


# --- 3. tenant_context fails closed ---


def test_get_current_tenant_raises_without_context() -> None:
    with pytest.raises(TenantContextError):
        get_current_tenant()


# --- 4. __init_subclass__ guard rejects models without tenant_id ---


def test_tenant_scoped_repository_rejects_model_without_tenant_id() -> None:
    with pytest.raises(MissingTenantColumnError):

        class BadRepository(TenantScopedRepository[Message]):
            model = Message
