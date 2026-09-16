import asyncio
import os
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse, urlunparse
from uuid import UUID

import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from arq.connections import ArqRedis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import Settings, get_settings
from app.core.encryption import encrypt
from app.core.passwords import hash_password
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.models.appointment import Appointment, AppointmentStatus
from app.models.channel import Channel
from app.models.conversation import Conversation
from app.models.doctor import Doctor
from app.models.knowledge_base import KnowledgeBase
from app.models.lead import Lead
from app.models.message import Message
from app.models.operator import Operator
from app.models.tenant import Tenant
from app.models.user import User
from app.rag.embeddings import EMBEDDING_DIMENSIONS
from app.repositories.appointment import AppointmentRepository
from app.repositories.channel import ChannelRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.doctor import DoctorRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.lead import LeadRepository
from app.repositories.message import MessageRepository
from app.repositories.operator import OperatorRepository
from app.repositories.tenant import TenantRepository
from app.repositories.user import UserRepository

# --- the database these tests run against ---------------------------------
#
# Its own, never the one the application is using. Several tests ask the
# database a genuinely global question -- "how many clinics exist?", "is
# anything provisioned yet?" -- and those cannot be scoped to rows the test
# created, because the code under test is not scoped either. Run against a
# development database with a clinic in it and they fail; run against an
# empty one and they pass. That made the suite a report on how the
# developer's machine happened to be configured.
#
# Derived from DATABASE_URL rather than configured separately, so there is
# nothing to keep in sync: whatever server the app is pointed at, the tests
# get a "<name>_test" database on it.
_TEST_DB_SUFFIX = "_test"


def _with_database(url: str, name: str) -> str:
    parts = urlparse(url)
    return urlunparse(parts._replace(path=f"/{name}"))


async def _create_database_if_missing(base_url: str, name: str) -> None:
    """CREATE DATABASE, over asyncpg directly.

    Not through SQLAlchemy: CREATE DATABASE cannot run inside a transaction,
    and the synchronous driver SQLAlchemy would reach for on a driverless URL
    (psycopg2) is not a dependency of this project.
    """
    parts = urlparse(_with_database(base_url, "postgres"))
    connection = await asyncpg.connect(
        user=parts.username,
        password=parts.password,
        host=parts.hostname,
        port=parts.port or 5432,
        database="postgres",
    )
    try:
        exists = await connection.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if not exists:
            # Quoted identifier, not a bind parameter -- CREATE DATABASE
            # takes none. The name is derived from our own configuration,
            # never from anything a test supplies.
            await connection.execute(f'CREATE DATABASE "{name}"')
    finally:
        await connection.close()


@pytest.fixture(scope="session", autouse=True)
def test_database() -> Iterator[str]:
    """Create the test database if it does not exist, migrate it, and point
    the whole process at it.

    The environment variable, not just a fixture: code under test opens its
    own sessions through app.core.db (provisioning and FAQ seeding both do),
    and those read get_settings() rather than anything a fixture could hand
    them. Setting DATABASE_URL and clearing the cache is what makes those
    connections land in the test database too.
    """
    original = os.environ.get("DATABASE_URL")
    base_url = get_settings().database_url
    name = urlparse(base_url).path.lstrip("/") + _TEST_DB_SUFFIX
    test_url = _with_database(base_url, name)

    asyncio.run(_create_database_if_missing(base_url, name))

    os.environ["DATABASE_URL"] = test_url
    get_settings.cache_clear()

    # Through Alembic rather than metadata.create_all: the schema the tests
    # run against is then the schema a deploy produces, migrations included
    # -- a create_all schema would silently diverge the moment a migration
    # did something the models do not describe, which is exactly where the
    # interesting bugs live.
    command.upgrade(Config("alembic.ini"), "head")

    try:
        yield test_url
    finally:
        if original is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = original
        get_settings.cache_clear()


def isolated_settings(**overrides: object) -> Settings:
    """A Settings built from these arguments and nothing else.

    `_env_file=None` is the point. Without it pydantic-settings reads the
    developer's own .env, so a test asserting "no clinic phone number is
    configured" passes on a machine where none is and fails on one where one
    is — again, the machine actually running the deployment. Every value the
    app requires at startup is supplied here so that switching the file off
    cannot fail validation.
    """
    base: dict[str, object] = {
        "database_url": "postgresql+asyncpg://test:test@localhost/test",
        "redis_url": "redis://localhost:6379/0",
        "webhook_verify_token": "test-verify-token",
        "meta_app_secret": "test-app-secret",
        "encryption_key": "Hq3_REB-V0twf7iBgCPCSUZQiG44egxyiZg9kOKRxUg=",
        "session_secret_key": "test-session-secret",
        "openai_api_key": "sk-test",
    }
    return Settings(_env_file=None, **{**base, **overrides})  # type: ignore[arg-type]


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """Wraps each test in an outer transaction (with a SAVEPOINT for any inner
    flush/commit) that's always rolled back afterward, so tests never leave
    data behind and never see another test's writes. Requires the Dockerized
    Postgres from docker-compose.yml to be reachable at DATABASE_URL.

    expire_on_commit=False matches app.core.db's real sessionmaker: without
    it, a route that calls session.commit() (e.g. the dashboard's
    create/cancel endpoints) would expire every ORM object already loaded in
    this shared test session — including the `seed` fixture's rows — and
    the next plain attribute access on one of them would need an implicit
    reload that isn't safe outside an awaited call.
    """
    engine = create_async_engine(get_settings().database_url, poolclass=NullPool)
    async with engine.connect() as connection:
        trans = await connection.begin()
        session = AsyncSession(
            bind=connection, join_transaction_mode="create_savepoint", expire_on_commit=False
        )
        try:
            yield session
        finally:
            await session.close()
            await trans.rollback()
    await engine.dispose()


# Dedicated Redis DB index for tests — never db 0 (dev/prod), so a blanket
# flushdb() before/after each test can't ever touch real data.
TEST_REDIS_URL = "redis://localhost:6379/15"


@pytest.fixture
async def redis_pool() -> AsyncIterator[ArqRedis]:
    """A real ArqRedis pool against a dedicated test DB, flushed clean
    before and after each test. Debounce logic relies on real Redis Lua
    scripts and list/counter semantics that aren't meaningfully fakeable.
    """
    pool = ArqRedis.from_url(TEST_REDIS_URL)
    await pool.flushdb()
    try:
        yield pool
    finally:
        await pool.flushdb()
        await pool.aclose()


@pytest.fixture
def as_tenant() -> Callable[[UUID], AbstractContextManager[None]]:
    """`with as_tenant(tenant.id): ...` sets the current tenant for the block
    and always resets it afterward, even if the block raises.
    """

    @contextmanager
    def _as_tenant(tenant_id: UUID) -> Iterator[None]:
        token = set_current_tenant(tenant_id)
        try:
            yield
        finally:
            reset_current_tenant(token)

    return _as_tenant


@dataclass
class TenantSeed:
    channel: Channel
    user: User
    conversation: Conversation
    knowledge_base: KnowledgeBase
    operator: Operator
    doctor: Doctor
    appointment: Appointment
    lead: Lead
    message: Message


@dataclass
class Seed:
    tenant_a: Tenant
    tenant_b: Tenant
    a: TenantSeed
    b: TenantSeed


@pytest.fixture
async def seed(
    db_session: AsyncSession, as_tenant: Callable[[UUID], AbstractContextManager[None]]
) -> Seed:
    """Creates two full tenants (A and B), each with one row in every
    tenant-scoped table plus a conversation + message, via the real
    repositories. Isolation tests use this as their starting data.
    """
    tenant_repo = TenantRepository(db_session)
    tenant_a = await tenant_repo.create(name="Clinic A", status="active")
    tenant_b = await tenant_repo.create(name="Clinic B", status="active")

    async def _build(tenant: Tenant) -> TenantSeed:
        with as_tenant(tenant.id):
            # Stored encrypted, like every real channel row (see
            # app.core.encryption, app.models.channel.Channel.credentials) —
            # app.workers.tasks._send_reply decrypts unconditionally, so a
            # plaintext seed value here would make every worker test that
            # sends a reply fail with DecryptionError instead of testing
            # what it's meant to test. Plaintext value stays "token" so
            # existing assertions on the decrypted value don't need to
            # change.
            channel = await ChannelRepository(db_session).create(
                type="instagram", credentials=encrypt("token"), external_id=f"ig-{tenant.id}"
            )
            user = await UserRepository(db_session).create(
                channel_id=channel.id, external_id=f"ext-{tenant.id}"
            )
            conversation = await ConversationRepository(db_session).create(
                user_id=user.id, status="open"
            )
            knowledge_base = await KnowledgeBaseRepository(db_session).create(
                question="What are your hours?",
                answer="9 to 5.",
                embedding=[0.0] * EMBEDDING_DIMENSIONS,
            )
            operator = await OperatorRepository(db_session).create(
                name="Dr. Smith",
                role="doctor",
                username=f"dr.smith-{tenant.id}",
                password_hash=hash_password("seed-password"),
            )
            doctor = await DoctorRepository(db_session).create(
                name="Dr. Smith",
                specialty="Stomatolog",
                working_hours="09:00 - 18:00",
            )
            appointment = await AppointmentRepository(db_session).create(
                user_id=user.id,
                doctor_id=doctor.id,
                doctor_name=doctor.name,
                scheduled_at=datetime.now(UTC),
                status=AppointmentStatus.SCHEDULED,
            )
            lead = await LeadRepository(db_session).create(
                user_id=user.id,
                conversation_id=conversation.id,
                patient_name="Aziza",
                phone="+998 90 000 00 00",
                topic="implant",
            )
            message = await MessageRepository(db_session).create(
                conversation_id=conversation.id,
                sender="patient",
                content="Hello",
                channel="instagram",
            )
        return TenantSeed(
            channel=channel,
            user=user,
            conversation=conversation,
            knowledge_base=knowledge_base,
            operator=operator,
            doctor=doctor,
            appointment=appointment,
            lead=lead,
            message=message,
        )

    seed_a = await _build(tenant_a)
    seed_b = await _build(tenant_b)
    return Seed(tenant_a=tenant_a, tenant_b=tenant_b, a=seed_a, b=seed_b)
