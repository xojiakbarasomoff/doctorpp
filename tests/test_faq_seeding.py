import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from uuid import UUID

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.base import ChannelType
from app.core.config import Settings
from app.core.encryption import encrypt
from app.core.faq_seeding import (
    FaqSeedingError,
    load_faqs,
    resolve_faq_tenant_id,
    seed_faqs_if_configured,
)
from app.models.channel import Channel
from app.models.tenant import Tenant
from tests.conftest import Seed, isolated_settings


def _write(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "faqs.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


# --- load_faqs: everything is validated before an embedding is ever paid for ---


def test_load_faqs_reads_questions_answers_and_categories(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        [
            {"question": "Manzilingiz qayerda?", "answer": "Yunusobod", "category": "umumiy"},
            {"question": "Narxlar?", "answer": "100 000 so'm"},
        ],
    )

    faqs = load_faqs(path)

    assert [faq.question for faq in faqs] == ["Manzilingiz qayerda?", "Narxlar?"]
    assert faqs[0].category == "umumiy"
    assert faqs[1].category is None


def test_load_faqs_rejects_a_blank_answer(tmp_path: Path) -> None:
    """A blank answer would be embedded, stored, and then retrieved as
    grounding that says nothing -- worse than the FAQ simply being absent,
    because retrieval reports a hit.
    """
    path = _write(tmp_path, [{"question": "Manzilingiz qayerda?", "answer": "   "}])

    with pytest.raises(FaqSeedingError, match="Invalid FAQ entry"):
        load_faqs(path)


def test_load_faqs_rejects_a_json_object(tmp_path: Path) -> None:
    path = _write(tmp_path, {"question": "Manzilingiz qayerda?", "answer": "Yunusobod"})

    with pytest.raises(FaqSeedingError, match="must contain a JSON array"):
        load_faqs(path)


def test_load_faqs_reports_malformed_json_rather_than_raising_json_error(tmp_path: Path) -> None:
    path = tmp_path / "faqs.json"
    path.write_text('[{"question": "x",}]', encoding="utf-8")

    with pytest.raises(FaqSeedingError, match="not valid JSON"):
        load_faqs(path)


def test_load_faqs_reports_a_missing_file_by_name(tmp_path: Path) -> None:
    with pytest.raises(FaqSeedingError, match="No such file"):
        load_faqs(tmp_path / "absent.json")


# --- tenant resolution: through the channel, never a raw UUID ---


async def test_resolve_tenant_id_finds_the_tenant_behind_a_channel(
    db_session: AsyncSession, seed: Seed
) -> None:
    tenant_id = await resolve_faq_tenant_id(db_session, seed.a.channel.external_id)

    assert tenant_id == seed.tenant_a.id


async def test_resolve_tenant_id_refuses_to_guess_between_two_channels(
    db_session: AsyncSession, seed: Seed
) -> None:
    """Two clinics share this database. Picking either one silently would
    load one clinic's prices into the other's knowledge base.
    """
    with pytest.raises(FaqSeedingError, match="clinics have channels"):
        await resolve_faq_tenant_id(db_session, None)


async def test_resolve_tenant_id_accepts_a_clinic_that_is_only_on_telegram(
    db_session: AsyncSession,
) -> None:
    """The knowledge base is shared across platforms, and a clinic can be
    live on Telegram while its Instagram token is still pending. Requiring an
    Instagram channel would leave exactly that deployment with an empty
    knowledge base, which is the state where the assistant answers nothing
    but a refusal.
    """
    tenant = Tenant(name="Faqat Telegram", status="active")
    db_session.add(tenant)
    await db_session.flush()
    db_session.add(
        Channel(
            tenant_id=tenant.id,
            type=ChannelType.TELEGRAM,
            external_id="8123456789",
            credentials=encrypt("pending"),
            is_active=True,
        )
    )
    await db_session.commit()

    assert await resolve_faq_tenant_id(db_session, None) == tenant.id


async def test_resolve_tenant_id_is_not_confused_by_one_clinic_on_two_platforms(
    db_session: AsyncSession,
) -> None:
    """Two channels, one clinic, one obvious answer -- ambiguity is measured
    in clinics, not channel rows.
    """
    tenant = Tenant(name="Ikki platforma", status="active")
    db_session.add(tenant)
    await db_session.flush()
    for channel_type, external_id in (
        (ChannelType.INSTAGRAM, "17841400000000000"),
        (ChannelType.TELEGRAM, "8123456789"),
    ):
        db_session.add(
            Channel(
                tenant_id=tenant.id,
                type=channel_type,
                external_id=external_id,
                credentials=encrypt("pending"),
                is_active=True,
            )
        )
    await db_session.commit()

    assert await resolve_faq_tenant_id(db_session, None) == tenant.id


async def test_resolve_tenant_id_falls_back_to_the_only_clinic_before_any_channel_exists(
    db_session: AsyncSession,
) -> None:
    """First boot provisions the tenant before either platform's token has
    been accepted. Requiring a channel here would leave exactly that
    deployment with an empty knowledge base, which is the state where the
    assistant answers nothing but a refusal.
    """
    tenant = Tenant(name="Hali kanalsiz", status="active")
    db_session.add(tenant)
    await db_session.commit()

    assert await resolve_faq_tenant_id(db_session, None) == tenant.id


async def test_resolve_tenant_id_still_refuses_to_guess_between_two_channelless_clinics(
    db_session: AsyncSession,
) -> None:
    for name in ("Birinchi", "Ikkinchi"):
        db_session.add(Tenant(name=name, status="active"))
    await db_session.commit()

    with pytest.raises(FaqSeedingError, match="none has a channel yet"):
        await resolve_faq_tenant_id(db_session, None)


async def test_resolve_tenant_id_says_so_when_nothing_is_provisioned_at_all(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(FaqSeedingError, match="PROVISION_TENANT_NAME"):
        await resolve_faq_tenant_id(db_session, None)


async def test_resolve_tenant_id_rejects_an_unknown_account_id(
    db_session: AsyncSession, seed: Seed
) -> None:
    with pytest.raises(FaqSeedingError, match="No instagram channel found"):
        await resolve_faq_tenant_id(db_session, "ig-does-not-exist")


# --- the startup hook itself ---


def _settings(**overrides: object) -> Settings:
    return isolated_settings(**overrides)


async def test_seeding_is_skipped_when_unconfigured() -> None:
    """Unset is the resting state, so this runs on every startup of every
    deployment that has already been seeded. It must not touch the database
    at all -- note there is no db_session fixture here, so a stray connection
    attempt would fail the test.
    """
    await seed_faqs_if_configured(_settings(seed_faqs_from=None))


async def test_a_broken_faq_file_does_not_stop_the_app_from_starting(
    tmp_path: Path,
    db_session: AsyncSession,
    seed: Seed,
    as_tenant: Callable[[UUID], AbstractContextManager[None]],
) -> None:
    """The webhook being up matters more than the knowledge base being
    loaded: the assistant can answer without FAQs, but a web process that
    refuses to boot answers nothing at all.
    """
    path = _write(tmp_path, [{"question": "Manzilingiz qayerda?", "answer": ""}])

    await seed_faqs_if_configured(_settings(seed_faqs_from=str(path)))


async def test_a_missing_faq_file_does_not_stop_the_app_from_starting(
    tmp_path: Path,
) -> None:
    await seed_faqs_if_configured(_settings(seed_faqs_from=str(tmp_path / "absent.json")))


# --- the clinician roster, loaded the same way ---


def test_load_doctors_reads_the_shipped_roster() -> None:
    """The file that ships with the repo has to parse, because nothing else
    checks it before a deploy reads it at startup: a typo here is a clinic
    that boots with no clinicians, which silently drops the appointment book
    to one seat per slot and leaves "qaysi shifokor qabul qiladi?"
    unanswerable.
    """
    from app.core.doctor_seeding import load_doctors

    doctors = load_doctors(Path("data/doctors.json"))

    assert len(doctors) == 6
    assert all(doctor.name and doctor.specialty for doctor in doctors)
    assert {doctor.working_hours for doctor in doctors} == {"09:00 - 18:00"}


def test_load_doctors_rejects_a_blank_name(tmp_path: Path) -> None:
    from app.core.doctor_seeding import load_doctors

    path = tmp_path / "doctors.json"
    path.write_text('[{"name": "  ", "specialty": "LOR"}]', encoding="utf-8")

    with pytest.raises(FaqSeedingError, match="Invalid doctor entry"):
        load_doctors(path)
