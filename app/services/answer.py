"""Turning a patient's message into a reply.

The model is given the clinic's persona (written in the dashboard, see
app.services.persona), the facts that go with it, and the conversation, and
writes the reply itself. Nothing it says is checked or rewritten here.

What this module adds to the persona is the part that is not the clinic's to
edit: the markers the rest of the system reads ([[BOOK:...]], [[CALLBACK:...]])
and the appointment book they refer to.
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.tenant_context import get_current_tenant
from app.models.tenant import Tenant
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import ChatMessage, LLMProvider, get_llm_provider
from app.rag.retrieval import retrieve_knowledge
from app.repositories.appointment import AppointmentRepository
from app.repositories.doctor import DoctorRepository
from app.repositories.knowledge_base import KnowledgeBaseMatch
from app.repositories.knowledge_document import ChunkMatch
from app.services import persona as persona_service
from app.services.language import conversation_script
from app.services.patient_profile import Profile
from app.services.persona import ClinicFacts, PatientState, Persona
from app.services.tenant_resolution import clinic_rules as clinic_rules_for

logger = logging.getLogger(__name__)

_CALLBACK_CONTRACT = """\
If the patient asks to be called and gives a number, end your reply with \
[[CALLBACK:<their number>|<why they want the call, at most twelve words>]]. \
The patient never sees it; it puts them on the list of people to ring. Write \
it once per request."""

_BOOKING_CONTRACT = """\
You can book appointments yourself, from THE APPOINTMENT BOOK below. Only \
times listed there exist. When the patient has agreed to one specific listed \
time and you have their name, telephone number and reason for the visit, \
confirm it in one short message and end that same message with \
[[BOOK:YYYY-MM-DDTHH:MM|full name|telephone|reason]], using the date and \
time exactly as the book gives them. The patient never sees the marker; it \
is what writes the appointment down. Never write it before they have agreed \
to a time, and never tell a patient they are booked in a message that does \
not carry it. Say the day the way a person would ("bugun", "ertaga", \
"1-sentabr"), never as 2026-09-01."""

_LANGUAGE_CONTRACT = "When the patient's language is unclear, reply in {default_language}."


def _setting(stored: dict[str, Any], key: str, fallback: str | None = None) -> str | None:
    value = stored.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _clinic_facts(
    tenant: Tenant | None,
    settings: Settings,
    doctors: Sequence[Any],
    rules: Sequence[str],
) -> ClinicFacts:
    """The dashboard's own settings first, the deployment's environment second.

    The dashboard has had boxes for the address, the telephone numbers and the
    hours from the start, and nothing read them: a clinic could type its
    address, press save, and watch the bot say it did not know.
    """
    stored = tenant.settings if tenant is not None else {}
    lead = (
        f"{settings.doctor_name}, {settings.doctor_specialty}"
        if settings.doctor_name and settings.doctor_specialty
        else None
    )
    return ClinicFacts(
        name=tenant.name if tenant is not None else None,
        address=_setting(stored, "clinic_address", settings.clinic_address),
        landmark=_setting(stored, "clinic_landmark"),
        phone_numbers=_setting(stored, "clinic_phone_numbers", settings.clinic_phone_numbers),
        work_hours=_setting(stored, "clinic_work_hours", settings.clinic_work_hours),
        doctors=[(d.name, d.specialty, d.working_hours) for d in doctors],
        lead_doctor=lead,
        lead_doctor_background=settings.doctor_background_text if lead else None,
        rules=rules,
    )


def _source(match: ChunkMatch) -> str:
    """Where a piece of an uploaded file came from: "narxlar.pdf, 3-bet"."""
    label = match.chunk.label
    return f"{match.filename}, {label}" if label else match.filename


def _build_system_prompt(
    persona: Persona,
    *,
    facts: ClinicFacts,
    matches: Sequence[KnowledgeBaseMatch],
    state: PatientState,
    default_language: str,
    appointment_book: str | None = None,
    chunks: Sequence[ChunkMatch] = (),
) -> str:
    prompt = persona_service.render(
        persona,
        facts=facts,
        knowledge=[(m.knowledge_base.question, m.knowledge_base.answer) for m in matches],
        state=state,
        excerpts=[(_source(c), c.chunk.content) for c in chunks],
    )
    prompt += "\n\n" + _LANGUAGE_CONTRACT.format(default_language=default_language)
    prompt += "\n\n" + _CALLBACK_CONTRACT
    if appointment_book is not None:
        prompt += "\n\n" + _BOOKING_CONTRACT + appointment_book
    return prompt


async def _appointment_book(session: AsyncSession, doctor_count: int) -> str:
    """The free slots, rendered for the prompt.

    Imported here rather than at the top: app.services.booking already
    imports from app.services.appointment, and keeping this one-way avoids a
    cycle through the worker.
    """
    from app.services.appointment import slot_capacity
    from app.services.booking import free_slots, render

    now = datetime.now(UTC)
    slots = await free_slots(
        AppointmentRepository(session), now, capacity=slot_capacity(doctor_count)
    )
    return render(slots, now)


async def generate_answer(
    session: AsyncSession,
    user_message: str,
    embedding_provider: EmbeddingProvider | None = None,
    llm_provider: LLMProvider | None = None,
    settings: Settings | None = None,
    history: Sequence[ChatMessage] | None = None,
    patient: Profile | None = None,
) -> str:
    """Ask the model to answer the patient, with the clinic's facts in hand.

    `history` is the conversation's earlier turns, oldest first. Callers get
    it from app.services.conversation.context_for_reply, which already
    excludes the messages being answered right now, so appending
    user_message here cannot repeat them.
    """
    resolved_settings = settings or get_settings()
    tenant_id = get_current_tenant()
    tenant = await session.get(Tenant, tenant_id)

    knowledge = await retrieve_knowledge(
        session, user_message, embedding_provider=embedding_provider
    )
    doctors = await DoctorRepository(session).list_active()

    # A language the patient asked for outranks the alphabet their latest
    # message happens to be in: somebody who said "по-русски" and then typed
    # "ok" was being answered in Uzbek again.
    script = (patient.language if patient else None) or conversation_script(history, user_message)
    state = PatientState(
        name=patient.name if patient else None,
        phone=patient.phone if patient else None,
        script=script,
    )
    system_prompt = _build_system_prompt(
        persona_service.from_settings(tenant.settings if tenant is not None else None),
        facts=_clinic_facts(
            tenant, resolved_settings, doctors, await clinic_rules_for(session, tenant_id)
        ),
        matches=knowledge.faqs,
        chunks=knowledge.chunks,
        state=state,
        default_language=resolved_settings.default_reply_language,
        appointment_book=(
            await _appointment_book(session, len(doctors))
            if resolved_settings.booking_enabled
            else None
        ),
    )
    provider = llm_provider or get_llm_provider()
    conversation: list[ChatMessage] = [
        *(history or []),
        ChatMessage(role="user", content=user_message),
    ]
    return await provider.generate(system_prompt, conversation)
