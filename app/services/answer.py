"""Turning a patient's message into a reply.

The model is given who it is, what the clinic has told us about itself, the
knowledge base rows that match the question, the free appointment slots, and
the conversation so far -- and writes the reply itself. There are no canned
answers and no rewriting of what it says; the only thing done to its text
afterwards is taking the [[BOOK:...]] and [[CALLBACK:...]] markers out, which
happens in the callers.
"""

import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.tenant_context import get_current_tenant
from app.models.doctor import Doctor
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import ChatMessage, LLMProvider, get_llm_provider
from app.rag.retrieval import retrieve_relevant_faqs
from app.repositories.appointment import AppointmentRepository
from app.repositories.doctor import DoctorRepository
from app.repositories.knowledge_base import KnowledgeBaseMatch
from app.services.tenant_resolution import clinic_rules as clinic_rules_for

logger = logging.getLogger(__name__)

_CLINIC_OPENING = """\
You answer patients who write to a medical clinic in a direct-message chat, \
as the clinic's front desk."""

_DOCTOR_OPENING = """\
You answer the direct messages sent to {doctor_name}, {doctor_specialty}, as \
the doctor's administrator."""

_DOCTOR_BACKGROUND = """

About the doctor:
{background}"""

_PREAMBLE = """\
{opening}

Reply in the language and alphabet the patient writes in. When that is \
unclear, reply in {default_language}.{clinic_facts}"""

_FAQ_BLOCK = """

Clinic information:
{faq_context}"""

_BOOKING_BLOCK = """

Booking: you can book appointments yourself from THE APPOINTMENT BOOK below. \
When a patient has agreed to a specific free time, end your reply with \
[[BOOK:YYYY-MM-DDTHH:MM|full name|telephone|reason]], using the date and \
time exactly as the book lists them. The patient never sees the marker; it \
is what writes the appointment down."""

_CALLBACK_BLOCK = """

If a patient asks to be called and gives a number, end your reply with \
[[CALLBACK:<their number>|<why they want the call>]]. The patient never sees \
it; it puts them on the list of people to ring."""


def _format_faq_context(matches: Sequence[KnowledgeBaseMatch]) -> str:
    return "\n\n".join(
        f"Q: {match.knowledge_base.question}\nA: {match.knowledge_base.answer}" for match in matches
    )


def _doctor_roster(doctors: Sequence[Doctor]) -> str:
    if not doctors:
        return ""
    lines = "\n".join(
        f"- {doctor.name} — {doctor.specialty} — {doctor.working_hours}" for doctor in doctors
    )
    return f"\n\nDoctors:\n{lines}"


def _clinic_facts_block(
    clinic_address: str | None,
    clinic_phone_numbers: str | None,
    doctors: Sequence[Doctor] = (),
    clinic_work_hours: str | None = None,
) -> str:
    lines: list[str] = []
    if clinic_address:
        lines.append(f"Address: {clinic_address}")
    if clinic_phone_numbers:
        lines.append(f"Phone: {clinic_phone_numbers}")
    if clinic_work_hours:
        lines.append(f"Open: {clinic_work_hours}")
    details = "\n\nClinic details:\n" + "\n".join(lines) if lines else ""
    return details + _doctor_roster(doctors)


def _clinic_rules_block(rules: Sequence[str]) -> str:
    """The clinic's own instructions, as typed into the dashboard."""
    if not rules:
        return ""
    listed = "\n".join(f"- {rule}" for rule in rules)
    return f"\n\nThe clinic's own instructions:\n{listed}"


def _opening(
    doctor_name: str | None, doctor_specialty: str | None, doctor_background: str | None = None
) -> str:
    if doctor_name and doctor_specialty:
        opening = _DOCTOR_OPENING.format(doctor_name=doctor_name, doctor_specialty=doctor_specialty)
        if doctor_background:
            opening += _DOCTOR_BACKGROUND.format(background=doctor_background)
        return opening
    return _CLINIC_OPENING


def _build_system_prompt(
    matches: Sequence[KnowledgeBaseMatch],
    default_language: str,
    clinic_phone_numbers: str | None,
    clinic_address: str | None,
    doctors: Sequence[Doctor] = (),
    clinic_work_hours: str | None = None,
    clinic_rules: Sequence[str] = (),
    doctor_name: str | None = None,
    doctor_specialty: str | None = None,
    doctor_background: str | None = None,
    appointment_book: str | None = None,
) -> str:
    prompt = _PREAMBLE.format(
        opening=_opening(doctor_name, doctor_specialty, doctor_background),
        default_language=default_language,
        clinic_facts=_clinic_facts_block(
            clinic_address, clinic_phone_numbers, doctors, clinic_work_hours
        ),
    )
    if matches:
        prompt += _FAQ_BLOCK.format(faq_context=_format_faq_context(matches))
    prompt += _clinic_rules_block(clinic_rules)
    prompt += _CALLBACK_BLOCK
    if appointment_book is not None:
        prompt += _BOOKING_BLOCK + appointment_book
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
) -> str:
    """Ask the model to answer the patient, with the clinic's facts in hand.

    `history` is the conversation's earlier turns, oldest first. Callers get
    it from app.services.conversation.context_for_reply, which already
    excludes the messages being answered right now, so appending
    user_message here cannot repeat them.
    """
    resolved_settings = settings or get_settings()

    matches = await retrieve_relevant_faqs(
        session, user_message, embedding_provider=embedding_provider
    )
    doctors = await DoctorRepository(session).list_active()
    system_prompt = _build_system_prompt(
        matches,
        doctors=doctors,
        default_language=resolved_settings.default_reply_language,
        clinic_phone_numbers=resolved_settings.clinic_phone_numbers,
        clinic_address=resolved_settings.clinic_address,
        clinic_work_hours=resolved_settings.clinic_work_hours,
        doctor_name=resolved_settings.doctor_name,
        doctor_specialty=resolved_settings.doctor_specialty,
        doctor_background=resolved_settings.doctor_background_text,
        appointment_book=(
            await _appointment_book(session, len(doctors))
            if resolved_settings.booking_enabled
            else None
        ),
        clinic_rules=await clinic_rules_for(session, get_current_tenant()),
    )
    provider = llm_provider or get_llm_provider()
    conversation: list[ChatMessage] = [
        *(history or []),
        ChatMessage(role="user", content=user_message),
    ]
    return await provider.generate(system_prompt, conversation)
