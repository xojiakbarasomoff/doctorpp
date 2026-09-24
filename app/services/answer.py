"""Turning a patient's message into a reply.

The model is given the clinic's persona (written in the dashboard, see
app.services.persona), the facts that go with it, and the conversation, and
writes the reply itself. Nothing it says is checked or rewritten here.

What this module adds to the persona is the part that is not the clinic's to
edit: the markers the rest of the system reads ([[BOOK:...]], [[CALLBACK:...]],
[[DOCTOR]])
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
from app.services import medical_safety
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

_DOCTOR_CONTRACT = """\
When a person from the clinic has to answer this patient themselves -- you \
told them the doctor will reply or you will pass their question to the \
doctor, the facts you were given do not answer what they asked and a person \
must, or they insist on the doctor personally -- end your reply with \
[[DOCTOR]]. The patient never sees it; it puts the conversation at the top of \
the clinic's inbox until somebody answers. Do not write it for anything you \
answered yourself, for a booking, or for the usual advice to come in for an \
examination."""

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

_LANGUAGE_CONTRACT = (
    'Reply only in the language and alphabet given as "Javob tili" above: '
    "Russian to a patient writing Russian, Uzbek Cyrillic to one writing Uzbek "
    "in Cyrillic, Uzbek Latin to one writing Latin. Never mix alphabets in one "
    "reply. When the patient's language is unclear, reply in {default_language}."
)


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
    prompt += "\n\n" + _DOCTOR_CONTRACT
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


_CORRECTION = """

YOUR LAST REPLY WAS REJECTED
It {what} — rule 3 above says you are not a medical professional and must \
never do that, whatever the patient asked and however they insisted. Write \
the same answer again, correctly: warmly decline the medical part, in the \
patient's own language, and point them to an appointment. Everything else \
about the reply -- what else it answered, any [[BOOK:...]], \
[[CALLBACK:...]] or [[DOCTOR]] marker it carried -- stays the same. Send only the \
corrected reply."""


async def _enforce(
    reply: str,
    *,
    provider: LLMProvider,
    system_prompt: str,
    conversation: Sequence[ChatMessage],
    script: str,
) -> str:
    """The reply, or a safe line in its place if it prescribes or diagnoses.

    Checked by pattern, not by asking the model to grade itself -- see
    app.services.medical_safety for why. Two tries at most: the first is
    free in the common case where nothing was wrong, and the second costs
    one call only when the pattern actually fired. A reply that fails
    twice is replaced rather than sent, and both attempts are logged at
    ERROR so the clinic can see what its assistant tried to say.
    """
    violation = medical_safety.check(reply)
    if violation is None:
        return reply

    logger.error(
        "medical_safety_rewriting",
        extra={"category": violation.category, "matched": violation.matched, "reply": reply[:300]},
    )
    instruction = _CORRECTION.format(what=medical_safety.DESCRIPTIONS[violation.category])
    try:
        second = await provider.generate(system_prompt + instruction, list(conversation))
    except Exception:
        logger.exception("medical_safety_rewrite_failed")
        return medical_safety.SAFE_REPLIES[script]

    remaining = medical_safety.check(second)
    if remaining is None:
        return second

    logger.error(
        "medical_safety_unfixable",
        extra={"category": remaining.category, "matched": remaining.matched, "reply": second[:300]},
    )
    return medical_safety.SAFE_REPLIES[script]


# How many of the assistant's own last replies are shown back to it, and how
# much of each. Two is enough to catch "rahmat" answered the same way twice
# running; the whole conversation is already in `history` for anything more,
# and this is only meant to save the model recalling its own voice from
# further up the same scrollback.
_MAX_RECENT_REPLIES = 2
_MAX_RECENT_REPLY_CHARS = 220


def _recent_replies(history: Sequence[ChatMessage] | None) -> tuple[str, ...]:
    """The assistant's own last couple of turns, oldest first."""
    if not history:
        return ()
    own = [turn["content"] for turn in history if turn.get("role") == "assistant"]
    return tuple(text[:_MAX_RECENT_REPLY_CHARS] for text in own[-_MAX_RECENT_REPLIES:])


async def generate_answer(
    session: AsyncSession,
    user_message: str,
    embedding_provider: EmbeddingProvider | None = None,
    llm_provider: LLMProvider | None = None,
    settings: Settings | None = None,
    history: Sequence[ChatMessage] | None = None,
    patient: Profile | None = None,
    long_gap: bool = False,
    situation: str | None = None,
) -> str:
    """Ask the model to answer the patient, with the clinic's facts in hand.

    `situation` is added to the system prompt as it stands, for a message
    that did not arrive as an ordinary chat line (a comment under a post).

    `history` is the conversation's earlier turns, oldest first. Callers get
    it from app.services.conversation.context_for_reply, which already
    excludes the messages being answered right now, so appending
    user_message here cannot repeat them.

    `long_gap` is whether it has been 24+ hours since anything was last said
    in this conversation (app.services.conversation.hours_since_last_contact)
    -- see PatientState.long_gap for what it changes.
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
        # Whether there was anything to send as history: context_for_reply
        # has already dropped the trailing patient turns being answered
        # right now, so an empty list here means, reliably, that this is
        # the first thing this patient has ever said.
        continuing=bool(history),
        recent_replies=_recent_replies(history),
        long_gap=long_gap,
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
    if situation:
        system_prompt += "\n\n" + situation
    provider = llm_provider or get_llm_provider()
    conversation: list[ChatMessage] = [
        *(history or []),
        ChatMessage(role="user", content=user_message),
    ]
    reply = await provider.generate(system_prompt, conversation)
    return await _enforce(
        reply,
        provider=provider,
        system_prompt=system_prompt,
        conversation=conversation,
        script=script,
    )
