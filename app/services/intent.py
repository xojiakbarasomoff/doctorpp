"""What the patient is asking for, decided before the model is asked anything.

The assistant used to infer this from the prompt, and it inferred it the
same way every time: somebody mentioned a symptom, so it asked for their
name, then their number, then offered times. A person who wrote "prostatada
muammo bor" wanted to be told something, not enrolled in a form.

Rules first, and for most messages rules are the whole answer: "rahmat" is
thanks in every conversation that has ever contained it, "bekor qilmoqchiman"
is a cancellation, a bare "12:00" is a time. What the rules cannot place is
left as UNKNOWN, and the caller lets the model handle it as ordinary
conversation -- which is the safe direction to be wrong in. Nothing here
books, cancels or answers anything; it only says which flow the turn belongs
to.

The patient's own words decide the intent. The state decides what that
intent means: "ha" is a confirmation only when something was asked.
"""

import re
from enum import StrEnum

from app.models.conversation_state import ConversationState, FlowStatus
from app.services import when as when_service
from app.services.conversation_signals import looks_like_a_greeting


class Intent(StrEnum):
    GREETING = "greeting"
    GENERAL_QUESTION = "general_question"
    DOCTOR_INFO = "doctor_info"
    CLINIC_INFO = "clinic_info"
    MEDICAL_QUESTION = "medical_question"
    BOOKING_REQUEST = "booking_request"
    BOOKING_DATE = "booking_date"
    BOOKING_TIME = "booking_time"
    BOOK_NEW = "book_new"
    RESCHEDULE_EXISTING = "reschedule_existing"
    CANCEL_REQUEST = "cancel_request"
    CANCEL_CONFIRM = "cancel_confirm"
    BOOKING_CONFIRM = "booking_confirm"
    EXISTING_BOOKING_QUERY = "existing_booking_query"
    THANKS = "thanks"
    GOODBYE = "goodbye"
    REFERENCE_PREVIOUS = "reference_previous"
    CLARIFICATION = "clarification"
    INJECTION_ATTEMPT = "injection_attempt"
    UNKNOWN = "unknown"


_THANKS = re.compile(
    r"^\s*(?:katta\s+)?(?:rahmat|raxmat|raxmet|tashakkur|спасибо|рахмат|thanks|thank you)"
    r"[\s!.,)]*$",
    re.IGNORECASE,
)
_GOODBYE = re.compile(
    r"^\s*(?:xayr|xop\s+bo'?ldi|salomat\s+bo'?ling|ko'rishguncha|до свидания|пока)[\s!.,]*$",
    re.IGNORECASE,
)

_CANCEL = re.compile(
    r"bekor\s*qil|qabulni\s*bekor|бекор\s*қил|отмен(?:ить|и|яю)|аннулир",
    re.IGNORECASE,
)
_RESCHEDULE = re.compile(
    r"o'?zgartir|ko'?chir|boshqa\s+(?:kun|vaqt)ga|surib|перенес|поменя",
    re.IGNORECASE,
)
_WANTS_TO_BOOK = re.compile(
    r"qabulga\s+yoz|yozilmoqchi|yozib\s+qo|navbat\s+ol|band\s+qil|yozsangiz|yozing"
    r"|қабулга\s+ёз|ёзилмоқчи|навбат"
    r"|запис(?:ать|аться|ите|ываюсь)|на\s+при[её]м",
    re.IGNORECASE,
)
_ANOTHER_BOOKING = re.compile(
    r"yana\s+bir|boshqa\s+kunga\s+ham|ikkinchi\s+marta|yana\s+qabulga|ещ[её]\s+один",
    re.IGNORECASE,
)
_EXISTING_QUERY = re.compile(
    r"(?:qabul\w*|navbat\w*|yozilgan\w*)[^.?!]{0,30}(?:bormi|bormidi|qachon|nechida|qaysi)"
    r"|qachon\s+yozilganman|yozilganmanmi|qabulim\s+bor"
    r"|когда\s+(?:я\s+)?запис|есть\s+ли\s+запис|моя\s+запис",
    re.IGNORECASE,
)

_REFERENCE_PREVIOUS = re.compile(
    r"tepada|yuqorida|oldin\s+(?:ayt|yoz|berdim|gapir)|allaqachon\s+(?:ayt|yoz|berdim)"
    r"|aytdimku|yozdimku|gaplashdikku|berdimku"
    r"|выше\s+(?:напис|сказ)|уже\s+(?:говорил|писал|сказал|дал)",
    re.IGNORECASE,
)

_DOCTOR_INFO = re.compile(
    r"doktor\w*\s+(?:haqida|kim|necha|qayer)|shifokor\w*\s+(?:haqida|kim|necha|qayer)"
    r"|tajriba|ta'?lim|o'?qigan|sertifikat|malaka"
    r"|о\s+врач|стаж|образовани",
    re.IGNORECASE,
)
_CLINIC_INFO = re.compile(
    r"manzil|qayerda|qayerdasiz|ish\s*vaqt|nechida\s+(?:ochil|yopil)|nechigacha|soat\s+nechada"
    r"|narx|qancha\s+tur|to'?lov|karta|parkovka|mashina\s+qo'?yish"
    r"|адрес|где\s+наход|график|во\s+сколько|сколько\s+стоит|цена|оплат",
    re.IGNORECASE,
)
_MEDICAL = re.compile(
    r"og'?ri|ogri|shishib|qichi|yonib|qon\s+kel|siyish|siydik|buyrak|prostat|jinsiy|erektil"
    r"|analiz|tahlil|natija|\bpsa\b|gormon|uzi|tosh\b|infeksiya|muammo\s+bor"
    r"|оғри|буйрак|простат|анализ|боль|болит|проблем",
    re.IGNORECASE,
)

# A patient trying to read the machinery rather than talk to the clinic.
_INJECTION = re.compile(
    r"ignore\s+(?:all\s+)?previous|system\s*prompt|internal\s+(?:rules?|instruction)"
    r"|developer\s+(?:mode|message)|prompt\s*injection|jailbreak"
    r"|qoida\w*\s+(?:nima|qanday|ro'?yxat|nechta|ayting|ko'?rsat)"
    r"|nechta\s+qoida|qolgan\s+\d+\s*ta\s+qoida|ko'?rsatmalar\w*\s+(?:nima|qanday|ayting)"
    r"|систем\w*\s+промпт|внутренн\w+\s+правил|покажи\s+(?:свои\s+)?правил"
    r"|\[\[|\bBOOK:|\bCALLBACK:",
    re.IGNORECASE,
)

_YES = re.compile(r"^\s*(?:ha|xa|ha'?a|mayli|bo'?ladi|tasdiqla\w*|да|давай|ок|ok)[\s!.,]*$", re.I)
_NO = re.compile(r"^\s*(?:yo'?q|yoq|kerak\s+emas|нет|не\s+надо)[\s!.,]*$", re.IGNORECASE)


def classify(
    message: str,
    *,
    state: ConversationState | None = None,
    now: object = None,
) -> Intent:
    """The intent of one message, read in the light of where the flow is.

    Order matters. An answer to a question the assistant just asked is that
    answer first and anything else second: a patient who replies "ha" to
    "shu qabulni bekor qilaymi?" has confirmed a cancellation, whatever else
    the word might mean elsewhere.
    """
    text = message.strip()
    if not text:
        return Intent.UNKNOWN

    if _INJECTION.search(text):
        return Intent.INJECTION_ATTEMPT

    status = FlowStatus(state.status) if state is not None else FlowStatus.IDLE

    # 1. The answer to the question that is open.
    if status is FlowStatus.AWAITING_CANCEL_CONFIRM and _YES.match(text):
        return Intent.CANCEL_CONFIRM
    if status is FlowStatus.AWAITING_CONFIRMATION and _YES.match(text):
        # "Ha" is a confirmation only when something is waiting for one --
        # the clinic's own rule, and the reason a stray "ok" cannot book
        # anybody.
        return Intent.BOOKING_CONFIRM
    if status is FlowStatus.AWAITING_CONFIRMATION and _NO.match(text):
        return Intent.BOOKING_REQUEST
    if status is FlowStatus.AWAITING_CANCEL_CHOICE and not _CANCEL.search(text):
        return Intent.CANCEL_REQUEST  # they are naming which one
    if status in {FlowStatus.AWAITING_DATE, FlowStatus.AWAITING_TIME, FlowStatus.COLLECTING}:
        found = when_service.read(text)
        if found.at is not None:
            return Intent.BOOKING_TIME
        if found.day is not None:
            return Intent.BOOKING_DATE

    # 2. What the patient plainly said.
    if _CANCEL.search(text):
        return Intent.CANCEL_REQUEST
    if _RESCHEDULE.search(text):
        return Intent.RESCHEDULE_EXISTING
    if _EXISTING_QUERY.search(text):
        return Intent.EXISTING_BOOKING_QUERY
    if _REFERENCE_PREVIOUS.search(text):
        return Intent.REFERENCE_PREVIOUS
    if _THANKS.match(text):
        return Intent.THANKS
    if _GOODBYE.match(text):
        return Intent.GOODBYE
    if _WANTS_TO_BOOK.search(text):
        return Intent.BOOK_NEW if _ANOTHER_BOOKING.search(text) else Intent.BOOKING_REQUEST

    # 3. A time on its own, with no flow open. A bare time is an answer to
    # something even when the state has been lost; a bare date is not, and
    # waits until after the subjects below -- "PSA 6.2 chiqdi" is a
    # frightened patient, not the sixth of February.
    found = when_service.read(text)
    if found.at is not None:
        return Intent.BOOKING_TIME

    # 4. Subjects, in the order that decides what the reply is made of.
    if _DOCTOR_INFO.search(text):
        return Intent.DOCTOR_INFO
    if _CLINIC_INFO.search(text):
        return Intent.CLINIC_INFO
    if _MEDICAL.search(text):
        return Intent.MEDICAL_QUESTION

    if found.day is not None and len(text) <= 40:
        return Intent.BOOKING_DATE
    if looks_like_a_greeting(text) and len(text) <= 40:
        return Intent.GREETING

    return Intent.UNKNOWN


# Which intents are about a booking at all. Everything else must not put a
# patient into the three questions, which is the failure this exists to end.
BOOKING_INTENTS = frozenset(
    {
        Intent.BOOKING_REQUEST,
        Intent.BOOKING_DATE,
        Intent.BOOKING_TIME,
        Intent.BOOKING_CONFIRM,
        Intent.BOOK_NEW,
        Intent.RESCHEDULE_EXISTING,
    }
)

CANCELLATION_INTENTS = frozenset({Intent.CANCEL_REQUEST, Intent.CANCEL_CONFIRM})
