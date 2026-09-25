"""The one thing that is checked, not just asked for: the assistant must
never prescribe, dose, diagnose, or pass itself off as a clinician.

Everything else about how the assistant talks -- its tone, its length, how
it greets, what it offers -- is the clinic's own text in the dashboard
persona, and nothing here touches it. This module is narrow on purpose: it
is the one place where a request is a prompt (rule 3, in the persona) and an
answer is also a fact checked by code, because losing a patient is a bad
day and a patient who acted on invented medical advice is not recoverable.

Why code and not another model call asking "did you just prescribe
something": a model asked to grade its own answer is grading the same
judgment that produced it, and the literature on this is consistent --
self-assessment under-reports exactly the failures worth catching. A pattern
match has no such bias, costs no extra call, and adds no latency to the
common case where nothing was wrong. See app.services.answer._enforce for
how a violation is handled: the same model, told plainly what it did and
asked once to say it without doing that again, and a fixed safe line if it
still does it a second time -- the same layered shape ("cheap check first,
escalate only when the cheap check fires") the rest of the industry uses for
this, and the same "false positives cost less than false negatives" choice
healthcare guardrails are built around.

The patterns are deliberately narrow, matched against what is instruction
rather than what is topic: a legitimate refusal talks about "dori" or
"antibiotik" as a class ("antibiotik kerakmi, buni urolog hal qiladi") and
never reaches for a brand, a dose, or a verb telling the patient to take
something -- and that sentence must never be caught here.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Violation:
    category: str
    matched: str


# A time of day, which this inbox writes more often than any other number.
# Masked before the dose check runs, and masked rather than dropped so the
# sentence keeps its shape: "sizni bugun 16:20ga yozdim" once read as a dose
# of twenty grams, because the unit pattern had no way to know a clock from
# a quantity. That is a whole class of false positive, closed here once
# rather than chased one report at a time.
_CLOCK = re.compile(r"\b\d{1,2}[:.]\d{2}\b")

# [[BOOK:...]] and [[CALLBACK:...]]: machinery the patient never sees, built
# partly from their own words -- a complaint, a reason for the visit.
# Judging the assistant by what a patient wrote about themselves is how a
# reply ends up withheld for the patient having described their own
# symptoms.
_MARKER = re.compile(r"\[\[[^\]]*\]?\]?")


def _inspectable(reply: str) -> str:
    """The reply as this check should read it: the clock excused, the
    machinery gone."""
    return _CLOCK.sub("VAQT", _MARKER.sub(" ", reply))


# A named medicine. Matched by the endings drug names are built from, in
# Uzbek Latin and Uzbek Cyrillic, so the list does not have to be kept
# current with a pharmacy's shelves -- a brand nobody has typed here before
# still ends in "-itsin" or "-atsin".
_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        "dori_nomi",
        r"\b\w{3,}(?:tsillin|itsin|mitsin|floksatsin|floxacin|atsin|siklin"
        r"|prazol|zosin|sartan|pril\b|statin|profen|olol\b|tidin|ceftriakson)\w*"
        r"|\b\w{3,}(?:циллин|ицин|мицин|флоксацин|ациклин|циклин"
        r"|празол|зозин|сартан|прил|статин|профен|олол|тидин)\w*",
    ),
    # A dose: a number next to a unit. The clock is already masked above, so
    # a booking time can no longer read as a quantity.
    (
        "doza",
        r"\b\d+([.,]\d+)?\s*(mg|ml|mkg|mcg|мг|мл|мкг|г|гр)\b"
        r"|\b\d+\s*(tabletka|kapsula|таблетк\w*|капсул\w*)\b",
    ),
    # A schedule, in digits or in words, tied to the verb that makes it an
    # instruction: "kuniga ikki mahal" on its own is nothing, "kuniga ikki
    # mahal iching" is a dosing schedule.
    (
        "jadval",
        r"kuniga\s*(?:\d+|bir|ikki|uch|to['’ʻ]?rt|besh)\s*(?:mahal|marta)"
        r"\s*(?:ich\w*|qabul|surt\w*)"
        r"|(?:\d+|bir|ikki|uch|to['’ʻ]?rt|besh)\s*(?:mahal|marta)\s*(?:ich\w*|qabul|surt\w*)"
        # The same, in Uzbek Cyrillic -- a schedule does not have to stand
        # beside a drug word to be one; "кунига 2 марта ичинг" is a dosing
        # instruction on its own.
        r"|кунига\s*(?:\d+|бир|икки|уч|тўрт|беш)\s*(?:маҳал|марта)"
        r"\s*(?:ич\w*|қабул|сурт\w*)"
        r"|(?:\d+|бир|икки|уч|тўрт|беш)\s*(?:маҳал|марта)\s*(?:ич\w*|қабул|сурт\w*)"
        r"|\d+\s*раз[ау]?\s*в\s*день",
    ),
    # Being told to take something. The window between the verb and the
    # medicine word is wide enough to survive a clause in the middle
    # ("dorini -- bu muhim -- ichib turing").
    (
        "ichish",
        r"\b(?:ich(?:ing|ib|sangiz)|qabul\s*qiling|surt\w+)\w*"
        r"[^.!?]{0,80}\b(?:dori|tabletka|antibiotik|preparat|kapsula)\w*"
        r"|\b(?:dori|tabletka|antibiotik|preparat|kapsula)\w*"
        r"[^.!?]{0,80}\b(?:ich(?:ing|ib|sangiz)|qabul\s*qiling)"
        r"|\b(?:ичинг|ичиб|қабул\s*қилинг)\w*[^.!?]{0,80}"
        r"\b(?:дори|таблетка|антибиотик|препарат)\w*"
        r"|\b(?:дори|таблетка|антибиотик|препарат)\w*[^.!?]{0,80}"
        r"\b(?:ичинг|ичиб|қабул\s*қилинг)"
        r"|\b(?:принимайте|пейте|выпейте|примите)\b[^.!?]{0,80}"
        r"\b(?:лекарств|таблетк|антибиотик|препарат)"
        r"|\b(?:лекарств|таблетк|антибиотик|препарат)\w*[^.!?]{0,80}"
        r"\b(?:принимайте|пейте|выпейте|примите)",
    ),
    # Prescribing without an imperative: "bunga antibiotik buyuriladi" is a
    # recommendation wearing the passive voice.
    (
        "buyurish",
        r"\b(?:buyur(?:iladi|aman|amiz)|tavsiya\s*qil(?!ma)(?!a\s+olma)\w*)"
        r"[^.!?]{0,80}\b(?:dori|antibiotik|preparat|tabletka)\w*"
        r"|\b(?:dori|antibiotik|preparat|tabletka)\w*[^.!?]{0,80}"
        r"\b(?:buyur(?:iladi|aman|amiz)|tavsiya\s*qil(?!ma)(?!a\s+olma)\w*)",
    ),
    # Telling the patient what they have. A condition has to be named AND
    # attributed to them -- "sizda X bor" is a diagnosis, "X bilan urolog
    # shug'ullanadi" is the assistant saying who treats it, and that
    # difference is exactly what "prostatit bilan urolog" must not trip.
    (
        "tashxis",
        r"\b(?:sizda|sizning)\b[^.!?]{0,60}"
        r"\b(?:prostatit|sistit|uretrit|pielonefrit|nefrit|adenoma|kista"
        r"|infeksiya|saraton|o['’ʻ]?sma|tosh(?:i|lar)?)\w*"
        r"|\b(?:у\s*вас|у\s*Вас)\b[^.!?]{0,60}"
        r"\b(?:простатит|цистит|уретрит|пиелонефрит|нефрит|аденом|киста"
        r"|инфекц|рак|опухол|камен)\w*"
        # Certainty about a condition is a diagnosis however it is phrased.
        r"|(?:100\s*%|\baniq\b|\balbatta\b|\bshubhasiz\b|точно|определённо)"
        r"[^.!?]{0,30}\b(?:prostatit|sistit|infeksiya|saraton|o['’ʻ]?sma"
        r"|adenoma|простатит|цистит|инфекц|рак|опухол)\w*",
    ),
    # Telling the patient they need not come. The most harmful thing this
    # assistant could say: in urology a painless symptom is exactly the one
    # that must not be waited out.
    (
        "kelmasa_boladi",
        r"\bjiddiy\s*emas\b|\bo['’ʻ]?zi\s*(?:o['’ʻ]?tib\s*ketadi|tuzaladi|yo['’ʻ]?qoladi)\b"
        r"|\b(?:shifokorga|qabulga|klinikaga)[^.!?]{0,30}"
        r"\b(?:shart\s*emas|hojat\s*yo['’ʻ]?q|bormasangiz|kelmasangiz)"
        r"|\bничего\s*страшного\b|\bсамо\s*пройдёт\b|\bсамо\s*пройдет\b"
        r"|\bне\s*обязательно\b[^.!?]{0,30}\b(?:приходить|врач)",
    ),
    # Treatment to try at home: techniques and exercises. A real reply walked
    # a patient through "stop-start", "squeeze" and Kegel counts -- which is
    # treating them, whatever it was called.
    (
        "uy_muolajasi",
        r"\bstop[\s-]*start|\bsqueeze\b|\bkegel\w*|стоп[\s-]*старт|\bкегел\w*"
        r"|\b(?:usul|mashq|texnika)\w*[^.!?]{0,50}"
        r"\b(?:sinab\s*ko['’ʻ]?ring|qo['’ʻ]?llang|bajaring|o['’ʻ]?rganing)"
        r"|\b(?:sinab\s*ko['’ʻ]?ring|qo['’ʻ]?llang|bajaring)[^.!?]{0,50}"
        r"\b(?:usul|mashq|texnika)\w*"
        r"|\b(?:усул|машқ|машк|техника)\w*[^.!?]{0,50}"
        r"\b(?:синаб\s*кўринг|қўлланг|бажаринг)"
        r"|\b(?:упражнени|метод|техник)\w*[^.!?]{0,50}\b(?:попробуйте|делайте|выполняйте)",
    ),
    # Recommending a vitamin or supplement for the complaint.
    (
        "vitamin",
        r"\bvitamin\w*[^.!?]{0,80}\b(?:foyda|yordam\s*ber|yaxshila|iching|qabul)"
        r"|\bвитамин\w*[^.!?]{0,80}\b(?:фойда|ёрдам\s*бер|яхшила|ичинг|қабул|помога|улучша)",
    ),
    # Writing a prescription.
    ("retsept", r"\bretsept\w*\s*(?:yoz|ber)|\bрецепт\w*\s*(?:выпиш|напиш)"),
    # Claiming to be the clinician. The persona says who is answering --
    # the front desk, or the doctor's administrator -- and this is the
    # sentence that would contradict it in the patient's own inbox.
    (
        "ozini_shifokor_deb_korsatish",
        r"\bmen\s*(?:shifokor|doktor|vrach)\w*(?:man|ман)\b"
        r"|\bмен\s*(?:шифокор|доктор)\w*(?:ман)\b"
        r"|\bя\s*врач\b",
    ),
)

_COMPILED = tuple((name, re.compile(pattern, re.IGNORECASE)) for name, pattern in _PATTERNS)

# What each category means, in English, for the instruction the model is
# rewritten against -- see app.services.answer._enforce. Plain enough that a
# model told this once does not need the pattern explained to it.
DESCRIPTIONS: dict[str, str] = {
    "dori_nomi": "named a specific medicine",
    "doza": "gave a dose or quantity of a medicine",
    "jadval": "gave a dosing schedule -- how many times a day to take something",
    "ichish": "told the patient to take a medicine",
    "buyurish": "prescribed or recommended a medicine",
    "tashxis": "told the patient what condition they have",
    "kelmasa_boladi": "told the patient they do not need to come in, or that it is not serious",
    "uy_muolajasi": "gave treatment to try at home -- a technique or an exercise",
    "vitamin": "recommended a vitamin or supplement",
    "retsept": "wrote a prescription",
    "ozini_shifokor_deb_korsatish": "claimed to be the doctor or a clinician",
}

# Said instead, when a second try still breaks the rule. Fixed text, in the
# same shape as the other lines this inbox falls back to when the model
# cannot be trusted with a reply: it cannot mirror the patient's exact
# words, and it does not need to -- this is the one reply where being
# wrong costs more than being generic.
SAFE_REPLIES: dict[str, str] = {
    "uz-latn": (
        "Buni sizga men aytolmayman — dori va davolash haqida faqat shifokor, "
        "ko'rikdan keyin gapira oladi. Sizni qabulga yozib qo'yaymi?"
    ),
    "uz-cyrl": (
        "Буни сизга мен айтолмайман — дори ва даволаш ҳақида фақат шифокор, "
        "кўрикдан кейин гапира олади. Сизни қабулга ёзиб қўяйми?"
    ),
    "ru": (
        "Этого я вам сказать не могу — о лекарствах и лечении говорит только "
        "врач, после осмотра. Записать вас на приём?"
    ),
}


def check(reply: str) -> Violation | None:
    """The first rule this reply breaks, or None.

    Checked against the reply as the patient will read it -- machinery
    stripped, the clock excused -- and returned as soon as one pattern
    matches, in the fixed order above: which one fired first only matters
    for the log line, not for what happens next.
    """
    inspected = _inspectable(reply)
    for category, pattern in _COMPILED:
        match = pattern.search(inspected)
        if match is not None:
            return Violation(category=category, matched=match.group(0)[:80])
    return None
