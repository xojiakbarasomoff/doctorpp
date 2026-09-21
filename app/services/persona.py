"""The assistant's voice, kept where the clinic can edit it.

What the bot says is decided by three things, and each has one home:

* the persona -- role, style, banned phrases, limits -- is text the clinic
  writes in the dashboard (tenants.settings["assistant_prompt"]);
* the example conversations are text the clinic writes beside it
  (tenants.settings["assistant_examples"]);
* the facts -- address, telephone, hours, doctors, prices -- are the
  clinic's settings, its doctors and its knowledge base, and reach the model
  through the sections below.

Nothing in the persona is enforced in code. It is a request to the model,
and the clinic changes it the way it would brief a new colleague: by editing
the text and reading the next replies. What code does own is the contract
the rest of the system depends on -- the [[BOOK]] and [[CALLBACK]] markers
and the appointment book -- which is appended in app.services.answer and is
not editable here, because a clinic that deletes a sentence from its persona
must not be able to stop bookings from being written down.

The persona is a template. Four tokens are replaced by generated sections,
each carrying its own heading so an empty one leaves nothing behind:

    {klinika_malumotlari}   the clinic's details, doctors
    {bilimlar_bazasi}       the knowledge-base rows matching this message
    {suhbat_holati}         what the clinic already knows about this patient
    {namunalar}             the example conversations

A token the persona does not contain is appended at the end, so a clinic
that rewrites the text from scratch and forgets one still gives the model
its facts. A brace that is not one of the four is left alone: the persona is
free text, and "{ism}" typed by somebody is not a template error.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

MAX_PROMPT_LENGTH = 20_000
MAX_EXAMPLES_LENGTH = 30_000

KEY_PROMPT = "assistant_prompt"
KEY_EXAMPLES = "assistant_examples"

DEFAULT_PROMPT = """\
# ROL
Siz urologiya klinikasining Instagram administratorisiz. Ismingiz: Madina.
Sizning ishingiz: bemorni qabulga yozish, narx / ish vaqti / manzil aytish,
shifokorga yo'naltirish. Siz shifokor emassiz.

# YOZISH USLUBI
- Telefondan yozayotgan odamdek yozing: qisqa, oddiy. Odatda 1-2 gap.
- Bemor qaysi yozuvda yozsa (lotin yoki kirill), ayni shu yozuvda javob bering.
  Rus tilida yozsa - rus tilida.
- Bir xabarda faqat bitta savol bering.
- Salomlashuvni faqat birinchi xabarda ayting, keyin hech qachon takrorlamang.
- Bemor aytgan ma'lumotni (ism, yosh, shikoyat, kun) qayta so'ramang.
- Ro'yxat, sarlavha, qalin shrift ishlatmang. Emoji va "!" ni kam ishlating.
- Bemorning savolini qaytarib aytmang, to'g'ridan-to'g'ri javob bering.

# TAQIQLANGAN IBORALAR
"Murojaatingiz uchun rahmat", "Sizga yordam berishdan mamnunman",
"Albatta!", "Ajoyib savol", "Qo'shimcha savollaringiz bo'lsa murojaat qiling",
"Sizga qanday yordam bera olaman?" (birinchi xabardan keyin),
"Men sun'iy intellekt sifatida...".

# CHEGARALAR
- Tashxis qo'ymang. Dori, doza, muolaja tavsiya qilmang.
  Shikoyat aytilsa: "Buni ko'rmasdan aytib bo'lmaydi, qabulga yozib qo'yaymi?"
- Quyidagi ma'lumotlarda yo'q narsani o'ylab topmang. Bilmasangiz:
  "Aniqlab, hozir yozaman."
- Shoshilinch belgilar (kuchli og'riq, qon ketishi, harorat, nafas qisishi):
  "Iltimos, darhol 103 ga qo'ng'iroq qiling yoki shifoxonaga boring."
- Agar bemor to'g'ridan-to'g'ri "siz odammisiz yoki botmisiz?" deb so'rasa,
  rost javob bering: "Men klinikaning virtual yordamchisiman, hozir sizni
  administratorimizga ulayman." Boshqa holatlarda rolni saqlang.
- Jahldor yoki murakkab holatda suhbatni haqiqiy xodimga o'tkazing.

{klinika_malumotlari}

{bilimlar_bazasi}

{suhbat_holati}

{namunalar}"""

DEFAULT_EXAMPLES = """\
1. Salomlashuv, keyin maqsad

Bemor: Assalomu alaykum
Admin: Va alaykum assalom. Eshitaman 😊
Bemor: Savolim bor edi
Admin: Ha, aytavering.

2. Salomlashuv, keyin xizmat

Bemor: Salom doktor
Admin: Va alaykum assalom. Qanday yordam bera olaman?
Bemor: Tez bo'shanish bo'yicha murojaat qilmoqchi edim
Admin: Tushundim. Yoshiz nechida?

3. Bir savolni ikki marta so'ramang

Bemor: Assalomu alaykum. Yoshim 34 da, Andijondanman.
Admin: Va alaykum assalom. Tushundim.
Bemor: Menda shu muammo bor edi.
Admin: Ha, tushundim. Oldin bu bo'yicha tekshiruvdan o'tgansizmi?

Yoshini qayta so'ramang: u allaqachon aytilgan.

4. Narx, Direct'da aytmasdan

Bemor: Qancha turadi?
Admin: Narxni telefon orqali aniqlashtirib beramiz.
Bemor: Hozir aytib bo'lmaydimi?
Admin: Aniq summani qo'ng'iroq qilib bilib olsangiz bo'ladi.

Birinchi gapni so'zma-so'z takrorlamang.

5. Qabulga yozilish

Bemor: Qabulga yozilmoqchiman
Admin: Albatta. Qaysi shifokor qabuliga yozilmoqchisiz?
Bemor: Urologga
Admin: Mayli. Sizga qaysi kun qulay?
Bemor: Juma
Admin: Xo'p, juma kuni bo'yicha yozuvni aniqlashtirib beraman.

6. Suhbatni yakunlash

Bemor: Rahmat, tushundim.
Admin: Arzimaydi 😊

Har safar avtomatik "Yana savollaringiz bormi?" deb qo'shmang.

7. Bemor noaniq yozganda

Bemor: Malumot kerak edi
Admin: Albatta. Qaysi xizmat bo'yicha ma'lumot kerak?
Bemor: Urologiya bo'yicha
Admin: Qaysi masala bo'yicha murojaat qilgandingiz?

8. Bemor takrorlaganda

Bemor: Manzilingiz qayerda?
Admin: Manzilni yuboraman.
Bemor: Qayerda edi o'zi?
Admin: Ha, manzilni nazarda tutyapsiz. Hozir yuboraman.

Ikkinchi javob takrorni sezdi, shablonni to'liq qayta yubormadi.

9. Bemor doktorning o'zi yozayotganini so'raganda

Bemor: Adminmisiz yoki doktorning o'zimi?
Admin: Sizga klinika bo'yicha yordam beraman. Qaysi masalada ma'lumot kerak edi?

Klinikaning oshkor qilish qoidasiga amal qiling; o'zingizni hech qachon doktor deb ko'rsatmang.

10. Qisqa, tabiiy gaplar

Bemor: Alo
Admin: Eshitaman.
Bemor: Bir savolim bor edi
Admin: Marhamat.
Bemor: Keyinroq yozsam bo'ladimi?
Admin: Albatta."""

_EXAMPLES_NOTE = (
    "Bu namunalar faqat yozish uslubi va ohangi uchun. Ulardagi gaplarni "
    "ko'chirmang; narx, manzil, vaqt kabi faktlarni faqat yuqoridagi "
    "klinika ma'lumotlari va bilimlar bazasidan oling."
)

_TOKEN = re.compile(r"\{(klinika_malumotlari|bilimlar_bazasi|suhbat_holati|namunalar)\}")
_SECTION_ORDER = ("klinika_malumotlari", "bilimlar_bazasi", "suhbat_holati", "namunalar")

_SCRIPT_LABELS = {
    "uz-latn": "o'zbek, lotin yozuvi",
    "uz-cyrl": "o'zbek, kirill yozuvi",
    "ru": "rus, kirill yozuvi",
}


@dataclass(frozen=True)
class Persona:
    """What the clinic wrote in the dashboard, or the defaults."""

    prompt: str = DEFAULT_PROMPT
    examples: str = DEFAULT_EXAMPLES


@dataclass(frozen=True)
class ClinicFacts:
    """Everything the model is told about the clinic, already resolved.

    Resolved by the caller (the dashboard's settings first, the deployment's
    environment second) so this module never has to know where a value came
    from.
    """

    name: str | None = None
    address: str | None = None
    landmark: str | None = None
    phone_numbers: str | None = None
    work_hours: str | None = None
    doctors: Sequence[tuple[str, str, str]] = ()  # (name, specialty, hours)
    lead_doctor: str | None = None
    lead_doctor_background: str | None = None
    rules: Sequence[str] = ()


@dataclass(frozen=True)
class PatientState:
    """What the backend is sure of about this patient.

    Only what is stored and reliable. Age, complaint and the day they want
    are said in the conversation, which the model reads in full; a field
    here that was usually "unknown" would only invite it to ask again.
    """

    name: str | None = None
    phone: str | None = None
    script: str | None = None


def from_settings(settings: dict[str, Any] | None) -> Persona:
    """The persona a clinic's stored settings describe.

    A missing or blank prompt is the default, never an empty prompt: a bot
    with no instructions at all is worse than one with the clinic's old
    ones. Examples differ -- blank is a choice ("no examples"), and only a
    missing value falls back.
    """
    stored = settings or {}
    prompt = stored.get(KEY_PROMPT)
    examples = stored.get(KEY_EXAMPLES)
    return Persona(
        prompt=prompt.strip() if isinstance(prompt, str) and prompt.strip() else DEFAULT_PROMPT,
        examples=examples.strip() if isinstance(examples, str) else DEFAULT_EXAMPLES,
    )


def clinic_section(facts: ClinicFacts) -> str:
    lines: list[str] = []
    if facts.name:
        lines.append(f"Klinika: {facts.name}")
    if facts.address:
        lines.append(f"Manzil: {facts.address}")
    if facts.landmark:
        lines.append(f"Mo'ljal: {facts.landmark}")
    if facts.phone_numbers:
        lines.append(f"Telefon: {facts.phone_numbers}")
    if facts.work_hours:
        lines.append(f"Ish vaqti: {facts.work_hours}")
    if facts.lead_doctor:
        lines.append(f"Shifokor: {facts.lead_doctor}")
        if facts.lead_doctor_background:
            lines.append("Shifokor haqida:")
            lines.append(facts.lead_doctor_background)
    if facts.doctors:
        # Six clinicians who all work the same hours produced six identical
        # tails, and a model copying the block back wrote the hours six
        # times in one reply. When the roster agrees with itself the hours
        # are a fact about the clinic, and are said once.
        shared = {hours for _, _, hours in facts.doctors}
        if len(shared) == 1 and len(facts.doctors) > 1:
            lines.append(f"Shifokorlar (hammasi bir xil vaqtda ishlaydi: {shared.pop()}):")
            lines.extend(f"- {name} — {specialty}" for name, specialty, _ in facts.doctors)
        else:
            lines.append("Shifokorlar:")
            lines.extend(
                f"- {name} — {specialty} — {hours}" for name, specialty, hours in facts.doctors
            )
    if facts.rules:
        lines.append("Klinikaning qo'shimcha qoidalari (ularga amal qiling):")
        lines.extend(f"- {rule}" for rule in facts.rules)
    if not lines:
        return ""
    return "# KLINIKA MA'LUMOTLARI\n" + "\n".join(lines)


def knowledge_section(rows: Sequence[tuple[str, str]]) -> str:
    """The knowledge-base rows that matched this message, or an honest gap."""
    if not rows:
        return (
            "# BILIMLAR BAZASI\n"
            "Bu savolga mos yozuv topilmadi. Narx, manzil, vaqt kabi faktlarni "
            "o'ylab topmang."
        )
    body = "\n\n".join(f"Savol: {question}\nJavob: {answer}" for question, answer in rows)
    return f"# BILIMLAR BAZASI\n{body}"


def state_section(state: PatientState) -> str:
    known = [
        f"{label}: {value}"
        for label, value in (
            ("Ism", state.name),
            ("Telefon", state.phone),
            ("Til/yozuv", _SCRIPT_LABELS.get(state.script or "", state.script)),
        )
        if value
    ]
    if not known:
        return ""
    return "# SUHBAT HOLATI (tizim to'ldirgan)\n" + " | ".join(known)


def examples_section(examples: str) -> str:
    if not examples.strip():
        return ""
    return f"# NAMUNA YOZISHMALAR\n{_EXAMPLES_NOTE}\n\n{examples.strip()}"


def render(
    persona: Persona,
    *,
    facts: ClinicFacts,
    knowledge: Sequence[tuple[str, str]],
    state: PatientState,
) -> str:
    """The persona with its four sections in place."""
    sections = {
        "klinika_malumotlari": clinic_section(facts),
        "bilimlar_bazasi": knowledge_section(knowledge),
        "suhbat_holati": state_section(state),
        "namunalar": examples_section(persona.examples),
    }
    used: set[str] = set()

    def fill(match: re.Match[str]) -> str:
        used.add(match.group(1))
        return sections[match.group(1)]

    text = _TOKEN.sub(fill, persona.prompt)
    missing = [sections[name] for name in _SECTION_ORDER if name not in used and sections[name]]
    # A blank left where an empty section was, collapsed: an unset state or
    # an examples box the clinic emptied must not leave a hole in the prompt.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return "\n\n".join([text, *missing])
