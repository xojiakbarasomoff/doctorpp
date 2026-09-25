import re
from collections.abc import Sequence

from app.rag.llm import ChatMessage

_UZBEK_CYRILLIC = frozenset("ўқғҳ")

# Letters Uzbek Cyrillic does not have. Evidence, not proof: Uzbeks write
# loanwords with them ("консультацияга") and misspell with them ("томищаман").
_RUSSIAN_ONLY = frozenset("ыщь")

# Most Uzbeks typing Cyrillic on a Russian keyboard write "к", "х", "у" for
# "қ", "ҳ", "ў", so "Рахмат" or "Канча туради" carry no Uzbek-only letter at
# all. Common whole words decide those instead.
_RUSSIAN_WORDS = frozenset(
    re.findall(
        r"\S+",
        "и в не на что как я вы мне меня с по это есть можно где когда "
        "здравствуйте привет пожалуйста спасибо сколько стоит цена хорошо "
        "нет ли уже или его её их мой моя вас нас запись "
        "записаться приём прием день добрый утро вечер хочу надо нужно "
        "какой какая какое какие почему зачем сейчас сегодня завтра здесь там "
        "очень болит болят подскажите скажите работаете работает ваш ваша "
        "записать запишите тоже только если",
    )
)
_UZBEK_WORDS = frozenset(
    re.findall(
        r"\S+",
        "салом ассалому ассалом алейкум рахмат рахмад рахмет ха хо йук йок "
        "бор борми йукми керак канча нарх нархи качон кайерда каерда кайерга "
        "мен ман сиз биз улар шу бу нима нега яхши илтимос ака опа ука "
        "дохтир булади булдими буладими килиш килса килиб эди экан "
        "ва лекин учун билан кейин хозир эртага бугун кеча соат манзил "
        "хам мумкин мумкинми шунга ухшаш ухшаган дори узи узим кандай канака "
        "духтир духтур доктор салом солом асалому саволим бор еди эди керакми",
    )
)
_UZBEK_SUFFIX = re.compile(r"(лар|ларни|ларга|дан|даги|нинг|миз|сиз|ман|ган|япти|ябди|мокда)$")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def _cyrillic_language(lowered: str) -> str:
    if any(letter in lowered for letter in _UZBEK_CYRILLIC):
        return "uz-cyrl"
    uzbek = russian = 0
    for word in _WORD.findall(lowered):
        if word in _UZBEK_WORDS:
            uzbek += 2
        elif word in _RUSSIAN_WORDS:
            russian += 2
        elif any(letter in word for letter in _RUSSIAN_ONLY):
            russian += 1
        elif len(word) > 4 and _UZBEK_SUFFIX.search(word):
            uzbek += 1
    # A tie goes to Uzbek: this inbox is in Uzbekistan, and Russian writing
    # almost never gets through a sentence without "ы", "ь" or a word above.
    return "ru" if russian > uzbek else "uz-cyrl"


def reply_script(user_message: str) -> str:
    """Which of "uz-latn", "uz-cyrl", "ru" to answer in.

    A deliberately small rule rather than a language detector. Anything not
    Cyrillic is answered in Uzbek Latin.
    """
    lowered = user_message.lower()
    if any("Ѐ" <= character <= "ӿ" for character in lowered):
        return _cyrillic_language(lowered)
    return "uz-latn"


# Whether a message has enough letters to be evidence of an alphabet:
# "93 444 444" and "11:00" are the two most common messages this inbox gets,
# and "ok" from a patient writing Cyrillic is not a switch to Latin.
_MIN_LETTERS = 3


def _letter_count(message: str) -> int:
    return sum(len(w) for w in _WORD.findall(message))


def conversation_script(history: Sequence[ChatMessage] | None, user_message: str) -> str:
    """The alphabet this conversation is being held in.

    Per conversation rather than per message: a patient who wrote Latin
    throughout and then sent a telephone number was answered in Cyrillic.
    The patient's own words decide it, newest first; what the assistant
    wrote does not, or one reply in the wrong alphabet would justify the
    next.
    """
    written = [_patient_words(user_message)]
    if history:
        written.extend(
            _patient_words(m["content"]) for m in reversed(history) if m.get("role") == "user"
        )
    for message in written:
        if _letter_count(message) >= _MIN_LETTERS:
            return reply_script(message)
    return reply_script(written[0])


# Labels the system puts in the transcript in its own words -- "🎤 Ovozli
# xabar", "📷 Rasm yubordi", "💬 Izoh: ..." -- are not the patient writing,
# and their Latin once had a patient writing Cyrillic answered in Latin.
_SYSTEM_LABEL = re.compile(r"^\s*(🎤|📷)")
_COMMENT_PREFIX = re.compile(r"^\s*💬\s*Izoh:\s*")


def _patient_words(message: str) -> str:
    if _SYSTEM_LABEL.match(message):
        return ""
    return _COMMENT_PREFIX.sub("", message)


# Latin words a reply in Cyrillic may still carry as they are: service
# names patients know in Latin. Handles, links and markers are removed
# before counting, and so are short capitalised abbreviations (UZI, PSA).
_LATIN_ALLOWED = frozenset(
    {"telegram", "whatsapp", "instagram", "direct", "zoom", "google", "youtube", "online"}
)
_NOT_WORDS = re.compile(r"@\w+|https?://\S+|\[\[[^\]]*\]\]|\b[A-Z]{2,5}\b")
_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
# Words in the other alphabet a reply may carry before it is a reply in the
# wrong one: a stray loanword in a long answer, not a sentence -- and not one
# word of five, which is how "Qaysi телефон рақамга" looked.
_FOREIGN_SHARE = 0.15


def _word_alphabets(text: str) -> tuple[int, int]:
    """How many words are written in Latin, and how many in Cyrillic."""
    latin_words = cyrillic_words = 0
    for word in text.split():
        latin = len(_LATIN.findall(word))
        cyrillic = len(_CYRILLIC.findall(word))
        if latin + cyrillic < 3:
            continue
        if latin > cyrillic:
            lowered = word.lower()
            if any(lowered.startswith(name) for name in _LATIN_ALLOWED):
                continue
            latin_words += 1
        else:
            cyrillic_words += 1
    return latin_words, cyrillic_words


def reply_fits(reply: str, script: str) -> bool:
    """Whether the assistant's reply is written in the language and alphabet
    it was told to use."""
    text = _NOT_WORDS.sub(" ", reply)
    latin, cyrillic = _word_alphabets(text)
    words = latin + cyrillic
    if not words:
        return True
    foreign = cyrillic if script == "uz-latn" else latin
    if foreign >= 2 or foreign / words > _FOREIGN_SHARE:
        return False
    if script == "uz-latn":
        return True
    lowered = text.lower()
    if script == "ru":
        return not any(letter in lowered for letter in _UZBEK_CYRILLIC) and (
            _cyrillic_language(lowered) == "ru" or len(lowered.split()) < 4
        )
    return _cyrillic_language(lowered) == "uz-cyrl"
