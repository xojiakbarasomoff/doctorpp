_UZBEK_CYRILLIC = frozenset("ўқғҳ")


def reply_script(user_message: str) -> str:
    """Which of "uz-latn", "uz-cyrl", "ru" to answer a fixed line in.

    A deliberately small rule rather than a language detector, for the few
    lines that are not written by the model (a voice note, a comment's
    public reply). Anything not Cyrillic is answered in Uzbek Latin.
    """
    lowered = user_message.lower()
    if any(letter in lowered for letter in _UZBEK_CYRILLIC):
        return "uz-cyrl"
    if any("Ѐ" <= character <= "ӿ" for character in lowered):
        return "ru"
    return "uz-latn"
