from app.rag.llm import ChatMessage
from app.services.language import conversation_script, reply_script


def test_the_alphabet_is_the_conversations_own_not_the_last_message() -> None:
    """A telephone number and a time have no alphabet. Deciding per message
    answered "93 444 444" in Cyrillic in the middle of a Latin conversation.
    """
    history: list[ChatMessage] = [
        {"role": "user", "content": "Man kelasi seshanba 10:00ga qabulga yozilmoqchiman"},
        {"role": "assistant", "content": "Исмингизни ёзинг"},
        {"role": "user", "content": "Asadbek Risqiyev"},
    ]

    assert conversation_script(history, "93 444 444") == "uz-latn"
    assert conversation_script(history, "11:00") == "uz-latn"
    # What the assistant wrote does not count, or one reply in the wrong
    # alphabet would justify the next one.
    assert conversation_script(history, "Buyrak ogrigi") == "uz-latn"


def test_a_patient_who_writes_cyrillic_is_answered_in_cyrillic() -> None:
    history: list[ChatMessage] = [{"role": "user", "content": "буйрагим оғрияпти"}]

    assert conversation_script(history, "16:20") == "uz-cyrl"
    assert conversation_script(None, "Здравствуйте") == "ru"


def test_anything_unrecognised_is_uzbek_latin() -> None:
    assert reply_script("hello") == "uz-latn"
    assert reply_script("") == "uz-latn"
