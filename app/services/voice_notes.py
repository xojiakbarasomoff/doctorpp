"""Voice messages, answered by the one line the clinic wants said.

The assistant cannot hear a recording. Asked to answer one anyway it writes
something plausible about a message nobody has listened to, which is the
worst of both: the patient believes they have been understood, and nobody
has read what they sent. So a voice note gets a fixed reply instead -- the
administrator does not listen to these, the doctor does -- in whichever
alphabet the patient has been writing in.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import MessageSender
from app.repositories.message import MessageRepository
from app.services.guardrail import reply_script

REPLIES = {
    "uz-latn": (
        "Men administratorman, ovozli xabarlarga doktorning o'zlari javob beradilar. "
        "Savolingizni yozib qoldirsangiz, men ham yordam bera olaman."
    ),
    "uz-cyrl": (
        "Мен администраторман, овозли хабарларга докторнинг ўзлари жавоб берадилар. "
        "Саволингизни ёзиб қолдирсангиз, мен ҳам ёрдам бера оламан."
    ),
    "ru": (
        "Я администратор, на голосовые сообщения отвечает сам доктор. "
        "Если напишете вопрос текстом, я тоже смогу помочь."
    ),
}


async def reply_for(session: AsyncSession, conversation_id: uuid.UUID) -> str:
    """The fixed reply, in the script this patient has been writing in."""
    for message in reversed(await MessageRepository(session).list_recent(conversation_id, 20)):
        if message.sender == MessageSender.PATIENT and not message.content.startswith(
            ("🎤", "📷", "💬")
        ):
            return REPLIES[reply_script(message.content)]
    return REPLIES["uz-latn"]
