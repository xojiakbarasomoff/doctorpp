"""Photos patients send: keep them, acknowledge them, tell the doctor.

A patient who sends a picture is usually sending a result -- an analysis, an
ultrasound report, a prescription -- and is waiting for a person to look at
it. Three things happen, in this order, and each is allowed to fail without
stopping the next:

1. The image is downloaded and stored. Instagram's attachment links are
   signed and expire, so a link alone would be a blank square by next week.
2. The patient is told "Hozir ko'rib beramiz", once per burst: three photos
   sent in a row are one acknowledgement, not three.
3. The doctor's Telegram gets the photo and who sent it.

The assistant never answers a photo itself. A medical image is the last thing
it should offer an opinion on (rule 3 of the answer prompt), and a reply that
pretended to have looked would be worse than none.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.message import MessageSender
from app.models.patient_media import PatientMedia
from app.models.user import User
from app.repositories.message import MessageRepository
from app.services.conversation import record_outbound_message, reply_context_for
from app.services.delivery import send_reply
from app.services.guardrail import reply_script

logger = logging.getLogger(__name__)

# Larger than any photo a phone sends, small enough that a stray video link
# cannot fill the database.
MAX_IMAGE_BYTES = 15 * 1024 * 1024

# One acknowledgement per burst of photos from the same conversation.
ACK_WINDOW_SECONDS = 120

ACKNOWLEDGEMENTS = {
    "uz-latn": "Hozir ko'rib beramiz.",
    "uz-cyrl": "Ҳозир кўриб берамиз.",
    "ru": "Сейчас посмотрим.",
}


async def download(url: str, http: httpx.AsyncClient | None = None) -> tuple[bytes, str] | None:
    """The image and its content type, or None if it cannot be had."""
    client = http or httpx.AsyncClient(timeout=30, follow_redirects=True)
    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        logger.warning("patient_media_download_failed", extra={"error": type(exc).__name__})
        return None
    finally:
        if http is None:
            await client.aclose()
    if response.is_error or len(response.content) > MAX_IMAGE_BYTES:
        logger.warning(
            "patient_media_download_refused",
            extra={"status_code": response.status_code, "size": len(response.content)},
        )
        return None
    content_type = response.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    return response.content, content_type


async def _patient_script(session: AsyncSession, conversation_id: uuid.UUID) -> str:
    """Which alphabet to acknowledge in: the one the patient last wrote in."""
    for message in reversed(await MessageRepository(session).list_recent(conversation_id, 20)):
        if message.sender == MessageSender.PATIENT and not message.content.startswith("📷"):
            return reply_script(message.content)
    return "uz-latn"


async def store(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    external_id: str | None,
    urls: list[str],
) -> list[PatientMedia]:
    stored: list[PatientMedia] = []
    for index, url in enumerate(urls):
        fetched = await download(url)
        media = PatientMedia(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            external_id=f"{external_id}:{index}" if external_id else None,
            source_url=url,
            content=fetched[0] if fetched else None,
            content_type=fetched[1] if fetched else None,
            size_bytes=len(fetched[0]) if fetched else None,
        )
        session.add(media)
        stored.append(media)
    await session.commit()
    return stored


async def acknowledge(
    session: AsyncSession,
    redis: Any,
    *,
    channel_id: uuid.UUID,
    conversation_id: uuid.UUID,
    recipient_external_id: str,
) -> None:
    """Tell the patient somebody will look, once per burst."""
    first = await redis.set(
        f"patient_media_ack:{conversation_id}", "1", ex=ACK_WINDOW_SECONDS, nx=True
    )
    if not first:
        return
    text = ACKNOWLEDGEMENTS[await _patient_script(session, conversation_id)]
    delivered = await send_reply(
        session,
        channel_id=channel_id,
        recipient_external_id=recipient_external_id,
        text=text,
        # The photo itself is the patient's latest message.
        last_user_message_at=datetime.now(UTC),
        reply_context=await reply_context_for(session, conversation_id),
    )
    if delivered is not None:
        await record_outbound_message(
            session,
            conversation_id=conversation_id,
            channel_type=delivered,
            text=text,
            sender=MessageSender.BOT,
        )
    await session.commit()


def caption(patient: User | None, count: int) -> str:
    who = "Bemor"
    if patient is not None:
        if patient.name and patient.username:
            who = f"{patient.name} (@{patient.username})"
        elif patient.username:
            who = f"@{patient.username}"
        elif patient.name:
            who = patient.name
    photos = "rasm" if count == 1 else f"{count} ta rasm"
    base = get_settings().public_base_url
    link = f"\n🖥 Dashboard → Rasm yuborganlar: {base.rstrip('/')}/admin/" if base else ""
    return f"📷 Bemor tomonidan {photos} yuborildi\n👤 {who}{link}"
