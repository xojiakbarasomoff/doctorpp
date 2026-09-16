"""Put a name to the patient the dashboard is showing.

Instagram delivers a message with the sender's id and nothing else, so until
now a conversation in the dashboard was headed "17841467434701445". The
operator reading that screen cannot recognise a patient by it, cannot search
for one, and cannot match the row to the person telephoning them -- which
makes the busiest screen in the clinic the one that identifies nobody.

The account's own messaging permission is enough to ask Instagram for the
sender's handle, and one lookup per patient is enough for good: the handle is
stored on the patient row, and a patient who already has one is never looked
up again.

Everything here is best-effort by construction. A message must be answered
whether or not the lookup works, so every failure path ends in the patient
keeping their id as a label -- which is what the dashboard showed anyway.
"""

import logging
import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.instagram.client import InstagramClient, get_instagram_client
from app.core.encryption import DecryptionError, decrypt
from app.channels.base import ChannelType
from app.repositories.channel import ChannelRepository
from app.repositories.user import UserRepository

logger = logging.getLogger(__name__)


def normalise_username(raw: str | None) -> str | None:
    """A handle as it should be stored: no "@", no surrounding space, or None.

    Operators paste "@asomov" and platforms send "asomov"; storing one shape
    is what lets the search box match either.
    """
    if raw is None:
        return None
    cleaned = raw.strip().lstrip("@").strip()
    return cleaned or None


async def remember_username(
    session: AsyncSession, *, user_id: uuid.UUID, username: str | None
) -> None:
    """Store a handle that arrived with the message itself.

    Telegram puts the sender's username in the update, so that channel needs
    no lookup at all -- it only needs somewhere to put it.
    """
    cleaned = normalise_username(username)
    if cleaned is None:
        return
    user = await UserRepository(session).get(user_id)
    if user is None or user.username == cleaned:
        return
    user.username = cleaned
    await session.flush()


async def ensure_instagram_username(
    session: AsyncSession,
    *,
    channel_id: uuid.UUID,
    user_id: uuid.UUID,
    client: InstagramClient | None = None,
) -> str | None:
    """Look the handle up once, the first time this patient is seen.

    Returns the handle now on the row, or None. Never raises: it is called
    on the path that answers a patient, and no label is worth failing that
    for.
    """
    repo = UserRepository(session)
    user = await repo.get(user_id)
    if user is None or user.username:
        return user.username if user is not None else None

    channel = await ChannelRepository(session).get(channel_id)
    if channel is None or channel.type != ChannelType.INSTAGRAM:
        return None

    try:
        credentials = decrypt(channel.credentials)
    except DecryptionError:
        # Delivery raises on this, loudly and deliberately, and will do so a
        # few lines later in the same job. Saying it twice would only make
        # the real failure harder to find.
        return None

    try:
        username = await (client or get_instagram_client()).fetch_username(
            access_token=credentials, igsid=user.external_id
        )
    except Exception:  # noqa: BLE001 - a label is never worth failing a reply
        logger.warning("username_lookup_raised", extra={"user_id": str(user_id)}, exc_info=True)
        return None

    cleaned = normalise_username(username)
    if cleaned is None:
        return None
    user.username = cleaned
    await session.flush()
    logger.info("username_resolved", extra={"user_id": str(user_id), "username": cleaned})
    return cleaned
