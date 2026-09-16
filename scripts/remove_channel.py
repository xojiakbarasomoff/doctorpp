r"""Delete one channel and everything that hangs off it.

Written for retiring the Instagram test account once the clinic's real
account is connected: the app finds a channel by (type, external_id), so
swapping the connected Instagram account leaves the old channel row behind,
still active, still owning its test patients.

Nothing here cascades on its own. users.channel_id is a plain foreign key
with no ON DELETE, so deleting the channel row alone fails on the first test
patient that ever messaged it; conversations, messages, leads and
appointments hang off those users in turn. So the rows are removed
innermost-first, in one transaction.

The clinic itself is never touched -- only this channel's own rows. A clinic
answering on both Instagram and Telegram keeps its knowledge base, its
doctors, its operators and the other channel's patients.

This deletes patient conversation history and cannot be undone. Prefer
deactivating instead, which needs no script and is reversible:

    UPDATE channels SET is_active = false WHERE id = '<uuid>';

Usage, from the instagram/ directory -- first without CONFIRM to see the
counts, then with it to actually delete:

    python scripts/remove_channel.py <channel-uuid>
    CONFIRM=yes python scripts/remove_channel.py <channel-uuid>
"""

import asyncio
import os
import sys
import uuid
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.sql.elements import ColumnElement

from app.core.db import db_session
from app.models.appointment import Appointment
from app.models.channel import Channel
from app.models.conversation import Conversation
from app.models.lead import Lead
from app.models.message import Message
from app.models.user import User


async def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <channel-uuid>")
    try:
        channel_id = uuid.UUID(sys.argv[1])
    except ValueError:
        sys.exit(f"Not a UUID: {sys.argv[1]!r}")

    confirmed = os.environ.get("CONFIRM") == "yes"

    async with db_session() as session:
        channel = await session.get(Channel, channel_id)
        if channel is None:
            # Idempotent: already gone is the state this script is for.
            print(f"No channel with id={channel_id}. Nothing to do.")
            return

        user_ids = select(User.id).where(User.channel_id == channel_id).scalar_subquery()
        conversation_ids = (
            select(Conversation.id).where(Conversation.user_id.in_(user_ids)).scalar_subquery()
        )

        async def count(model: type[Any], where: ColumnElement[bool]) -> int:
            return await session.scalar(select(func.count()).select_from(model).where(where)) or 0

        # Leads and appointments hang off this channel by either foreign key,
        # and both are nullable: create_appointment supports a booking with a
        # patient_name and no user at all. Matching on user_id alone would
        # leave those rows behind, and the conversations delete below would
        # then abort the whole transaction on a foreign-key violation --
        # after the operator had typed CONFIRM=yes on counts that never
        # mentioned them.
        lead_where = Lead.user_id.in_(user_ids) | Lead.conversation_id.in_(conversation_ids)
        appointment_where = Appointment.user_id.in_(user_ids) | Appointment.conversation_id.in_(
            conversation_ids
        )

        counts = {
            "messages": await count(Message, Message.conversation_id.in_(conversation_ids)),
            "conversations": await count(Conversation, Conversation.user_id.in_(user_ids)),
            "leads": await count(Lead, lead_where),
            "appointments": await count(Appointment, appointment_where),
            "users": await count(User, User.channel_id == channel_id),
        }

        print(
            f"channel_id={channel.id} type={channel.type} "
            f"external_id={channel.external_id!r} is_active={channel.is_active}"
        )
        for name, n in counts.items():
            print(f"  {name}: {n}")

        if not confirmed:
            print("\nDry run -- nothing deleted. Re-run with CONFIRM=yes to delete these rows.")
            return

        # Innermost first, so every delete leaves the graph consistent even
        # if the transaction is inspected mid-flight.
        await session.execute(delete(Message).where(Message.conversation_id.in_(conversation_ids)))
        await session.execute(delete(Lead).where(lead_where))
        await session.execute(delete(Appointment).where(appointment_where))
        await session.execute(delete(Conversation).where(Conversation.user_id.in_(user_ids)))
        await session.execute(delete(User).where(User.channel_id == channel_id))
        await session.delete(channel)
        await session.commit()

    print("\nDeleted.")


if __name__ == "__main__":
    asyncio.run(main())
