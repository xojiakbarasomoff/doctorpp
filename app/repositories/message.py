import uuid
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import select

from app.core.tenant_context import get_current_tenant
from app.models.conversation import Conversation
from app.models.message import Message, MessageSender
from app.repositories.base import BaseRepository, CrossTenantAccessError


class MessageRepository(BaseRepository[Message]):
    """Messages have no tenant_id column — they're scoped through their
    conversation. Every method here explicitly joins/checks
    conversations.tenant_id against the current tenant, since the automatic
    filtering in TenantScopedRepository (which relies on a direct tenant_id
    column) doesn't apply to this table.
    """

    model = Message

    async def get(self, id_: uuid.UUID) -> Message | None:
        stmt = (
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(Message.id == id_, Conversation.tenant_id == get_current_tenant())
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def list_for_conversation(self, conversation_id: uuid.UUID) -> Sequence[Message]:
        stmt = (
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Message.conversation_id == conversation_id,
                Conversation.tenant_id == get_current_tenant(),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def list_recent(self, conversation_id: uuid.UUID, limit: int) -> Sequence[Message]:
        """The last `limit` messages of a conversation, oldest first.

        Selected newest-first in SQL (so Postgres uses the index and reads
        `limit` rows, not the whole conversation) then reversed here, since
        every caller wants them in the order they were said — an LLM's
        message list, or a transcript on screen.
        """
        stmt = (
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Message.conversation_id == conversation_id,
                Conversation.tenant_id == get_current_tenant(),
            )
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(reversed(result.scalars().all()))

    async def last_inbound_at(self, conversation_id: uuid.UUID) -> datetime | None:
        """When this conversation last heard from the patient, or None if it
        never has.

        This is what a platform's reply window is measured against (see
        ChannelAdapter.delivery_block_reason) — a real recorded timestamp,
        rather than the "assume now" placeholder the worker used before
        inbound messages were persisted at all.
        """
        stmt = (
            select(Message.created_at)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Message.conversation_id == conversation_id,
                Conversation.tenant_id == get_current_tenant(),
                Message.sender == MessageSender.PATIENT,
            )
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def create(self, **values: Any) -> Message:
        conversation_id = values.get("conversation_id")
        if conversation_id is None:
            raise ValueError("conversation_id is required to create a Message")

        owns_conversation = await self.session.execute(
            select(Conversation.id).where(
                Conversation.id == conversation_id,
                Conversation.tenant_id == get_current_tenant(),
            )
        )
        if owns_conversation.scalar_one_or_none() is None:
            raise CrossTenantAccessError("conversation_id does not belong to the current tenant")

        return await self._create(**values)
