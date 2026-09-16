import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class MessageSender(StrEnum):
    """Who produced a message, as stored in messages.sender.

    Named rather than left as bare strings because three different writers
    fill this column — the inbound webhook, the bot's own reply, and (once
    the operator takeover UI exists) a human at the dashboard — and a reader
    telling a bot reply apart from an operator's is the whole point of the
    column.
    """

    PATIENT = "patient"
    BOT = "bot"
    OPERATOR = "operator"


class Message(Base):
    # Append-only: messages are never updated or deleted, only inserted.
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=False, index=True
    )
    sender: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    channel: Mapped[str] = mapped_column(String(50), nullable=False)
    # clock_timestamp(), not now(): now() is transaction_timestamp() in
    # Postgres, so every message written inside one transaction gets an
    # identical value and the transcript's order collapses onto the tiebreak
    # — a random UUID. That reorders a conversation, which is not a
    # cosmetic problem when the ordered transcript is what the next reply is
    # generated from. clock_timestamp() reads the real clock per row.
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("clock_timestamp()"), nullable=False
    )
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
