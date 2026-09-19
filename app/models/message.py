import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class DeliveryStatus(StrEnum):
    """What became of an outbound message."""

    SENT = "sent"
    FAILED = "failed"
    # Generated, and deliberately not sent: outside Instagram's messaging
    # window, or the channel is switched off.
    UNDELIVERABLE = "undeliverable"


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
    # Whether this message actually reached the patient.
    #
    # A reply that could not be delivered -- Instagram's 24-hour window
    # closed, the channel switched off, the API refusing -- used not to be
    # stored at all, so the clinic opened a chat, saw the patient's
    # questions and no answers, and could not tell a bot that said nothing
    # from a bot whose answer never arrived. It is stored either way now,
    # and this column says which happened.
    delivery_status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default="sent"
    )
    delivery_error: Mapped[str | None] = mapped_column(String(255), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("clock_timestamp()"), nullable=False
    )
    meta: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
