import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Index, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class SuggestionKind(StrEnum):
    RULE = "rule"  # a standing instruction, onto strict_rules
    FAQ = "faq"  # a question and answer, into the knowledge base


class SuggestionStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class Suggestion(Base):
    """A fix the nightly review proposes, for a person to accept or reject.

    The review reads the day's conversations and writes these; nothing it
    writes reaches a patient until somebody at the clinic accepts it. That
    is the whole safety of the feature: an assistant answering patients
    about their health must not rewrite its own rules unseen.
    """

    __tablename__ = "suggestions"
    __table_args__ = (Index("ix_suggestions_tenant_status", "tenant_id", "status"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, server_default=text(f"'{SuggestionStatus.PENDING.value}'")
    )
    # What went wrong, in a sentence the clinic reads.
    problem: Mapped[str] = mapped_column(Text, nullable=False)
    # The fix: a rule, or a question and its answer.
    rule_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Where it was seen: [{"conversation_id": ..., "quote": ...}].
    evidence: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    decided_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("operators.id", ondelete="SET NULL"), nullable=True
    )
