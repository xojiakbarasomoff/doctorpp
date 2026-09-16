import uuid
from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, ForeignKey, String, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class LeadStatus(StrEnum):
    """How far the call centre has got with a lead.

    Stored in English, like every other status column in this schema
    (conversations.status, appointments.status, messages.sender). The
    Telegram side stored these in Uzbek — `yangi`, `bog_lanildi`,
    `qabulga_yozildi`, `bekor_qilindi` — which reads better on screen but
    puts two languages in one database and makes every query and filter
    depend on which half of the product wrote the row. The Uzbek wording
    belongs in the interface, mapped from these values.

    ASSUMPTION, easily reversed: this is one of the open decisions in the
    merge plan. If the team prefers Uzbek values in the database, only this
    enum and one migration change.
    """

    NEW = "new"  # yangi
    CONTACTED = "contacted"  # bog'lanildi
    BOOKED = "booked"  # qabulga yozildi
    CANCELLED = "cancelled"  # bekor qilindi


class Lead(Base):
    """A patient who left a phone number for the clinic to call back.

    Comes from the Telegram side, but belongs to both bots: the Instagram
    assistant's system prompt already ends every reply by asking for a phone
    number and a convenient time to call (rule 7), and had nowhere to put the
    answer — it stayed in the conversation text and nothing acted on it.
    """

    __tablename__ = "leads"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    # Nullable: an operator can log a lead for someone who phoned the clinic
    # and has never messaged either bot, so there is no User row to attach.
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True
    )
    conversation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True, index=True
    )
    patient_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # What they were asking about, so whoever calls back does not open the
    # conversation cold.
    topic: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # When the patient said they could talk. Free text, because "kechqurun
    # 6 dan keyin" is what a patient actually writes — and asking for it is
    # half the point of rule 7: these patients are messaging precisely
    # because they cannot take a call right now.
    convenient_time: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, server_default=text(f"'{LeadStatus.NEW.value}'")
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), onupdate=func.now(), nullable=False
    )
