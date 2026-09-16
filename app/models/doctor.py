import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Doctor(Base):
    """A clinician a patient can be booked with.

    Comes from the Telegram side, where the table was created by raw SQL at
    application startup rather than by a migration — so its shape was
    whatever the last deploy happened to run. It is a real part of the
    unified schema, so it is declared here and created by the migration like
    every other table.

    Replaces app.services.appointment.DEFAULT_DOCTOR_NAME, which was a single
    hardcoded name standing in until multi-doctor support existed.
    """

    __tablename__ = "doctors"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    specialty: Mapped[str] = mapped_column(String(255), nullable=False)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Free text for now ("09:00 - 18:00"), carried over from the Telegram
    # side as-is. Structured hours (start, end, weekdays, breaks) are what
    # automatic slot generation actually needs — see the open question in the
    # merge plan. Kept free-form here so this migration does not force a
    # decision that belongs to the booking UI.
    working_hours: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
