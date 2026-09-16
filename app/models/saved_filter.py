import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class SavedFilter(Base):
    """A named view of one of the dashboard's lists, saved by the clinic.

    "Bugungi Telegram qabullari", "Javobsiz lidlar" -- the questions a clinic
    asks its own data every morning. Until now each new one of those was a
    developer's job: a query parameter to add, a control to wire up, a
    deploy. The clinic knows which views it needs and should not have to ask.

    The filter itself is stored as JSON rather than as columns, because the
    interesting part is exactly the part that varies: appointments filter by
    day and status, leads by status, conversations by whether a human has
    taken over. Columns for the union of those would be mostly NULL and
    would need a migration every time a list gains a filter, which is the
    cost this is meant to remove. What is stored is a set of the query
    parameters the existing endpoints already accept, so a saved filter is
    replayable by the same API a person clicking the controls uses -- there
    is no second query path to keep correct.

    Deliberately not a saved *query*: no SQL, no field names, nothing the
    clinic could write that reaches the database directly. The API validates
    the parameters against the endpoint it is for and ignores what it does
    not recognise, so the worst a bad filter can do is show the wrong rows.
    """

    __tablename__ = "saved_filters"
    __table_args__ = (
        # Per clinic, per list. Two views called "Bugun" on different lists
        # are two different questions; two on the same list are a mistake,
        # and the second one would be unreachable in the interface anyway.
        UniqueConstraint("tenant_id", "resource", "name", name="uq_saved_filters_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    # Which list it belongs to: "appointments", "leads" or "conversations".
    resource: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    # Where it sits in the clinic's own list of views. An integer the
    # dashboard sorts by, so reordering never means renaming.
    position: Mapped[int] = mapped_column(nullable=False, server_default=text("0"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
