import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


# Patients contacting a tenant's clinic over a channel (staff accounts live in Operator).
class User(Base):
    __tablename__ = "users"
    # One row per patient per channel. This is what makes first contact
    # safe to handle concurrently: two webhook deliveries for a patient's
    # first two bubbles both find no row and both insert, and the database
    # — not a lucky interleaving — decides there is one patient. See
    # app.services.conversation._get_or_create_user.
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "channel_id", "external_id", name="uq_users_tenant_channel_external"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    channel_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("channels.id"), nullable=False, index=True
    )
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # The handle the patient is known by on their platform -- "asomov" on
    # Instagram, the @name on Telegram -- without the @.
    #
    # The dashboard shows this instead of external_id, which is a number an
    # operator has never seen and cannot match to a person. Nullable and
    # staying that way: it is fetched after the fact (app.services.profile),
    # the lookup can fail, and an Instagram account need not have one at all.
    # A patient without a username is still a patient the clinic must see.
    username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    phone: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # The language this patient asked to be answered in, when they have said
    # so: "rus tilida yozing", "по-русски". Stored on the patient rather
    # than worked out from each message, because a patient who asks for
    # Russian and then sends "ok" was being answered in Uzbek again.
    preferred_language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    # Whether this patient may use the bot's own admin commands. From the
    # Telegram side, and a different thing from Operator: an Operator logs
    # into the dashboard, this is a patient chatting to the bot who is also
    # clinic staff.
    is_admin: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("false"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
