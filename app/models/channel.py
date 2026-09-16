import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Channel(Base):
    __tablename__ = "channels"
    __table_args__ = (UniqueConstraint("type", "external_id", name="uq_channels_type_external_id"),)

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False, index=True
    )
    type: Mapped[str] = mapped_column(String(50), nullable=False)  # "telegram" | "instagram"
    # The platform's own identifier for this account (IG page/account id,
    # Telegram bot id, etc). This is what an inbound webhook gives us to
    # figure out which tenant it belongs to — see
    # app.services.tenant_resolution.resolve_instagram_channel(). Unique
    # per type, not globally: different platforms have separate id
    # namespaces, so two channels of different types could coincidentally
    # share an id string.
    external_id: Mapped[str] = mapped_column(String(255), nullable=False)
    # Always ciphertext at rest, never plaintext — encrypted with
    # app.core.encryption.encrypt (Fernet, keyed by ENCRYPTION_KEY) before
    # insert, decrypted with app.core.encryption.decrypt at the one place
    # that needs the real value (app.workers.tasks._send_reply). This
    # includes the "no real token yet" placeholder sentinels (see
    # app.channels.instagram.client.is_placeholder_credential) — nothing
    # ever goes into this column unencrypted, so there's no plaintext
    # special case to accidentally leave unprotected.
    credentials: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default=text("true"))
    # Per-channel settings that belong to the platform rather than to the
    # clinic: a Telegram webhook secret, the ids the bot treats as admins,
    # a Mini App URL. Deliberately not folded into Tenant.settings — one
    # clinic can run two accounts on the same platform, and each needs its
    # own values.
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, server_default=text("'{}'::jsonb")
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=text("now()"), nullable=False
    )
