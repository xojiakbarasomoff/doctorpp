"""Loading a clinic's FAQ file into knowledge_base from inside the app.

scripts/ingest_faqs.py does the same thing from an operator's shell and stays
the right tool wherever one is available. This exists for the same reason
app.core.provisioning does: on a managed host the database is reachable only
from inside the cluster's private network, so the only process that can write
these rows is the application itself, driven by configuration.

Set SEED_FAQS_FROM to a path (e.g. "data/faqs.json") to arm it; leave it unset
and startup skips this entirely. Like PROVISION_TENANT_NAME, it is an
instruction to seed, not a description of the running system -- unset it once
the rows exist. Left armed, every redeploy re-embeds the whole file: one
batched embedding call rather than one per question (see
app.rag.embeddings), so the cost is small but not nothing, and it is time
spent before the web process starts accepting webhooks.

Re-running is safe: ingest_faqs matches on exact question text, so an edited
answer updates its row in place. An edited *question* leaves the old row
behind, since nothing in the file matches it any more.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.base import ChannelType
from app.core.config import Settings
from app.core.db import db_session
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.models.channel import Channel
from app.models.tenant import Tenant
from app.services.knowledge_base import FAQImport, ingest_faqs

logger = logging.getLogger(__name__)

CHANNEL_TYPE = ChannelType.INSTAGRAM

# Generous next to provisioning's 15s, because this waits on an embedding API
# round trip for the whole file as well as the database. Still bounded: this
# sits in the startup path, and an unreachable provider must not hold the port
# closed indefinitely.
_STARTUP_TIMEOUT_SECONDS = 60.0


class FaqSeedingError(Exception):
    """Raised when the FAQ file or its target tenant cannot be resolved."""


def load_faqs(path: Path) -> list[FAQImport]:
    """Read and validate the whole file before any embedding call is made, so
    a typo in the last row fails fast instead of surfacing after the rest have
    already been paid for.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FaqSeedingError(f"No such file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FaqSeedingError(f"{path} is not valid JSON: {exc}") from exc

    if not isinstance(raw, list):
        raise FaqSeedingError(
            f"{path} must contain a JSON array of FAQ objects, got {type(raw).__name__}."
        )

    try:
        return [FAQImport.model_validate(item) for item in raw]
    except Exception as exc:  # noqa: BLE001 - pydantic's own message is the useful part
        raise FaqSeedingError(f"Invalid FAQ entry in {path}: {exc}") from exc


async def _only_tenant_id(session: AsyncSession) -> uuid.UUID:
    """The one clinic in this database, when no channel exists to point at it."""
    tenants = list((await session.execute(select(Tenant.id))).scalars())
    if not tenants:
        raise FaqSeedingError("No clinic exists yet — set PROVISION_TENANT_NAME first.")
    if len(tenants) > 1:
        raise FaqSeedingError(f"{len(tenants)} clinics exist and none has a channel yet. Name one.")
    return tenants[0]


async def resolve_faq_tenant_id(
    session: AsyncSession, ig_account_id: str | None = None
) -> uuid.UUID:
    """The tenant whose knowledge base a FAQ file belongs to.

    Resolved from a channel rather than taken as a raw UUID: a mistyped UUID
    would seed a tenant nobody serves, and the rows would then be invisible to
    every query the webhook makes -- a failure that looks exactly like the
    FAQs never having been loaded at all.

    Any channel, not only an Instagram one. The knowledge base is shared by
    every platform a clinic answers on, and a deployment can perfectly well
    be live on Telegram while its Instagram token is still pending -- which
    is precisely the deployment whose FAQs have to load, since without them
    the assistant answers nothing but a refusal.

    Ambiguity is measured in *clinics*, not channels: one clinic reachable on
    both Instagram and Telegram is two channel rows and still one obvious
    answer.
    """
    query = select(Channel)
    if ig_account_id is not None:
        query = query.where(Channel.type == CHANNEL_TYPE, Channel.external_id == ig_account_id)

    channels = list((await session.execute(query)).scalars())

    if not channels:
        if ig_account_id is not None:
            raise FaqSeedingError(
                f"No {CHANNEL_TYPE} channel found with external_id={ig_account_id!r}."
            )
        # No channel yet, but the clinic itself may already exist: on a first
        # boot the tenant is provisioned before either platform's token has
        # been accepted, and a channel-only rule would leave that deployment
        # with an empty knowledge base -- the state where the assistant
        # answers nothing but a refusal. There is no id to mistype in this
        # path, which is what the channel indirection existed to catch, so
        # falling back to the tenant costs nothing and is still refused the
        # moment it would have to guess between two.
        return await _only_tenant_id(session)

    tenant_ids = {channel.tenant_id for channel in channels}
    if len(tenant_ids) > 1:
        ids = ", ".join(sorted(str(tenant_id) for tenant_id in tenant_ids))
        raise FaqSeedingError(f"{len(tenant_ids)} clinics have channels ({ids}). Name one.")
    return channels[0].tenant_id


async def seed_faqs(
    session: AsyncSession,
    faqs: list[FAQImport],
    ig_account_id: str | None = None,
    *,
    reembed_existing: bool = False,
) -> tuple[uuid.UUID, int]:
    """Ingest `faqs` for the resolved tenant and commit. Returns the tenant and
    how many rows were written, for the caller to log or print.
    """
    tenant_id = await resolve_faq_tenant_id(session, ig_account_id)
    token = set_current_tenant(tenant_id)
    try:
        rows = await ingest_faqs(session, faqs, reembed_existing=reembed_existing)
        await session.commit()
    finally:
        reset_current_tenant(token)
    return tenant_id, len(rows)


async def seed_faqs_if_configured(settings: Settings) -> None:
    """Load SEED_FAQS_FROM into the knowledge base when it is set.

    No-op when unconfigured, and never fatal: a seeding failure must not stop
    the web process from serving, since continuing to accept Meta's
    deliveries is worth more than refusing to start over a knowledge base
    someone can reload afterwards. The assistant answers without FAQs; it
    cannot answer at all if the webhook is down.
    """
    configured = settings.seed_faqs_from
    if configured is None:
        return

    path = Path(configured)

    async def _run() -> None:
        faqs = load_faqs(path)
        if not faqs:
            logger.warning("faq_seeding_skipped_empty_file path=%s", path)
            return
        async with db_session() as session:
            tenant_id, count = await seed_faqs(session, faqs, settings.provision_ig_account_id)
        # WARNING, not INFO: "this deployment reloaded its knowledge base" is
        # worth finding later without having to widen a log filter.
        logger.warning("faq_seeding_complete tenant_id=%s rows=%s path=%s", tenant_id, count, path)

    try:
        await asyncio.wait_for(_run(), timeout=_STARTUP_TIMEOUT_SECONDS)
    except TimeoutError:
        logger.error("faq_seeding_timed_out path=%s seconds=%s", path, _STARTUP_TIMEOUT_SECONDS)
    except FaqSeedingError as exc:
        # Configuration mistakes (missing file, ambiguous channel) get their
        # message rather than a stack trace -- the message is the whole point.
        logger.error("faq_seeding_failed path=%s error=%s", path, exc)
    except Exception:
        logger.exception("faq_seeding_failed path=%s", path)
