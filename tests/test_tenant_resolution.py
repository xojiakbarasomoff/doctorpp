import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.channels.base import ChannelType
from app.models.tenant import Tenant
from app.services.tenant_resolution import resolve_channel, resolve_instagram_channel
from tests.conftest import Seed


async def test_resolves_known_ig_account_to_its_tenant_and_channel(
    db_session: AsyncSession, seed: Seed
) -> None:
    resolved = await resolve_instagram_channel(db_session, seed.a.channel.external_id)

    assert resolved is not None
    assert resolved.tenant_id == seed.tenant_a.id
    # The channel id matters as much as the tenant: it is what the reply is
    # later sent over, instead of whichever channel a list query returned
    # first.
    assert resolved.channel_id == seed.a.channel.id
    assert resolved.channel_type == ChannelType.INSTAGRAM


async def test_unknown_ig_account_returns_none(db_session: AsyncSession, seed: Seed) -> None:
    assert await resolve_instagram_channel(db_session, "no-such-account") is None


async def test_two_tenants_with_different_ig_accounts_do_not_cross_resolve(
    db_session: AsyncSession, seed: Seed
) -> None:
    resolved_a = await resolve_instagram_channel(db_session, seed.a.channel.external_id)
    resolved_b = await resolve_instagram_channel(db_session, seed.b.channel.external_id)

    assert resolved_a is not None and resolved_b is not None
    assert resolved_a.tenant_id == seed.tenant_a.id
    assert resolved_b.tenant_id == seed.tenant_b.id
    assert resolved_a.tenant_id != resolved_b.tenant_id
    assert resolved_a.channel_id != resolved_b.channel_id


async def test_inactive_channel_does_not_resolve(db_session: AsyncSession, seed: Seed) -> None:
    """A deactivated channel is not one we serve, so its traffic must not be
    processed under the tenant that used to own it.
    """
    seed.a.channel.is_active = False
    await db_session.flush()

    assert await resolve_instagram_channel(db_session, seed.a.channel.external_id) is None


async def test_same_external_id_on_another_channel_type_does_not_resolve(
    db_session: AsyncSession, seed: Seed
) -> None:
    """Platform id namespaces are separate: an id that is an Instagram
    account here could be a Telegram bot id somewhere else, and resolving
    across the two would hand one platform's traffic to the other's channel.
    """
    resolved = await resolve_channel(
        db_session,
        channel_type=ChannelType.TELEGRAM,
        external_id=seed.a.channel.external_id,
    )

    assert resolved is None


@pytest.mark.parametrize("channel_type", [ChannelType.INSTAGRAM, "instagram"])
async def test_resolve_channel_accepts_the_enum_or_its_value(
    db_session: AsyncSession, seed: Seed, channel_type: str
) -> None:
    """Channel.type is a plain string column, so callers holding either form
    must resolve the same row.
    """
    resolved = await resolve_channel(
        db_session, channel_type=channel_type, external_id=seed.a.channel.external_id
    )

    assert resolved is not None
    assert resolved.channel_id == seed.a.channel.id


# --- the clinic's own on/off switch for the assistant ---


async def test_bot_replies_enabled_defaults_to_on_when_never_set(
    db_session: AsyncSession, seed: Seed
) -> None:
    """A clinic that has never touched the switch has had a bot answering all
    along. A missing key must read as on, or shipping this feature would
    silence every existing deployment.
    """
    from app.services.tenant_resolution import bot_replies_enabled

    assert await bot_replies_enabled(db_session, seed.tenant_a.id) is True


async def test_bot_replies_enabled_follows_the_setting(
    db_session: AsyncSession, seed: Seed
) -> None:
    from app.services.tenant_resolution import bot_replies_enabled

    tenant = await db_session.get(Tenant, seed.tenant_a.id)
    assert tenant is not None
    tenant.settings = {**tenant.settings, "bot_replies_enabled": False}
    await db_session.flush()

    assert await bot_replies_enabled(db_session, seed.tenant_a.id) is False

    tenant.settings = {**tenant.settings, "bot_replies_enabled": True}
    await db_session.flush()

    assert await bot_replies_enabled(db_session, seed.tenant_a.id) is True


async def test_bot_replies_enabled_is_per_clinic(db_session: AsyncSession, seed: Seed) -> None:
    """One clinic switching its assistant off must not silence another's."""
    from app.services.tenant_resolution import bot_replies_enabled

    tenant_a = await db_session.get(Tenant, seed.tenant_a.id)
    assert tenant_a is not None
    tenant_a.settings = {**tenant_a.settings, "bot_replies_enabled": False}
    await db_session.flush()

    assert await bot_replies_enabled(db_session, seed.tenant_a.id) is False
    assert await bot_replies_enabled(db_session, seed.tenant_b.id) is True
