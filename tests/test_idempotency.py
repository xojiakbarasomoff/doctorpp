import uuid

from arq.connections import ArqRedis

from app.services.idempotency import CLAIM_TTL_SECONDS, claim_event


async def test_first_claim_succeeds_and_the_redelivery_does_not(redis_pool: ArqRedis) -> None:
    tenant_id = uuid.uuid4()

    first = await claim_event(
        redis_pool, tenant_id=tenant_id, channel_type="instagram", event_id="mid-1"
    )
    second = await claim_event(
        redis_pool, tenant_id=tenant_id, channel_type="instagram", event_id="mid-1"
    )

    assert first is True
    assert second is False


async def test_distinct_events_each_get_their_own_claim(redis_pool: ArqRedis) -> None:
    tenant_id = uuid.uuid4()

    assert await claim_event(
        redis_pool, tenant_id=tenant_id, channel_type="instagram", event_id="mid-1"
    )
    assert await claim_event(
        redis_pool, tenant_id=tenant_id, channel_type="instagram", event_id="mid-2"
    )


async def test_claims_are_namespaced_by_tenant(redis_pool: ArqRedis) -> None:
    """One clinic's message id must not silence another clinic's."""
    event_id = "mid-shared"

    assert await claim_event(
        redis_pool, tenant_id=uuid.uuid4(), channel_type="instagram", event_id=event_id
    )
    assert await claim_event(
        redis_pool, tenant_id=uuid.uuid4(), channel_type="instagram", event_id=event_id
    )


async def test_claims_are_namespaced_by_channel_type(redis_pool: ArqRedis) -> None:
    """Nothing stops a Telegram update id from colliding with an Instagram
    message id — they are separate id spaces, so one must not claim the
    other's event.
    """
    tenant_id = uuid.uuid4()
    event_id = "12345"

    assert await claim_event(
        redis_pool, tenant_id=tenant_id, channel_type="instagram", event_id=event_id
    )
    assert await claim_event(
        redis_pool, tenant_id=tenant_id, channel_type="telegram", event_id=event_id
    )


async def test_claim_expires_rather_than_accumulating_forever(redis_pool: ArqRedis) -> None:
    tenant_id = uuid.uuid4()
    await claim_event(redis_pool, tenant_id=tenant_id, channel_type="instagram", event_id="mid-1")

    ttl = await redis_pool.ttl(f"event_claim:{tenant_id}:instagram:mid-1")

    assert 0 < ttl <= CLAIM_TTL_SECONDS
