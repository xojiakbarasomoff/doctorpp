import time
import uuid

import pytest
from arq import Retry
from arq.connections import ArqRedis
from arq.jobs import Job

from app.services.debounce import (
    FIRE_DEBOUNCE_WINDOW_JOB,
    PROCESS_INBOUND_MESSAGE_JOB,
    handle_inbound_message,
    join_messages,
    pop_batch_if_current_generation,
    restore_batch,
)
from app.workers.tasks import fire_debounce_window, process_inbound_message

SENDER = "sender-1"


def _keys(tenant_id: uuid.UUID, channel_id: uuid.UUID, sender: str = SENDER) -> tuple[str, str]:
    prefix = f"debounce:{tenant_id}:{channel_id}:{sender}"
    return f"{prefix}:messages", f"{prefix}:generation"


def test_job_name_constants_match_the_real_functions() -> None:
    """debounce.py references these jobs by string literal (to avoid a
    circular import with app.workers.tasks) — this is the guard against that
    string silently drifting from the real function name.
    """
    assert process_inbound_message.__name__ == PROCESS_INBOUND_MESSAGE_JOB
    assert fire_debounce_window.__name__ == FIRE_DEBOUNCE_WINDOW_JOB


# --- timer reset via generation counter ---


async def test_handle_inbound_message_buffers_and_schedules_deferred_job(
    redis_pool: ArqRedis,
) -> None:
    tenant_id, channel_id, conversation_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    messages_key, generation_key = _keys(tenant_id, channel_id)

    await handle_inbound_message(
        redis_pool,
        tenant_id=tenant_id,
        channel_id=channel_id,
        conversation_id=conversation_id,
        sender_external_id=SENDER,
        message_text="What are your hours?",
        window_seconds=25,
    )

    messages = await redis_pool.lrange(messages_key, 0, -1)
    assert [m.decode() for m in messages] == ["What are your hours?"]
    assert await redis_pool.get(generation_key) == b"1"

    job_ids = await redis_pool.zrange("arq:queue", 0, -1)
    assert len(job_ids) == 1
    # Score is the scheduled fire time in ms; confirm it's ~25s out, not now.
    score = await redis_pool.zscore("arq:queue", job_ids[0])
    now_ms = time.time() * 1000
    assert 24_000 < (score - now_ms) < 26_000

    # The deferred job carries the channel and conversation it belongs to,
    # so the worker never has to guess which account to reply from.
    job = Job(job_ids[0].decode(), redis=redis_pool, _queue_name="arq:queue")
    info = await job.info()
    assert info is not None
    assert info.function == FIRE_DEBOUNCE_WINDOW_JOB
    # The trailing None is the platform's reply context (see
    # ChannelAdapter.send_text) — nothing to route by on this channel.
    assert info.args == (str(tenant_id), str(channel_id), str(conversation_id), SENDER, 1, None)


async def test_second_message_resets_timer_via_generation_counter(redis_pool: ArqRedis) -> None:
    tenant_id, channel_id, conversation_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    messages_key, generation_key = _keys(tenant_id, channel_id)

    for text in ("first message", "second message"):
        await handle_inbound_message(
            redis_pool,
            tenant_id=tenant_id,
            channel_id=channel_id,
            conversation_id=conversation_id,
            sender_external_id=SENDER,
            message_text=text,
            window_seconds=25,
        )

    assert await redis_pool.get(generation_key) == b"2"

    messages = await redis_pool.lrange(messages_key, 0, -1)
    assert [m.decode() for m in messages] == ["first message", "second message"]

    # Two messages -> two scheduled deferred jobs (the first becomes a
    # no-op via the generation check when it eventually fires; see
    # test_pop_batch_if_current_generation_returns_none_for_stale_generation).
    job_ids = await redis_pool.zrange("arq:queue", 0, -1)
    assert len(job_ids) == 2


async def test_same_sender_id_on_two_channels_buffers_separately(
    redis_pool: ArqRedis,
) -> None:
    """The reason the buffer key carries a channel id.

    Platform user ids are unique only within that platform's own account, so
    one tenant can genuinely see the same id string from two different people
    — a Telegram chat id and an Instagram-scoped id, or two Instagram
    accounts the clinic owns. Keyed on tenant and sender alone, their
    messages would land in one buffer and one of them would be answered with
    the other's question.
    """
    tenant_id = uuid.uuid4()
    instagram_channel, telegram_channel = uuid.uuid4(), uuid.uuid4()

    await handle_inbound_message(
        redis_pool,
        tenant_id=tenant_id,
        channel_id=instagram_channel,
        conversation_id=uuid.uuid4(),
        sender_external_id=SENDER,
        message_text="tishim og'riyapti",
        window_seconds=25,
    )
    await handle_inbound_message(
        redis_pool,
        tenant_id=tenant_id,
        channel_id=telegram_channel,
        conversation_id=uuid.uuid4(),
        sender_external_id=SENDER,
        message_text="narxi qancha?",
        window_seconds=25,
    )

    ig_messages_key, _ = _keys(tenant_id, instagram_channel)
    tg_messages_key, _ = _keys(tenant_id, telegram_channel)
    assert [m.decode() for m in await redis_pool.lrange(ig_messages_key, 0, -1)] == [
        "tishim og'riyapti"
    ]
    assert [m.decode() for m in await redis_pool.lrange(tg_messages_key, 0, -1)] == [
        "narxi qancha?"
    ]


# --- batch joins in order ---


def test_join_messages_preserves_order_with_newlines() -> None:
    assert join_messages(["first", "second", "third"]) == "first\nsecond\nthird"


# --- window fires once after quiet period / stale generation no-ops ---


async def test_pop_batch_if_current_generation_pops_and_clears_when_current(
    redis_pool: ArqRedis,
) -> None:
    tenant_id, channel_id = uuid.uuid4(), uuid.uuid4()
    messages_key, generation_key = _keys(tenant_id, channel_id)

    await redis_pool.rpush(messages_key, "hello", "again")
    await redis_pool.set(generation_key, "2")

    result = await pop_batch_if_current_generation(redis_pool, tenant_id, channel_id, SENDER, 2)

    assert result == ["hello", "again"]
    assert await redis_pool.exists(messages_key) == 0
    assert await redis_pool.exists(generation_key) == 0


async def test_pop_batch_if_current_generation_returns_none_for_stale_generation(
    redis_pool: ArqRedis,
) -> None:
    tenant_id, channel_id = uuid.uuid4(), uuid.uuid4()
    messages_key, generation_key = _keys(tenant_id, channel_id)

    await redis_pool.rpush(messages_key, "hello", "again")
    await redis_pool.set(generation_key, "2")

    # generation=1 is stale — a later message (which set generation to 2)
    # arrived after this (hypothetical) job was scheduled.
    result = await pop_batch_if_current_generation(redis_pool, tenant_id, channel_id, SENDER, 1)

    assert result is None
    # Must not have touched anything — a newer message is still
    # accumulating into this buffer.
    messages = await redis_pool.lrange(messages_key, 0, -1)
    assert [m.decode() for m in messages] == ["hello", "again"]
    assert await redis_pool.get(generation_key) == b"2"


async def test_pop_batch_if_current_generation_returns_none_when_nothing_pending(
    redis_pool: ArqRedis,
) -> None:
    result = await pop_batch_if_current_generation(
        redis_pool, uuid.uuid4(), uuid.uuid4(), SENDER, 1
    )
    assert result is None


# --- a failed batch is put back, not lost ------------------------------


async def test_a_restored_batch_can_be_claimed_again(redis_pool: ArqRedis) -> None:
    """The claim is destructive: after popping, Redis no longer holds the
    patient's words. If answering them fails, the only copy is in the
    worker's memory, and this is what puts it back.
    """
    tenant_id, channel_id = uuid.uuid4(), uuid.uuid4()
    messages_key, generation_key = _keys(tenant_id, channel_id)

    await redis_pool.rpush(messages_key, "salom", "buyragim og'riyapti")
    await redis_pool.set(generation_key, "1")

    claimed = await pop_batch_if_current_generation(redis_pool, tenant_id, channel_id, SENDER, 1)
    assert claimed == ["salom", "buyragim og'riyapti"]
    assert await redis_pool.exists(messages_key) == 0

    await restore_batch(redis_pool, tenant_id, channel_id, SENDER, 1, claimed)

    # The retry runs with the same generation, so it must claim the same batch.
    again = await pop_batch_if_current_generation(redis_pool, tenant_id, channel_id, SENDER, 1)
    assert again == claimed


async def test_a_restored_batch_goes_in_front_of_newer_messages(
    redis_pool: ArqRedis,
) -> None:
    """A message that arrived while the failed job was running started a new
    generation with its own job already scheduled. The restored words were
    said first, so they belong at the head -- and the counter must be left
    alone, or two jobs would both think they own the buffer.
    """
    tenant_id, channel_id = uuid.uuid4(), uuid.uuid4()
    messages_key, generation_key = _keys(tenant_id, channel_id)

    await redis_pool.rpush(messages_key, "yana bir savol")
    await redis_pool.set(generation_key, "7")

    await restore_batch(redis_pool, tenant_id, channel_id, SENDER, 3, ["birinchi", "ikkinchi"])

    assert await redis_pool.get(generation_key) == b"7"
    assert await redis_pool.lrange(messages_key, 0, -1) == [
        b"birinchi",
        b"ikkinchi",
        b"yana bir savol",
    ]
    # The failed job's own retry expects generation 3 and must now no-op,
    # which is what stops the same words being answered twice.
    assert (
        await pop_batch_if_current_generation(redis_pool, tenant_id, channel_id, SENDER, 3) is None
    )


async def test_a_failing_job_leaves_the_patients_words_in_redis(
    redis_pool: ArqRedis,
) -> None:
    """The whole point. Before this, a rate-limited model meant the patient
    wrote, nobody answered, and nothing recorded that it happened.
    """
    tenant_id, channel_id = uuid.uuid4(), uuid.uuid4()
    messages_key, generation_key = _keys(tenant_id, channel_id)

    await redis_pool.rpush(messages_key, "qabulga yozilmoqchiman")
    await redis_pool.set(generation_key, "1")

    class Failing:
        """A session factory whose context manager raises on entry, the way a
        rate-limited model surfaces once the job is already holding the batch."""

        def __call__(self) -> "Failing":
            return self

        async def __aenter__(self) -> None:
            raise RuntimeError("429 quota exceeded")

        async def __aexit__(self, *exc: object) -> None:
            return None

    # An early attempt asks arq to come back later rather than giving up.
    with pytest.raises(Retry):
        await fire_debounce_window(
            {"redis": redis_pool, "job_try": 1},
            str(tenant_id),
            str(channel_id),
            str(uuid.uuid4()),
            SENDER,
            1,
            None,
            session_factory=Failing(),
        )

    assert await redis_pool.lrange(messages_key, 0, -1) == [b"qabulga yozilmoqchiman"]
    assert await redis_pool.get(generation_key) == b"1"

    # The last attempt lets the real error out instead of deferring forever,
    # so the failure is recorded rather than looping quietly.
    with pytest.raises(RuntimeError):
        await fire_debounce_window(
            {"redis": redis_pool, "job_try": 5},
            str(tenant_id),
            str(channel_id),
            str(uuid.uuid4()),
            SENDER,
            1,
            None,
            session_factory=Failing(),
        )
