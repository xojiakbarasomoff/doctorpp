"""One-shot claims on platform events, so a redelivery cannot be answered twice.

Every messaging platform redelivers: Meta repeats a webhook whose 200 came
back too slowly or not at all, and Telegram repeats an update that was not
acknowledged. Without a claim, a redelivered message is registered again and
answered again — the patient gets the same reply twice, and the transcript
records a message they only sent once.

Redis rather than a database row because the check must be atomic against
concurrent deliveries and is pure short-lived bookkeeping: SET NX is exactly
"claim this if nobody has", in one round trip, and the key expires on its
own rather than needing a cleanup job.
"""

import uuid

from arq.connections import ArqRedis

# Comfortably longer than any redelivery window a platform actually uses
# (Meta retries a failed delivery for up to a few hours), short enough that
# the keyspace stays proportional to a day of traffic rather than growing
# forever.
CLAIM_TTL_SECONDS = 24 * 60 * 60


def _claim_key(tenant_id: uuid.UUID, channel_type: str, event_id: str) -> str:
    return f"event_claim:{tenant_id}:{channel_type}:{event_id}"


async def claim_event(
    pool: ArqRedis, *, tenant_id: uuid.UUID, channel_type: str, event_id: str
) -> bool:
    """Claim `event_id` for processing. True the first time, False after.

    The caller must skip the event entirely on False. Namespaced by tenant
    and channel type because a platform's ids are only unique within its own
    account — nothing stops a Telegram update id from colliding with an
    Instagram message id.
    """
    claimed = await pool.set(
        _claim_key(tenant_id, channel_type, event_id), "1", ex=CLAIM_TTL_SECONDS, nx=True
    )
    return bool(claimed)
