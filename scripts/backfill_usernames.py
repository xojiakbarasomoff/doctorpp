r"""Give the patients already in the database a handle, once.

app.services.profile resolves a handle when a patient writes, which covers
everybody from now on and nobody from before. On the day the column landed
there were sixteen conversations in this deployment, all of them still
labelled with a platform id -- so the dashboard would have looked unchanged
to the clinic until each of those patients happened to write again.

    python scripts/backfill_usernames.py            # do it
    python scripts/backfill_usernames.py --dry-run  # count them first

One Graph API call per patient, sequential on purpose: this runs once, over
a list measured in tens, and there is nothing to gain by asking Meta for
sixteen things at the same time. Patients whose handle cannot be had are
left alone and reported -- they keep their id as a label, which is what they
had anyway, and the next message they send will try again.

Safe to run twice: a patient who already has a handle is skipped without a
call.
"""

import argparse
import asyncio
import sys

from sqlalchemy import select

from app.channels.base import ChannelType
from app.core.db import db_session
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.models.channel import Channel
from app.models.user import User
from app.services.profile import ensure_instagram_username


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="count who is missing a handle without asking Meta for any",
    )
    args = parser.parse_args()

    async with db_session() as session:
        # Straight through the model rather than the tenant-scoped
        # repository: this is an operator's one-off over the whole
        # deployment, not a request serving one clinic, and the tenant is
        # set per row below so the lookup itself stays scoped.
        rows = list(
            (
                await session.execute(
                    select(User, Channel)
                    .join(Channel, Channel.id == User.channel_id)
                    .where(User.username.is_(None))
                    .where(Channel.type == ChannelType.INSTAGRAM)
                    .order_by(User.created_at)
                )
            ).all()
        )

    print(f"patients without a handle: {len(rows)}")
    if args.dry_run or not rows:
        return

    resolved = 0
    failed: list[str] = []
    for user, channel in rows:
        token = set_current_tenant(user.tenant_id)
        try:
            async with db_session() as session:
                username = await ensure_instagram_username(
                    session, channel_id=channel.id, user_id=user.id
                )
                if username is None:
                    failed.append(user.external_id)
                    continue
                await session.commit()
                resolved += 1
                print(f"  {user.external_id} -> @{username}")
        finally:
            reset_current_tenant(token)

    print(f"\nresolved {resolved}, still without one {len(failed)}")
    if failed:
        # Not an error. An account can have no username, and a patient who
        # wrote once a year ago may be beyond what the messaging permission
        # covers. They keep their id, which is what they had before.
        print("  " + ", ".join(failed))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
