r"""Create a Tenant + its Instagram Channel row in an otherwise-empty database
(e.g. right after the first `alembic upgrade head` on a fresh environment).

Every other bootstrapping script (create_operator.py, set_channel_credentials.py)
assumes a Tenant and Channel already exist — this is the one that creates them
in the first place. credentials is seeded with an encrypted placeholder value
("pending", recognized by app.channels.instagram.client.is_placeholder_credential)
since the real access token isn't known yet; run set_channel_credentials.py
afterward to fill it in.

Idempotent: if a Channel with this external_id already exists, prints its
existing tenant_id instead of inserting a duplicate (external_id is
unique per channel type).

Usage (PowerShell), from the repo root:

    $env:TENANT_NAME = "Example Dental Clinic"
    $env:IG_ACCOUNT_ID = "17841435883696894"
    ./.venv/Scripts/python.exe scripts/bootstrap_tenant.py
"""

import asyncio
import os
import sys

from sqlalchemy import select

from app.core.db import db_session
from app.core.encryption import encrypt
from app.models.channel import Channel
from app.models.tenant import Tenant

_CHANNEL_TYPE = "instagram"


async def main() -> None:
    try:
        tenant_name = os.environ["TENANT_NAME"]
        ig_account_id = os.environ["IG_ACCOUNT_ID"]
    except KeyError as exc:
        sys.exit(f"Missing required environment variable: {exc}")

    async with db_session() as session:
        result = await session.execute(
            select(Channel).where(
                Channel.type == _CHANNEL_TYPE, Channel.external_id == ig_account_id
            )
        )
        existing = result.scalar_one_or_none()
        if existing is not None:
            print(
                f"Channel already exists for external_id={ig_account_id!r} "
                f"(channel_id={existing.id}, tenant_id={existing.tenant_id}) - nothing created."
            )
            return

        tenant = Tenant(name=tenant_name, status="active")
        session.add(tenant)
        await session.flush()

        channel = Channel(
            tenant_id=tenant.id,
            type=_CHANNEL_TYPE,
            external_id=ig_account_id,
            credentials=encrypt("pending"),
            is_active=True,
        )
        session.add(channel)
        await session.commit()

    print(f"Created tenant_id={tenant.id} name={tenant_name!r}")
    print(f"Created channel_id={channel.id} type={_CHANNEL_TYPE} external_id={ig_account_id!r}")
    print("credentials is a placeholder - run scripts/set_channel_credentials.py next.")


if __name__ == "__main__":
    asyncio.run(main())
