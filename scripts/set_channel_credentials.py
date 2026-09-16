r"""Encrypt a plaintext channel access token and store it, without the token
ever appearing in chat, git history, shell command-line args, or logs.

Run from your own terminal — reads the token from an environment variable
you set yourself, encrypts it with the app's real ENCRYPTION_KEY, and writes
only the ciphertext to channels.credentials.

Usage (PowerShell), from the repo root:

    $env:CHANNEL_TYPE = "instagram"
    $env:CHANNEL_EXTERNAL_ID = "17841435883696894"
    $env:CHANNEL_TOKEN = "<paste the real access token here>"
    ./.venv/Scripts/python.exe scripts/set_channel_credentials.py
    Remove-Item Env:\CHANNEL_TOKEN

CHANNEL_TYPE defaults to "instagram" if not set.
"""

import asyncio
import os
import sys

from sqlalchemy import select
from sqlalchemy.exc import NoResultFound

from app.core.db import db_session
from app.core.encryption import decrypt, encrypt
from app.models.channel import Channel


async def main() -> None:
    channel_type = os.environ.get("CHANNEL_TYPE", "instagram")
    try:
        external_id = os.environ["CHANNEL_EXTERNAL_ID"]
        token = os.environ["CHANNEL_TOKEN"]
    except KeyError as exc:
        sys.exit(f"Missing required environment variable: {exc}")

    async with db_session() as session:
        result = await session.execute(
            select(Channel).where(Channel.type == channel_type, Channel.external_id == external_id)
        )
        try:
            channel = result.scalar_one()
        except NoResultFound:
            sys.exit(f"No {channel_type} channel found with external_id={external_id!r}")

        channel.credentials = encrypt(token)
        await session.commit()

        # Round-trip check before declaring success — proves the value
        # written can actually be read back correctly, without ever
        # printing the token itself.
        assert decrypt(channel.credentials) == token

    print(f"Stored encrypted credentials for channel_id={channel.id} (external_id={external_id}).")
    print(f"token_length={len(token)}")


if __name__ == "__main__":
    asyncio.run(main())
