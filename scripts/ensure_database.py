"""Create this deployment's database on its Postgres server, if it is missing.

Run before migrations on a host where one Postgres server is shared with
another project -- the doctor's bot living beside the clinic's. Each project
gets its own database on the server, so neither can read or write the other's
rows, and this is what makes that database exist on the first deploy.

Only ever creates. It never drops, never truncates, and never touches a
database other than the one named in DATABASE_URL.

    python scripts/ensure_database.py && alembic upgrade head
"""

import asyncio
import sys
from urllib.parse import urlparse

import asyncpg

from app.core.config import get_settings


async def main() -> int:
    parts = urlparse(get_settings().database_url.replace("+asyncpg", ""))
    name = parts.path.lstrip("/")
    if not name:
        print("DATABASE_URL names no database", file=sys.stderr)
        return 1
    # The server's maintenance database, which always exists, to run CREATE
    # DATABASE from -- it cannot run inside the database being created.
    connection = await asyncpg.connect(
        user=parts.username,
        password=parts.password,
        host=parts.hostname,
        port=parts.port or 5432,
        database="postgres",
    )
    try:
        exists = await connection.fetchval("SELECT 1 FROM pg_database WHERE datname = $1", name)
        if exists:
            print(f"database {name!r} already exists")
            return 0
        # Quoted identifier, not a bind parameter: CREATE DATABASE takes none.
        # The name comes from our own configuration, never from a request.
        await connection.execute(f'CREATE DATABASE "{name}"')
        print(f"database {name!r} created")
        return 0
    finally:
        await connection.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
