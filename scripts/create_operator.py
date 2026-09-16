r"""Create (or reset the password of) an Operator login for the dashboard.

There's no admin UI to do this yet — the dashboard needs at least one
operator account to log in with at all, so this is how the first ones get
created. Reads the password from an environment variable, never a CLI arg,
so it never lands in shell history or process listings.

Usage (PowerShell), from the repo root:

    $env:TENANT_ID = "ad15de96-15c3-4e97-a51f-66873dfcdcf9"
    $env:OPERATOR_NAME = "Dr. Aziza Karimova"
    $env:OPERATOR_ROLE = "operator"        # see app.core.roles: "admin", "operator" or "doctor"
    $env:OPERATOR_USERNAME = "aziza"
    $env:OPERATOR_PASSWORD = "<choose a real password>"
    ./.venv/Scripts/python.exe scripts/create_operator.py
    Remove-Item Env:\OPERATOR_PASSWORD

If OPERATOR_USERNAME already exists, its password and role are updated in
place rather than creating a duplicate row.
"""

import asyncio
import os
import sys
import uuid

from sqlalchemy import select

from app.core.db import db_session
from app.core.passwords import hash_password
from app.core.roles import Role
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.models.operator import Operator


async def main() -> None:
    try:
        tenant_id = uuid.UUID(os.environ["TENANT_ID"])
        name = os.environ["OPERATOR_NAME"]
        role = os.environ["OPERATOR_ROLE"]
        username = os.environ["OPERATOR_USERNAME"]
        password = os.environ["OPERATOR_PASSWORD"]
    except KeyError as exc:
        sys.exit(f"Missing required environment variable: {exc}")

    # Checked here as well as by the database's own constraint, so a typo
    # gets a sentence naming the three roles instead of an asyncpg
    # IntegrityError from four frames down.
    if role not in tuple(Role):
        allowed = ", ".join(sorted(Role))
        sys.exit(f"OPERATOR_ROLE must be one of {allowed}, got {role!r}")

    async with db_session() as session:
        token = set_current_tenant(tenant_id)
        try:
            result = await session.execute(select(Operator).where(Operator.username == username))
            operator = result.scalar_one_or_none()
            if operator is None:
                operator = Operator(
                    tenant_id=tenant_id,
                    name=name,
                    role=role,
                    username=username,
                    password_hash=hash_password(password),
                )
                session.add(operator)
                action = "Created"
            else:
                operator.name = name
                operator.role = role
                operator.password_hash = hash_password(password)
                action = "Updated"
            await session.commit()
        finally:
            reset_current_tenant(token)

    print(f"{action} operator username={username!r} role={role!r} tenant_id={tenant_id}")


if __name__ == "__main__":
    asyncio.run(main())
