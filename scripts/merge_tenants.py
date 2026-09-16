r"""Move everything belonging to one clinic onto another, then delete the empty one.

This exists because a deployment can end up with the same clinic recorded
twice. app.core.provisioning finds a clinic by PROVISION_TENANT_NAME, so a
deployment whose Instagram channel was provisioned under one name and whose
Telegram token arrived after the name had changed gets a second clinic --
and the second one has no knowledge base, so the assistant answers on that
channel without any of the clinic's own FAQs. Nothing in the provisioning
path repairs this on its own: the Telegram channel is looked up by bot id
alone, so once the row exists it keeps whichever clinic it was created
under, whatever the name is set to afterwards.

Both clinics are named explicitly rather than guessed. Picking the target by
"the one with FAQs" would do the right thing today and the wrong thing on a
deployment that seeded the wrong clinic, and this is not an operation with a
comfortable undo.

Idempotent: with the source clinic already gone, it reports that and exits 0,
which is what makes it safe to leave wired up as a pre-deploy command for a
deploy or two.

Usage, from the instagram/ directory:

    python scripts/merge_tenants.py <source-uuid> <target-uuid>

Every row the schema scopes to a clinic is moved -- channels, conversations,
knowledge base, appointments, doctors, leads, operators, users. The source
clinic is deleted only once nothing points at it any more; if anything still
does, the script says so and changes nothing.
"""

import asyncio
import sys
import uuid
from typing import Any, cast

from sqlalchemy import func, select, update
from sqlalchemy.engine import CursorResult

from app.core.db import db_session
from app.models.appointment import Appointment
from app.models.channel import Channel
from app.models.conversation import Conversation
from app.models.doctor import Doctor
from app.models.knowledge_base import KnowledgeBase
from app.models.lead import Lead
from app.models.operator import Operator
from app.models.tenant import Tenant
from app.models.user import User

# Every model the schema scopes to a clinic. Listed rather than discovered by
# reflection so that a new tenant-scoped table is a deliberate addition here:
# a table silently left out would strand its rows on a clinic that is about
# to be deleted, and the delete would fail with a foreign key error rather
# than lose them -- noisy, but only after the move is half done.
TENANT_SCOPED = (
    Channel,
    Conversation,
    KnowledgeBase,
    Appointment,
    Doctor,
    Lead,
    Operator,
    User,
)


async def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(f"Usage: {sys.argv[0]} <source-tenant-uuid> <target-tenant-uuid>")

    try:
        source_id = uuid.UUID(sys.argv[1])
        target_id = uuid.UUID(sys.argv[2])
    except ValueError as exc:
        sys.exit(f"Not a UUID: {exc}")

    if source_id == target_id:
        sys.exit("Source and target are the same clinic.")

    async with db_session() as session:
        tenants = {
            tenant.id: tenant
            for tenant in (
                await session.execute(select(Tenant).where(Tenant.id.in_([source_id, target_id])))
            ).scalars()
        }

        target = tenants.get(target_id)
        if target is None:
            sys.exit(f"No clinic with id {target_id} — nothing to merge into.")

        source = tenants.get(source_id)
        if source is None:
            # The already-merged case, and the reason this is safe to leave
            # wired up: a second run must not be an error.
            print(f"merge_tenants: no clinic {source_id}; already merged, nothing to do.")
            return

        print(f"merge_tenants: {source.name!r} ({source_id}) -> {target.name!r} ({target_id})")

        moved: list[str] = []
        for model in TENANT_SCOPED:
            # CursorResult, not Result: .rowcount is what says how much moved,
            # and an UPDATE always yields one.
            result = cast(
                "CursorResult[Any]",
                await session.execute(
                    update(model)
                    .where(model.tenant_id == source_id)
                    .values(tenant_id=target_id)
                    .execution_options(synchronize_session=False)
                ),
            )
            if result.rowcount:
                moved.append(f"{model.__tablename__}={result.rowcount}")

        remaining = [
            model.__tablename__
            for model in TENANT_SCOPED
            if (
                await session.execute(
                    select(func.count()).select_from(model).where(model.tenant_id == source_id)
                )
            ).scalar_one()
        ]
        if remaining:  # pragma: no cover - defensive
            await session.rollback()
            sys.exit(f"Rows still reference {source_id}: {', '.join(remaining)}. Nothing changed.")

        await session.delete(source)
        await session.commit()

    print(f"merge_tenants: moved {', '.join(moved) if moved else 'nothing'}; deleted {source_id}.")


if __name__ == "__main__":
    asyncio.run(main())
