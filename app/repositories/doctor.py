from collections.abc import Sequence
from typing import Any

from sqlalchemy import select

from app.core.tenant_context import get_current_tenant
from app.models.doctor import Doctor
from app.repositories.base import CrossTenantAccessError, TenantScopedRepository


class DoctorRepository(TenantScopedRepository[Doctor]):
    model = Doctor

    async def list_active(self) -> Sequence[Doctor]:
        """The clinicians a patient may currently be booked with, by name.

        Ordered so the booking UI and the bot offer the same list in the
        same order — an arbitrary order would reshuffle the Mini App's
        doctor picker on every load.
        """
        stmt = (
            select(Doctor)
            .where(Doctor.tenant_id == get_current_tenant(), Doctor.is_active.is_(True))
            .order_by(Doctor.name)
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def update(self, obj: Doctor, **values: Any) -> Doctor:
        """Mutates and flushes an already-loaded row.

        Re-checks obj.tenant_id itself — the same defensive pattern as
        KnowledgeBaseRepository.update() and AppointmentRepository.update().
        """
        current_tenant_id = get_current_tenant()
        if obj.tenant_id != current_tenant_id:
            raise CrossTenantAccessError(
                f"obj.tenant_id={obj.tenant_id} does not match the current tenant "
                f"({current_tenant_id})"
            )
        for field, value in values.items():
            setattr(obj, field, value)
        await self.session.flush()
        return obj
