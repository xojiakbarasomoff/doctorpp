import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import or_, select

from app.core.tenant_context import get_current_tenant
from app.models.lead import Lead
from app.models.user import User
from app.repositories.base import CrossTenantAccessError, TenantScopedRepository
from app.services import search


class LeadRepository(TenantScopedRepository[Lead]):
    model = Lead

    async def list_recent(
        self, *, status: str | None = None, q: str | None = None, limit: int = 100
    ) -> Sequence[Lead]:
        """The call centre's queue, newest first.

        Newest first because a lead is worth most immediately after the
        patient asked — a callback two days late is a lost patient, so the
        list is ordered the way it should be worked.

        Filtered here rather than in the dashboard: the dashboard only ever
        holds `limit` rows, and a search that looked through those alone
        could not find the patient who asked last month.
        """
        stmt = select(Lead).where(Lead.tenant_id == get_current_tenant())
        if status is not None:
            stmt = stmt.where(Lead.status == status)
        if q and q.strip():
            matches = [
                search.contains(Lead.patient_name, q),
                search.contains(Lead.topic, q),
                search.contains(User.username, q),
                search.contains(User.name, q),
            ]
            digits = search.phone_digits(q)
            if digits:
                matches += [
                    search.phone_contains(Lead.phone, digits),
                    search.phone_contains(User.phone, digits),
                ]
            stmt = stmt.outerjoin(User, User.id == Lead.user_id).where(or_(*matches))
        stmt = stmt.order_by(Lead.created_at.desc()).limit(limit)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def get_open_for_conversation(self, conversation_id: uuid.UUID) -> Lead | None:
        """The lead already raised from this conversation, if any.

        Used so a patient who leaves a phone number and then keeps chatting
        does not generate a second lead for the same request — the call
        centre would ring them twice.
        """
        return await self._get(tenant_id=get_current_tenant(), conversation_id=conversation_id)

    async def update(self, obj: Lead, **values: Any) -> Lead:
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
