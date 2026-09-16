import uuid
from collections.abc import Sequence
from typing import Any

from app.models.tenant import Tenant
from app.repositories.base import BaseRepository


class TenantRepository(BaseRepository[Tenant]):
    """Tenants are not tenant-scoped: a Tenant row *is* the tenant, so there is
    no current-tenant filter to apply. This repository deliberately never
    touches tenant_context.
    """

    model = Tenant

    async def get(self, id_: uuid.UUID) -> Tenant | None:
        return await self._get(id=id_)

    async def list(self) -> Sequence[Tenant]:
        return await self._list()

    async def create(self, **values: Any) -> Tenant:
        return await self._create(**values)
