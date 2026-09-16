import uuid

from app.core.tenant_context import get_current_tenant
from app.models.user import User
from app.repositories.base import TenantScopedRepository


class UserRepository(TenantScopedRepository[User]):
    model = User

    async def get_by_external_id(self, *, channel_id: uuid.UUID, external_id: str) -> User | None:
        """The patient behind a platform id on one channel, or None.

        Keyed on (channel_id, external_id) rather than external_id alone
        because the id namespaces are per-account, not global: the same
        person messaging two of a tenant's channels is two rows, and two
        different people on two platforms can hold the same id string.
        Tenant-scoped like every other read here.
        """
        return await self._get(
            tenant_id=get_current_tenant(), channel_id=channel_id, external_id=external_id
        )
