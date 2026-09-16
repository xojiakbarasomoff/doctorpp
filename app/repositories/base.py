import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tenant_context import get_current_tenant
from app.models.base import Base


class TenantIsolationError(Exception):
    """Base class for tenant-isolation violations raised by the repository layer."""


class MissingTenantColumnError(TenantIsolationError):
    """Raised when TenantScopedRepository is used for a model with no tenant_id column."""


class CrossTenantAccessError(TenantIsolationError):
    """Raised when an operation would reach across tenant boundaries."""


class BaseRepository[ModelType: Base]:
    """Common async CRUD primitives shared by every repository.

    These are intentionally protected (leading underscore), not a public API:
    a bare BaseRepository has no opinion on tenant isolation, so exposing
    get()/list()/create() directly here would make it too easy to
    accidentally query without a tenant filter. Concrete repositories decide,
    explicitly, how they scope access — see TenantScopedRepository (automatic
    tenant_id filtering), TenantRepository (tenants aren't tenant-scoped), and
    MessageRepository (scoped indirectly through conversation).
    """

    model: type[ModelType]

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def _get(self, **filters: Any) -> ModelType | None:
        stmt = select(self.model)
        for field, value in filters.items():
            stmt = stmt.where(getattr(self.model, field) == value)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def _list(self, **filters: Any) -> Sequence[ModelType]:
        stmt = select(self.model)
        for field, value in filters.items():
            stmt = stmt.where(getattr(self.model, field) == value)
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def _create(self, **values: Any) -> ModelType:
        obj = self.model(**values)
        self.session.add(obj)
        await self.session.flush()
        return obj


class TenantScopedRepository[ModelType: Base](BaseRepository[ModelType]):
    """Base for repositories over models with a direct tenant_id column.

    Every read filters on tenant_id == get_current_tenant(). create() stamps
    tenant_id from that same context when the caller doesn't pass one; if the
    caller does pass a tenant_id that disagrees with the current tenant, that's
    treated as a bug and raises CrossTenantAccessError immediately, rather than
    silently overwriting it and masking the mistake.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        model = getattr(cls, "model", None)
        if model is not None and not hasattr(model, "tenant_id"):
            raise MissingTenantColumnError(
                f"{model.__name__} has no tenant_id column, so it can't use "
                "TenantScopedRepository's automatic filtering. Models scoped "
                "indirectly through a parent (e.g. Message via Conversation) "
                "should subclass BaseRepository directly and enforce isolation "
                "explicitly — see MessageRepository for the pattern."
            )

    def _resolve_tenant_id(self, provided: uuid.UUID | None) -> uuid.UUID:
        current_tenant_id = get_current_tenant()
        if provided is not None and provided != current_tenant_id:
            raise CrossTenantAccessError(
                f"tenant_id={provided} does not match the current tenant ({current_tenant_id})"
            )
        return current_tenant_id

    async def get(self, id_: uuid.UUID) -> ModelType | None:
        return await self._get(id=id_, tenant_id=get_current_tenant())

    async def list(self, **filters: Any) -> Sequence[ModelType]:
        filters["tenant_id"] = self._resolve_tenant_id(filters.get("tenant_id"))
        return await self._list(**filters)

    async def create(self, **values: Any) -> ModelType:
        values["tenant_id"] = self._resolve_tenant_id(values.get("tenant_id"))
        return await self._create(**values)
