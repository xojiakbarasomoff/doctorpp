import uuid
from contextvars import ContextVar, Token


class TenantContextError(RuntimeError):
    """Raised when tenant-scoped code runs without a tenant set in context."""


# ContextVar (not threading.local) so each asyncio Task/request gets its own
# isolated tenant value, even though tasks share the same OS thread.
_current_tenant_id: ContextVar[uuid.UUID | None] = ContextVar("current_tenant_id", default=None)


def set_current_tenant(tenant_id: uuid.UUID) -> Token[uuid.UUID | None]:
    """Bind tenant_id to the current context (request/task).

    Returns a token for reset_current_tenant(), so callers such as request
    middleware can restore the previous value once the request/task ends.
    """
    return _current_tenant_id.set(tenant_id)


def reset_current_tenant(token: Token[uuid.UUID | None]) -> None:
    _current_tenant_id.reset(token)


def get_current_tenant() -> uuid.UUID:
    """Return the tenant bound to the current context.

    Raises TenantContextError instead of returning None, so a missing tenant
    fails loudly instead of letting a query silently run unscoped.
    """
    tenant_id = _current_tenant_id.get()
    if tenant_id is None:
        raise TenantContextError(
            "No tenant is set in the current context. Call set_current_tenant() "
            "before performing tenant-scoped operations (e.g. in request "
            "middleware or at the start of a worker job)."
        )
    return tenant_id
