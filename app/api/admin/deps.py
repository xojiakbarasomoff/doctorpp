"""Shared dependencies for the admin API.

The tenant is never a request parameter here. It comes from the logged-in
operator's own row (app.api.auth.get_current_operator sets it in context for
the lifetime of the request), and every repository read and write filters on
it. The Telegram admin API this replaces took `tenant_id: int = Query(1)` on
every endpoint, so any authenticated operator could read and modify another
clinic's patients, appointments and conversations by changing a number in
the URL.
"""

import secrets
from collections.abc import Awaitable, Callable

from fastapi import Depends, Header, HTTPException, status

from app.api.auth import get_current_operator, get_current_session
from app.core.roles import Permission, has_permission
from app.core.session import Session
from app.models.operator import Operator

# The header a browser fetch sends the session's CSRF token in. The dashboard
# templates use a hidden form field (app.api.auth.verify_csrf); a JSON API
# has no form to hide it in, so the same double-submit check reads a header
# instead. Both compare against the token bound into the signed session
# cookie, which a cross-site caller cannot read.
CSRF_HEADER = "X-CSRF-Token"


async def verify_csrf_header(
    x_csrf_token: str | None = Header(default=None),
    session: Session = Depends(get_current_session),
) -> None:
    if not x_csrf_token or not secrets.compare_digest(x_csrf_token, session.csrf_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing or invalid {CSRF_HEADER}",
        )


def require(permission: Permission) -> Callable[..., Awaitable[Operator]]:
    """Dependency factory gating a route on one permission.

    A factory rather than one dependency per permission so that the route
    declares the capability it needs — `require(Permission.MANAGE_CLINIC)`
    reads as what it is, and adding a permission does not mean adding a
    near-identical function beside three others.

    The check itself lives in app.core.roles: an unrecognised role is
    refused rather than waved through, which is the whole point of the
    mapping being an allow-list.
    """

    async def dependency(operator: Operator = Depends(get_current_operator)) -> Operator:
        if not has_permission(operator.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Bu amal uchun ruxsatingiz yo'q",
            )
        return operator

    return dependency


# Named dependencies, resolved once at import: FastAPI compares Depends()
# markers by identity when it caches a dependency within a request, so
# building a fresh one per route would solve the same check repeatedly.
require_patient_access = require(Permission.HANDLE_PATIENTS)
require_manage_clinic = require(Permission.MANAGE_CLINIC)
require_manage_staff = require(Permission.MANAGE_STAFF)
