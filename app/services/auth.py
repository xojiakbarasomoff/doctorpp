from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.passwords import verify_password
from app.models.operator import Operator


async def authenticate_operator(
    session: AsyncSession, username: str, password: str
) -> Operator | None:
    """Verifies a login attempt, or returns None on any failure (unknown
    username or wrong password — never distinguished, so a login form can't
    be used to enumerate valid usernames).

    Runs BEFORE any tenant is known — the username is what determines which
    tenant this operator belongs to — so like
    tenant_resolution.resolve_instagram_channel, this queries Operator
    directly rather than through OperatorRepository/TenantScopedRepository,
    both of which require get_current_tenant() to already have a value.
    """
    result = await session.execute(select(Operator).where(Operator.username == username))
    operator = result.scalar_one_or_none()
    if operator is None:
        return None
    if not verify_password(password, operator.password_hash):
        return None
    return operator
