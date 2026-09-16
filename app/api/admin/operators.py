"""Staff accounts: who can sign in to this clinic's dashboard, and as what.

Before this, accounts existed only where an operator could reach a shell --
scripts/create_operator.py, or PROVISION_OPERATOR_* on the first boot. A
clinic that hired a receptionist had to ask whoever deployed the thing, so
in practice everyone shared one login, which is also the end of any
question about who cancelled an appointment.

Two rules here are about not locking a clinic out of its own dashboard
rather than about permissions, and both are enforced on the server because
the dashboard's own buttons are not a security boundary:

* You cannot demote or delete yourself. Being the person holding
  MANAGE_STAFF is exactly the state in which one wrong click removes the
  only account that could undo it.
* The last remaining admin cannot be demoted or deleted at all. A clinic
  with no admin keeps working -- patients are answered, appointments are
  booked -- and quietly cannot change its own FAQ, settings, or staff ever
  again without someone opening a database console.

Passwords are set here, never read: a hash goes in, and the only way back
out is for the person to sign in with it. Setting somebody's password is
not the same as knowing it.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_manage_staff, verify_csrf_header
from app.core.db import get_db_session
from app.core.passwords import MIN_PASSWORD_LENGTH, hash_password
from app.core.roles import Role
from app.models.operator import Operator
from app.repositories.operator import OperatorRepository

router = APIRouter(prefix="/api/admin/operators", tags=["Admin — Staff"])


class StaffOut(BaseModel):
    id: uuid.UUID
    name: str
    username: str
    role: Role


class StaffCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    username: str = Field(min_length=3, max_length=255, pattern=r"^[a-zA-Z0-9._-]+$")
    password: str = Field(min_length=MIN_PASSWORD_LENGTH)
    role: Role


class StaffUpdate(BaseModel):
    """Every field optional: renaming somebody must not require resending
    their role, and resetting a password must not require either.
    """

    name: str | None = Field(default=None, min_length=1, max_length=255)
    role: Role | None = None
    password: str | None = Field(default=None, min_length=MIN_PASSWORD_LENGTH)


def _staff_out(operator: Operator) -> StaffOut:
    return StaffOut(
        id=operator.id,
        name=operator.name,
        username=operator.username,
        role=Role(operator.role),
    )


async def _admin_count(session: AsyncSession, tenant_id: uuid.UUID) -> int:
    """How many admins this clinic has. Counted in the database rather than
    from a list already loaded, because the answer decides whether an
    account may be demoted and must reflect the row that is about to change.
    """
    return int(
        (
            await session.execute(
                select(func.count())
                .select_from(Operator)
                .where(Operator.tenant_id == tenant_id, Operator.role == Role.ADMIN)
            )
        ).scalar_one()
    )


async def _refuse_if_last_admin(session: AsyncSession, target: Operator) -> None:
    if target.role != Role.ADMIN:
        return
    if await _admin_count(session, target.tenant_id) > 1:
        return
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Bu klinikadagi yagona administrator — avval boshqa administrator tayinlang",
    )


async def _load(session: AsyncSession, operator_id: uuid.UUID) -> Operator:
    """The staff member, or a 404.

    Through the tenant-scoped repository, so another clinic's account is a
    404 rather than a 403 — the same answer as an id that does not exist,
    which is the answer that does not confirm it does.
    """
    target = await OperatorRepository(session).get(operator_id)
    if target is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Xodim topilmadi")
    return target


@router.get("", response_model=list[StaffOut])
async def list_staff(
    operator: Operator = Depends(require_manage_staff),
    session: AsyncSession = Depends(get_db_session),
) -> list[StaffOut]:
    staff = await OperatorRepository(session).list()
    return [_staff_out(member) for member in sorted(staff, key=lambda m: m.username)]


@router.post(
    "",
    response_model=StaffOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_csrf_header)],
)
async def create_staff(
    payload: StaffCreate,
    operator: Operator = Depends(require_manage_staff),
    session: AsyncSession = Depends(get_db_session),
) -> StaffOut:
    # The whole insert is inside the try, not just the commit: the
    # repository flushes as it creates, so a duplicate username raises there
    # rather than at commit time.
    try:
        created = await OperatorRepository(session).create(
            name=payload.name,
            role=payload.role.value,
            username=payload.username,
            password_hash=hash_password(payload.password),
        )
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        # Usernames are unique across every clinic, not per clinic, because
        # the login form has no clinic selector — so this can collide with a
        # name in a clinic the caller cannot see, and the message must not
        # say which.
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Bu login band",
        ) from exc
    return _staff_out(created)


@router.patch("/{operator_id}", response_model=StaffOut, dependencies=[Depends(verify_csrf_header)])
async def update_staff(
    operator_id: uuid.UUID,
    payload: StaffUpdate,
    operator: Operator = Depends(require_manage_staff),
    session: AsyncSession = Depends(get_db_session),
) -> StaffOut:
    target = await _load(session, operator_id)
    changes = payload.model_dump(exclude_unset=True)

    if "role" in changes and changes["role"] != Role(target.role):
        if target.id == operator.id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="O'z rolingizni o'zgartira olmaysiz",
            )
        await _refuse_if_last_admin(session, target)
        target.role = Role(changes["role"]).value

    if changes.get("name") is not None:
        target.name = changes["name"]
    if changes.get("password") is not None:
        target.password_hash = hash_password(changes["password"])

    await session.commit()
    return _staff_out(target)


@router.delete(
    "/{operator_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_csrf_header)],
)
async def delete_staff(
    operator_id: uuid.UUID,
    operator: Operator = Depends(require_manage_staff),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    target = await _load(session, operator_id)
    if target.id == operator.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="O'zingizni o'chira olmaysiz",
        )
    await _refuse_if_last_admin(session, target)
    await session.delete(target)
    await session.commit()
