"""Named views of the dashboard's lists, saved by the clinic itself.

"Bugungi Telegram qabullari", "Javobsiz lidlar" -- the questions a clinic
asks its own data every morning. Each new one used to be a developer's job:
a query parameter, a control, a deploy. This lets the clinic write them
down once and keep them.

A saved filter is a set of query parameters for a list the API already
serves, not a query. Applying one means calling the same endpoint a person
clicking the controls calls, with the same parameters -- so there is no
second query path that could disagree with the first, and nothing a clinic
saves ever reaches the database as anything but a value.

Parameters are validated against the list they claim to be for and anything
unrecognised is dropped, so the worst a bad filter can do is show the wrong
rows. That check is here rather than in the dashboard because the dashboard
is not a security boundary.
"""

import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_manage_clinic, verify_csrf_header
from app.api.auth import get_current_operator
from app.core.db import get_db_session
from app.models.operator import Operator
from app.models.saved_filter import SavedFilter
from app.repositories.saved_filter import SavedFilterRepository

router = APIRouter(prefix="/api/admin/filters", tags=["Admin — Saved filters"])

Resource = Literal["appointments", "leads", "conversations"]

# What each list actually accepts, taken from the endpoints themselves
# (app.api.admin.schedule, .clinic, .conversations). Kept here as an
# allow-list rather than passing whatever was saved: a filter is written by
# a person typing into a browser, and the parameters it can carry should be
# the ones the endpoint would have accepted from that browser anyway.
ALLOWED_PARAMS: dict[str, frozenset[str]] = {
    "appointments": frozenset({"date", "days", "status", "source"}),
    "leads": frozenset({"status", "limit"}),
    "conversations": frozenset({"status", "only_taken_over", "limit"}),
}

# A filter is a handful of controls, not a document. The cap is here so a
# saved view cannot become a way to store arbitrary data on the clinic's row.
MAX_PARAMS = 12
MAX_VALUE_LENGTH = 120


class FilterOut(BaseModel):
    id: uuid.UUID
    resource: Resource
    name: str
    params: dict[str, Any]
    position: int


class FilterCreate(BaseModel):
    resource: Resource
    name: str = Field(min_length=1, max_length=80)
    params: dict[str, Any] = Field(default_factory=dict)
    position: int = Field(default=0, ge=0, le=999)

    @field_validator("name")
    @classmethod
    def _tidy(cls, value: str) -> str:
        collapsed = " ".join(value.split())
        if not collapsed:
            raise ValueError("Nom bo'sh bo'lishi mumkin emas")
        return collapsed


class FilterUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=80)
    params: dict[str, Any] | None = None
    position: int | None = Field(default=None, ge=0, le=999)


def _clean_params(resource: str, params: dict[str, Any]) -> dict[str, Any]:
    """Keep the parameters this list understands, drop the rest.

    Dropping rather than refusing: a filter saved before a list gained or
    lost a control should keep working, showing the rows it still can,
    instead of becoming an error the clinic cannot fix from the interface.
    """
    allowed = ALLOWED_PARAMS[resource]
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if key not in allowed or value is None or value == "":
            continue
        if isinstance(value, str) and len(value) > MAX_VALUE_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"'{key}' qiymati juda uzun",
            )
        if not isinstance(value, str | int | float | bool):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"'{key}' uchun oddiy qiymat kutilgan",
            )
        cleaned[key] = value
        if len(cleaned) > MAX_PARAMS:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Filtrda juda ko'p shart bor",
            )
    return cleaned


def _out(saved: SavedFilter) -> FilterOut:
    return FilterOut(
        id=saved.id,
        resource=saved.resource,
        name=saved.name,
        params=saved.params,
        position=saved.position,
    )


@router.get("", response_model=list[FilterOut])
async def list_filters(
    operator: Operator = Depends(get_current_operator),
    session: AsyncSession = Depends(get_db_session),
) -> list[FilterOut]:
    """Every staff account reads these: a view is only useful if the people
    working the lists can click it. Creating and deleting is the admin's.
    """
    result = await session.execute(
        select(SavedFilter)
        .where(SavedFilter.tenant_id == operator.tenant_id)
        .order_by(SavedFilter.resource, SavedFilter.position, SavedFilter.name)
    )
    return [_out(saved) for saved in result.scalars()]


@router.post(
    "",
    response_model=FilterOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_csrf_header)],
)
async def create_filter(
    payload: FilterCreate,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> FilterOut:
    try:
        saved = await SavedFilterRepository(session).create(
            resource=payload.resource,
            name=payload.name,
            params=_clean_params(payload.resource, payload.params),
            position=payload.position,
        )
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Bu ro'yxatda shu nomli filtr allaqachon bor",
        ) from exc
    return _out(saved)


@router.patch("/{filter_id}", response_model=FilterOut, dependencies=[Depends(verify_csrf_header)])
async def update_filter(
    filter_id: uuid.UUID,
    payload: FilterUpdate,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> FilterOut:
    repo = SavedFilterRepository(session)
    saved = await repo.get(filter_id)
    if saved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Filtr topilmadi")

    changes = payload.model_dump(exclude_unset=True)
    if changes.get("name") is not None:
        saved.name = " ".join(str(changes["name"]).split())
    if changes.get("params") is not None:
        saved.params = _clean_params(saved.resource, changes["params"])
    if changes.get("position") is not None:
        saved.position = int(changes["position"])

    try:
        await session.commit()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Bu ro'yxatda shu nomli filtr allaqachon bor",
        ) from exc
    return _out(saved)


@router.delete(
    "/{filter_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_csrf_header)],
)
async def delete_filter(
    filter_id: uuid.UUID,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    repo = SavedFilterRepository(session)
    saved = await repo.get(filter_id)
    if saved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Filtr topilmadi")
    await session.delete(saved)
    await session.commit()
