"""The nightly review's suggestions, for the clinic to accept or reject.

Only a clinic admin sees them: each one quotes patients' conversations, and
accepting one changes what the assistant tells every patient after.
"""

import uuid

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_manage_clinic, verify_csrf_header
from app.api.admin.schemas import (
    ReviewStatus,
    SuggestionAccept,
    SuggestionEvidence,
    SuggestionOut,
)
from app.core.db import get_db_session
from app.core.queue import get_arq_pool
from app.core.tenant_context import get_current_tenant
from app.models.conversation import Conversation
from app.models.operator import Operator
from app.models.suggestion import Suggestion, SuggestionStatus
from app.models.tenant import Tenant
from app.models.user import User
from app.services import review
from app.services.knowledge_base import MAX_RULES, TooManyRulesError

router = APIRouter(prefix="/api/admin/suggestions", tags=["Admin — Suggestions"])

# One manual run per clinic per this many seconds: each run is a few calls to
# a model thinking hard, and a button pressed ten times should not be ten.
RUN_COOLDOWN_SECONDS = 600


async def _out(session: AsyncSession, suggestions: list[Suggestion]) -> list[SuggestionOut]:
    ids = review.conversation_ids(suggestions)
    names: dict[str, str] = {}
    if ids:
        rows = await session.execute(
            select(Conversation.id, User.name, User.username)
            .join(User, User.id == Conversation.user_id)
            .where(Conversation.id.in_(ids), Conversation.tenant_id == get_current_tenant())
        )
        names = {
            str(row.id): (row.name or (f"@{row.username}" if row.username else "Bemor"))
            for row in rows
        }
    return [
        SuggestionOut(
            id=s.id,
            kind=s.kind,
            status=s.status,
            problem=s.problem,
            rule_text=s.rule_text,
            question=s.question,
            answer=s.answer,
            # A conversation deleted since, or not this clinic's, is not linked.
            evidence=[
                SuggestionEvidence(
                    conversation_id=str(e.get("conversation_id")),
                    quote=str(e.get("quote") or ""),
                    patient=names[str(e.get("conversation_id"))],
                )
                for e in s.evidence or []
                if str(e.get("conversation_id")) in names
            ],
            created_at=s.created_at,
            decided_at=s.decided_at,
        )
        for s in suggestions
    ]


async def _own(session: AsyncSession, suggestion_id: uuid.UUID) -> Suggestion:
    suggestion = await session.scalar(
        select(Suggestion)
        .where(Suggestion.id == suggestion_id, Suggestion.tenant_id == get_current_tenant())
        # Two people deciding the same card at once: the second waits, then
        # finds it decided.
        .with_for_update()
    )
    if suggestion is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tavsiya topilmadi")
    return suggestion


@router.get("", response_model=list[SuggestionOut])
async def list_suggestions(
    state: SuggestionStatus = Query(default=SuggestionStatus.PENDING, alias="status"),
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> list[SuggestionOut]:
    rows = (
        await session.execute(
            select(Suggestion)
            .where(Suggestion.tenant_id == get_current_tenant(), Suggestion.status == state)
            .order_by(Suggestion.created_at.desc())
            .limit(100)
        )
    ).scalars()
    return await _out(session, list(rows))


@router.get("/status", response_model=ReviewStatus)
async def review_status(
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> ReviewStatus:
    tenant = await session.get(Tenant, get_current_tenant())
    assert tenant is not None
    return ReviewStatus(
        pending=await review.pending_count(session, tenant.id), last_run=review.last_run(tenant)
    )


@router.post(
    "/{suggestion_id}/accept",
    response_model=SuggestionOut,
    dependencies=[Depends(verify_csrf_header)],
)
async def accept_suggestion(
    suggestion_id: uuid.UUID,
    payload: SuggestionAccept,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> SuggestionOut:
    suggestion = await _own(session, suggestion_id)
    # Locked: accepting rewrites tenants.settings, as the settings screen does.
    tenant = await session.get(Tenant, get_current_tenant(), with_for_update=True)
    assert tenant is not None
    try:
        await review.accept(
            session,
            tenant,
            suggestion,
            operator.id,
            rule_text=payload.rule_text,
            question=payload.question,
            answer=payload.answer,
        )
    except TooManyRulesError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Online qoidalar soni {MAX_RULES} tadan oshmasligi kerak",
        ) from None
    except review.SuggestionError as error:
        code = status.HTTP_409_CONFLICT if suggestion.status != SuggestionStatus.PENDING else 422
        raise HTTPException(status_code=code, detail=str(error)) from None
    return (await _out(session, [suggestion]))[0]


@router.post(
    "/{suggestion_id}/reject",
    response_model=SuggestionOut,
    dependencies=[Depends(verify_csrf_header)],
)
async def reject_suggestion(
    suggestion_id: uuid.UUID,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> SuggestionOut:
    suggestion = await _own(session, suggestion_id)
    try:
        await review.reject(session, suggestion, operator.id)
    except review.SuggestionError as error:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(error)) from None
    return (await _out(session, [suggestion]))[0]


@router.post(
    "/run", status_code=status.HTTP_202_ACCEPTED, dependencies=[Depends(verify_csrf_header)]
)
async def run_now(
    operator: Operator = Depends(require_manage_clinic),
    pool: ArqRedis = Depends(get_arq_pool),
) -> dict[str, str]:
    """Review now rather than tonight. The result lands in the list."""
    tenant_id = str(get_current_tenant())
    if not await pool.set(f"review-run:{tenant_id}", "1", nx=True, ex=RUN_COOLDOWN_SECONDS):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Tahlil yaqinda boshlangan. 10 daqiqadan keyin qayta urinib ko'ring",
        )
    try:
        await pool.enqueue_job("review_tenant", tenant_id)
    except Exception:
        # Not queued, so not a run to wait ten minutes after.
        await pool.delete(f"review-run:{tenant_id}")
        raise
    return {"status": "queued"}
