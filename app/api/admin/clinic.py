"""The clinic's own records: doctors, leads, FAQs and settings.

Everything here is tenant-scoped through the repositories, which take the
tenant from the logged-in operator rather than from the request — see
app.api.admin.deps for what that replaces.
"""

import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.admin.deps import require_manage_clinic, require_patient_access, verify_csrf_header
from app.api.admin.schemas import (
    DoctorCreate,
    DoctorOut,
    DoctorUpdate,
    FaqCreate,
    FaqOut,
    LeadCreate,
    LeadOut,
    LeadUpdate,
    TenantSettings,
)
from app.api.auth import get_current_operator
from app.core.db import get_db_session
from app.core.tenant_context import get_current_tenant
from app.models.knowledge_base import KnowledgeBase
from app.models.operator import Operator
from app.models.tenant import Tenant
from app.repositories.doctor import DoctorRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.lead import LeadRepository
from app.services.knowledge_base import FAQImport, ingest_faqs

router = APIRouter(prefix="/api/admin", tags=["Admin — Clinic"])


# --- doctors ---------------------------------------------------------------


def _doctor_out(doctor: Any) -> DoctorOut:
    return DoctorOut(
        id=doctor.id,
        name=doctor.name,
        specialty=doctor.specialty,
        phone=doctor.phone,
        working_hours=doctor.working_hours,
        is_active=doctor.is_active,
    )


@router.get("/doctors", response_model=list[DoctorOut])
async def list_doctors(
    operator: Operator = Depends(get_current_operator),
    session: AsyncSession = Depends(get_db_session),
    include_inactive: bool = Query(default=False),
) -> list[DoctorOut]:
    repo = DoctorRepository(session)
    doctors = await repo.list() if include_inactive else await repo.list_active()
    return [_doctor_out(doctor) for doctor in doctors]


@router.post(
    "/doctors",
    response_model=DoctorOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_csrf_header)],
)
async def create_doctor(
    payload: DoctorCreate,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> DoctorOut:
    doctor = await DoctorRepository(session).create(**payload.model_dump())
    await session.commit()
    return _doctor_out(doctor)


@router.patch(
    "/doctors/{doctor_id}", response_model=DoctorOut, dependencies=[Depends(verify_csrf_header)]
)
async def update_doctor(
    doctor_id: uuid.UUID,
    payload: DoctorUpdate,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> DoctorOut:
    repo = DoctorRepository(session)
    doctor = await repo.get(doctor_id)
    if doctor is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Shifokor topilmadi")

    # exclude_unset, so a form that only edits the phone does not blank the
    # working hours by sending them as null.
    await repo.update(doctor, **payload.model_dump(exclude_unset=True))
    await session.commit()
    return _doctor_out(doctor)


# --- leads -----------------------------------------------------------------


def _lead_out(lead: Any) -> LeadOut:
    return LeadOut(
        id=lead.id,
        patient_name=lead.patient_name,
        phone=lead.phone,
        topic=lead.topic,
        convenient_time=lead.convenient_time,
        status=lead.status,
        notes=lead.notes,
        created_at=lead.created_at,
    )


@router.get("/leads", response_model=list[LeadOut])
async def list_leads(
    operator: Operator = Depends(get_current_operator),
    session: AsyncSession = Depends(get_db_session),
    status_filter: str | None = Query(default=None, alias="status"),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[LeadOut]:
    leads = await LeadRepository(session).list_recent(status=status_filter, limit=limit)
    return [_lead_out(lead) for lead in leads]


@router.post(
    "/leads",
    response_model=LeadOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_csrf_header)],
)
async def create_lead(
    payload: LeadCreate,
    operator: Operator = Depends(require_patient_access),
    session: AsyncSession = Depends(get_db_session),
) -> LeadOut:
    lead = await LeadRepository(session).create(**payload.model_dump())
    await session.commit()
    return _lead_out(lead)


@router.patch(
    "/leads/{lead_id}", response_model=LeadOut, dependencies=[Depends(verify_csrf_header)]
)
async def update_lead(
    lead_id: uuid.UUID,
    payload: LeadUpdate,
    operator: Operator = Depends(require_patient_access),
    session: AsyncSession = Depends(get_db_session),
) -> LeadOut:
    repo = LeadRepository(session)
    lead = await repo.get(lead_id)
    if lead is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lid topilmadi")

    await repo.update(lead, **payload.model_dump(exclude_unset=True))
    await session.commit()
    return _lead_out(lead)


# --- knowledge base --------------------------------------------------------


def _faq_out(faq: KnowledgeBase) -> FaqOut:
    return FaqOut(
        id=faq.id,
        question=faq.question,
        answer=faq.answer,
        category=faq.category,
        is_active=faq.is_active,
    )


@router.get("/knowledge-base", response_model=list[FaqOut])
async def list_faqs(
    operator: Operator = Depends(get_current_operator),
    session: AsyncSession = Depends(get_db_session),
) -> list[FaqOut]:
    stmt = (
        select(KnowledgeBase)
        .where(KnowledgeBase.tenant_id == get_current_tenant())
        .order_by(KnowledgeBase.question)
    )
    return [_faq_out(faq) for faq in (await session.execute(stmt)).scalars()]


@router.post(
    "/knowledge-base",
    response_model=FaqOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(verify_csrf_header)],
)
async def create_faq(
    payload: FaqCreate,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> FaqOut:
    """Add or update one FAQ.

    Goes through ingest_faqs rather than the repository directly, because a
    row without an embedding is invisible to retrieval — the assistant would
    show it in the dashboard and never use it to answer anything. Editing an
    existing question updates it in place.
    """
    [faq] = await ingest_faqs(session, [FAQImport(**payload.model_dump())])
    await session.commit()
    return _faq_out(faq)


@router.delete(
    "/knowledge-base/{faq_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    dependencies=[Depends(verify_csrf_header)],
)
async def deactivate_faq(
    faq_id: uuid.UUID,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> None:
    """Deactivates rather than deletes.

    Retrieval already filters on is_active, and keeping the row means a FAQ
    switched off by mistake can be switched back on without being retyped
    and re-embedded.
    """
    repo = KnowledgeBaseRepository(session)
    faq = await repo.get(faq_id)
    if faq is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="FAQ topilmadi")
    await repo.update(faq, is_active=False)
    await session.commit()


# --- settings --------------------------------------------------------------


@router.get("/settings", response_model=TenantSettings)
async def get_settings(
    operator: Operator = Depends(get_current_operator),
    session: AsyncSession = Depends(get_db_session),
) -> TenantSettings:
    tenant = await session.get(Tenant, get_current_tenant())
    assert tenant is not None  # the operator's own tenant
    return TenantSettings(**{k: v for k, v in tenant.settings.items() if v is not None})


@router.patch(
    "/settings", response_model=TenantSettings, dependencies=[Depends(verify_csrf_header)]
)
async def update_settings(
    payload: TenantSettings,
    operator: Operator = Depends(require_manage_clinic),
    session: AsyncSession = Depends(get_db_session),
) -> TenantSettings:
    """Merges, rather than replaces.

    Only the fields actually sent are changed, so a form that edits the
    address cannot blank the phone numbers by omitting them — and a setting
    added by a later version is not dropped by an older dashboard.
    """
    tenant = await session.get(Tenant, get_current_tenant())
    assert tenant is not None
    merged = {**tenant.settings, **payload.model_dump(exclude_unset=True)}
    tenant.settings = merged
    await session.commit()
    return TenantSettings(**{k: v for k, v in merged.items() if v is not None})
