"""Reading and writing where a conversation has got to.

Deliberately small. The state is one row, it is read once per turn and
written once per turn, and every write bumps a version so that two workers
answering the same patient at the same moment cannot each save their own
idea of what is happening over the other's.
"""

import logging
import uuid
from datetime import date, time
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tenant_context import get_current_tenant
from app.models.conversation_state import CompletedAction, ConversationState, FlowIntent, FlowStatus

logger = logging.getLogger(__name__)

# "Leave this field alone", as distinct from "set this field to None" --
# clearing the chosen time is what starting a fresh booking has to do, so
# None cannot mean both.
_UNSET: Any = object()


class StaleStateError(RuntimeError):
    """Raised when the row changed under us. The caller re-reads and retries."""


class ConversationStateRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, conversation_id: uuid.UUID) -> ConversationState | None:
        result = await self.session.execute(
            select(ConversationState).where(
                ConversationState.conversation_id == conversation_id,
                ConversationState.tenant_id == get_current_tenant(),
            )
        )
        return result.scalar_one_or_none()

    async def get_or_create(self, conversation_id: uuid.UUID) -> ConversationState:
        existing = await self.get(conversation_id)
        if existing is not None:
            return existing
        state = ConversationState(
            conversation_id=conversation_id,
            tenant_id=get_current_tenant(),
            status=FlowStatus.IDLE,
            intent=FlowIntent.NONE,
            last_completed_action=CompletedAction.NONE,
            version=1,
        )
        self.session.add(state)
        await self.session.flush()
        return state

    async def save(
        self,
        state: ConversationState,
        *,
        status: FlowStatus | None = None,
        intent: FlowIntent | None = None,
        awaiting_field: str | None | object = _UNSET,
        requested_date: date | None | object = _UNSET,
        requested_time: time | None | object = _UNSET,
        reason: str | None | object = _UNSET,
        appointment_id: uuid.UUID | None | object = _UNSET,
        last_completed_action: CompletedAction | None = None,
    ) -> ConversationState:
        """Write the fields given, leave the rest, and bump the version.

        `_UNSET` rather than None as the "don't touch this" marker, because
        None is a value here: clearing the chosen time is exactly what
        starting a new booking has to do.
        """
        values: dict[str, Any] = {}
        if status is not None:
            values["status"] = str(status)
        if intent is not None:
            values["intent"] = str(intent)
        if awaiting_field is not _UNSET:
            values["awaiting_field"] = awaiting_field
        if requested_date is not _UNSET:
            values["requested_date"] = requested_date
        if requested_time is not _UNSET:
            values["requested_time"] = requested_time
        if reason is not _UNSET:
            values["reason"] = reason
        if appointment_id is not _UNSET:
            values["appointment_id"] = appointment_id
        if last_completed_action is not None:
            values["last_completed_action"] = str(last_completed_action)
        if not values:
            return state

        values["version"] = state.version + 1
        result = await self.session.execute(
            update(ConversationState)
            .where(
                ConversationState.conversation_id == state.conversation_id,
                ConversationState.tenant_id == get_current_tenant(),
                ConversationState.version == state.version,
            )
            .values(**values)
            .returning(ConversationState.version)
        )
        if result.scalar_one_or_none() is None:
            raise StaleStateError(
                f"conversation state {state.conversation_id} changed under this worker"
            )
        await self.session.refresh(state)
        return state

    async def clear_flow(
        self, state: ConversationState, *, action: CompletedAction, appointment_id: uuid.UUID | None
    ) -> ConversationState:
        """Finish whatever was running and record what was done.

        Called when the backend has actually acted -- a row written, a row
        cancelled. The collection fields go, the outcome stays: that is what
        stops "rahmat" restarting a form that has already finished.
        """
        return await self.save(
            state,
            status=FlowStatus.IDLE,
            intent=FlowIntent.NONE,
            awaiting_field=None,
            requested_date=None,
            requested_time=None,
            reason=None,
            appointment_id=appointment_id,
            last_completed_action=action,
        )
