from app.repositories.appointment import AppointmentRepository
from app.repositories.base import (
    BaseRepository,
    CrossTenantAccessError,
    MissingTenantColumnError,
    TenantIsolationError,
    TenantScopedRepository,
)
from app.repositories.channel import ChannelRepository
from app.repositories.conversation import ConversationRepository
from app.repositories.doctor import DoctorRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.lead import LeadRepository
from app.repositories.message import MessageRepository
from app.repositories.operator import OperatorRepository
from app.repositories.tenant import TenantRepository
from app.repositories.user import UserRepository

__all__ = [
    "AppointmentRepository",
    "BaseRepository",
    "ChannelRepository",
    "ConversationRepository",
    "CrossTenantAccessError",
    "DoctorRepository",
    "KnowledgeBaseRepository",
    "LeadRepository",
    "MessageRepository",
    "MissingTenantColumnError",
    "OperatorRepository",
    "TenantIsolationError",
    "TenantRepository",
    "TenantScopedRepository",
    "UserRepository",
]
