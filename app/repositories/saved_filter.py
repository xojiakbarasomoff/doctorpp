from app.models.saved_filter import SavedFilter
from app.repositories.base import TenantScopedRepository


class SavedFilterRepository(TenantScopedRepository[SavedFilter]):
    model = SavedFilter
