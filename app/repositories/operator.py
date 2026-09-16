from app.models.operator import Operator
from app.repositories.base import TenantScopedRepository


class OperatorRepository(TenantScopedRepository[Operator]):
    model = Operator
