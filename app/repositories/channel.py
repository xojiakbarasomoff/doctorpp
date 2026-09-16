from app.models.channel import Channel
from app.repositories.base import TenantScopedRepository


class ChannelRepository(TenantScopedRepository[Channel]):
    model = Channel
