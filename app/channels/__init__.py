"""Channel adapters, and the registration that makes them findable.

Importing this package is what puts an adapter in the registry, so anything
resolving a channel by type (app.services.delivery) imports from here rather
than from a concrete platform module.

Both channels the product serves are registered below. A third would be one
more line here and one more package beside them, with nothing else in the
codebase changing.
"""

from app.channels.base import (
    ChannelAdapter,
    ChannelType,
    DeliveryBlocked,
    UnknownChannelTypeError,
    get_adapter,
    register_adapter,
    registered_channel_types,
)
from app.channels.instagram.adapter import get_instagram_adapter
from app.channels.telegram.adapter import get_telegram_adapter

register_adapter(get_instagram_adapter())
register_adapter(get_telegram_adapter())

__all__ = [
    "ChannelAdapter",
    "ChannelType",
    "DeliveryBlocked",
    "UnknownChannelTypeError",
    "get_adapter",
    "register_adapter",
    "registered_channel_types",
]
