from app.api.admin import router as admin_router
from app.api.auth import router as auth_router
from app.api.dashboard import router as dashboard_router
from app.api.telegram_webhook import router as telegram_webhook_router
from app.api.webapp import router as webapp_router
from app.api.webhook import router as webhook_router

__all__ = [
    "admin_router",
    "auth_router",
    "dashboard_router",
    "telegram_webhook_router",
    "webapp_router",
    "webhook_router",
]
