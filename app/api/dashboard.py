"""Redirects the old appointments page to the operator dashboard.

/dashboard/appointments was this application's dashboard before the merge: a
single server-rendered day of appointments and a two-field booking form,
with no link to anything else. The operator dashboard that replaced it is
served from app/static/admin at /admin/ and has the conversations, leads,
doctors, knowledge base, clinic settings and metrics on it -- everything an
operator actually opens the dashboard for. Keeping both meant two interfaces
that looked alike and disagreed about how much the product could do, and the
smaller one was the one people landed on.

The page is gone; the path stays, because bookmarks and browser history
outlive a refactor and a 404 is a worse answer than the right page.

Temporary rather than permanent (301): a permanent redirect is cached by the
browser indefinitely, so restoring anything at /dashboard later would mean
asking every operator to clear their cache. The extra round trip costs
nothing on an internal route.

Only GET is redirected. The old POST endpoints (book, cancel) are not
forwarded anywhere -- /admin/ posts to /api/admin/, with a different shape
and its own CSRF handling, so a form replayed from a stale page must fail
loudly rather than be quietly pointed at an endpoint that cannot honour it.
"""

from fastapi import APIRouter, status
from fastapi.responses import RedirectResponse

router = APIRouter(prefix="/dashboard")

DASHBOARD_URL = "/admin/"


@router.get("", include_in_schema=False)
@router.get("/{_path:path}", include_in_schema=False)
async def moved_to_operator_dashboard(_path: str = "") -> RedirectResponse:
    return RedirectResponse(url=DASHBOARD_URL, status_code=status.HTTP_302_FOUND)
