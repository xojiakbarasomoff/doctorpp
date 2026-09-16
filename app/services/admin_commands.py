"""Letting the clinic's own account change how the assistant answers, by DM.

The clinic's admin does not open the dashboard. They are on Instagram all
day, so a rule they want the assistant to follow -- "always mention the
Saturday clinic", "never say we do IVF" -- reaches it fastest from the same
place they noticed it was needed.

    Aiadm1in: har doim shanba qabulini eslat

Everything after the keyword becomes one of the clinic's standing rules
(tenants.settings.strict_rules), which app.services.answer puts in front of
the model on every reply from then on, and which the dashboard's settings
screen shows and can edit or delete.

Two things this is deliberately not.

It is not a way around the medical guard. The rules block in the prompt says
outright that no clinic rule permits diagnosing, naming a medicine or a dose,
or claiming to be a clinician -- because this door opens from a phone, and
whoever holds that account must not be one sentence away from an assistant
that prescribes.

And it is not authentication. The keyword travels in plain text through
Instagram, and a username can be changed by its owner or, once released,
registered by somebody else. What actually gates this is
ADMIN_INSTAGRAM_USERNAMES, checked against the handle Meta gave us for the
sender (app.services.profile), and the fact that the worst a holder can do is
change wording -- which the dashboard shows and any operator can undo.
"""

import logging
import uuid
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import Tenant

logger = logging.getLogger(__name__)

# Longer than a patient would type by accident and short enough to thumb in.
# A rule is a sentence, not an essay: past this it is a paste, and a paste in
# the prompt is a paste in front of every reply.
MAX_RULE_LENGTH = 400


def is_admin(username: str | None, admins: Sequence[str]) -> bool:
    """Whether this handle is one the clinic nominated.

    Case-insensitive and "@"-insensitive on both sides: the deployment
    variable is typed by a person, and so is the handle it is compared with.
    """
    if not username:
        return False
    wanted = {admin.strip().lstrip("@").lower() for admin in admins if admin.strip()}
    return username.strip().lstrip("@").lower() in wanted


def parse_rule(text: str, keyword: str) -> str | None:
    """The rule inside an admin's message, or None if this is not one.

    The keyword is matched at the start and case-insensitively, because it is
    typed on a phone keyboard that capitalises the first letter for you.
    Anything before it means this is a sentence that merely mentions the
    keyword, not a command -- a patient quoting it back cannot set a rule.
    """
    if not keyword:
        return None
    stripped = text.strip()
    if not stripped.lower().startswith(keyword.strip().lower()):
        return None
    rule = stripped[len(keyword.strip()) :].strip().strip(":").strip()
    if not rule:
        return None
    return rule[:MAX_RULE_LENGTH]


async def add_rule(session: AsyncSession, *, tenant_id: uuid.UUID, rule: str) -> list[str]:
    """Append the rule to the clinic's settings and return the whole list.

    Appended, not replaced: the clinic collects these over time, and a second
    command must not silently drop the first. Duplicates are dropped, so
    sending the same rule twice is not two copies of it in front of every
    reply.

    Written the way the dashboard writes settings -- merge the one key, leave
    the rest -- so a rule set from a phone and a rule set from the settings
    screen are the same rule in the same place.
    """
    tenant = await session.get(Tenant, tenant_id)
    if tenant is None:
        return []
    existing = tenant.settings.get("strict_rules")
    rules = [r for r in existing if isinstance(r, str) and r.strip()] if isinstance(existing, list) else []
    if rule not in rules:
        rules.append(rule)
    tenant.settings = {**tenant.settings, "strict_rules": rules}
    await session.flush()
    logger.info(
        "clinic_rule_added",
        extra={"tenant_id": str(tenant_id), "rule_count": len(rules), "rule": rule},
    )
    return rules
