"""Letting the clinic's own account change how the assistant answers, by DM.

The clinic's admin does not open the dashboard. They are on Instagram all
day, so something they want the assistant to do -- "always mention the
Saturday clinic", "never say we do IVF", "UZI is 200 000 now" -- reaches it
fastest from the same place they noticed it was needed.

    //: shanba kuni ishlamaymiz

What follows the keyword is an instruction, not yet a rule. The model reads
it first (app.services.rule_interpreter) and works out what it means: the
standing rules it implies, the facts patients will ask about, and the
existing rules it replaces. Only that is stored -- the rules in
tenants.settings.strict_rules, which app.services.answer puts in front of
the model on every reply, and the facts in the knowledge base. The dashboard
shows both and can edit or delete them.

Two things this is deliberately not.

It is not a way around the medical guard. The interpreter refuses an
instruction that would have the assistant diagnose, name a medicine or a
dose, or claim to be a clinician -- because this door opens from a phone,
and whoever holds that account must not be one sentence away from that.

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
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tenant import Tenant
from app.rag.embeddings import EmbeddingProvider
from app.rag.llm import LLMProvider
from app.services.appointment import CLINIC_TIMEZONE
from app.services.knowledge_base import (
    MAX_RULES,
    FAQImport,
    ingest_faqs,
    record_rule_in_knowledge_base,
)
from app.services.rule_interpreter import (
    MAX_RULE_LENGTH,
    Fact,
    Interpretation,
    interpret,
    verbatim,
)
from app.services.tenant_resolution import clinic_rules

logger = logging.getLogger(__name__)

# What an admin may type after the keyword. Longer than a rule, because it is
# now read and condensed rather than pasted in front of every reply; still
# short enough that it is a message and not a document.
MAX_INSTRUCTION_LENGTH = 1500


# Instagram refuses a text over 1000 characters.
_MAX_REPLY_LENGTH = 900

FACT_CATEGORY = "Admin ko'rsatmasi"


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
    """The instruction inside an admin's message, or None if this is not one.

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
    return rule[:MAX_INSTRUCTION_LENGTH]


def _stored_rules(tenant: Tenant) -> list[str]:
    raw = tenant.settings.get("strict_rules")
    if not isinstance(raw, list):
        return []
    return [rule for rule in raw if isinstance(rule, str) and rule.strip()]


@dataclass(frozen=True)
class RuleChange:
    added: tuple[str, ...]
    removed: tuple[str, ...]
    rules: tuple[str, ...]
    full: bool = False


async def change_rules(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    add: Sequence[str],
    remove: Sequence[str],
) -> RuleChange:
    """Remove and add rules in the clinic's settings, and return what changed.

    The tenant row is locked for the read-modify-write: two admins sending an
    instruction within a second of each other would otherwise each write back
    a list that lacks the other's rule. Rules to remove are matched by their
    text rather than by number, because the number was true when the model
    read the list and the dashboard may have edited it since.

    Written the way the dashboard writes settings -- merge the one key, leave
    the rest -- so a rule set from a phone and a rule set from the settings
    screen are the same rule in the same place.
    """
    tenant = await session.get(Tenant, tenant_id, with_for_update=True)
    if tenant is None:
        return RuleChange((), (), ())
    current = _stored_rules(tenant)
    removed = [rule for rule in remove if rule in current]
    kept = [rule for rule in current if rule not in removed]

    seen = {rule.casefold() for rule in kept}
    added: list[str] = []
    full = False
    for rule in add:
        rule = rule.strip()[:MAX_RULE_LENGTH]
        if not rule or rule.casefold() in seen:
            continue
        if len(kept) + len(added) >= MAX_RULES:
            full = True
            break
        added.append(rule)
        seen.add(rule.casefold())

    rules = [*kept, *added]
    tenant.settings = {**tenant.settings, "strict_rules": rules}
    await session.flush()
    logger.info(
        "clinic_rules_changed",
        extra={
            "tenant_id": str(tenant_id),
            "added": len(added),
            "removed": len(removed),
            "rule_count": len(rules),
        },
    )
    return RuleChange(tuple(added), tuple(removed), tuple(rules), full)


async def _store_facts(
    session: AsyncSession,
    facts: Sequence[Fact],
    embedding_provider: EmbeddingProvider | None,
) -> bool:
    """Put the facts in the knowledge base. Whether they all went in.

    A savepoint, and its own failure path: an embedding call can fail, and a
    fact that could not be filed must not take the rules the same message
    set down with it.
    """
    if not facts:
        return True
    try:
        async with session.begin_nested():
            await ingest_faqs(
                session,
                [
                    FAQImport(question=fact.question, answer=fact.answer, category=FACT_CATEGORY)
                    for fact in facts
                ],
                embedding_provider,
            )
    except Exception:
        logger.exception("admin_facts_not_stored")
        return False
    return True


def _clip(text: str) -> str:
    if len(text) <= _MAX_REPLY_LENGTH:
        return text
    return text[: _MAX_REPLY_LENGTH - 1].rstrip() + "…"


def _listing(rules: Sequence[str]) -> str:
    if not rules:
        return "Hozircha qoidalar yo'q."
    body = "\n".join(f"{number}. {rule}" for number, rule in enumerate(rules, start=1))
    return _clip(f"Hozirgi qoidalar ({len(rules)} ta):\n{body}")


def _confirmation(
    interpretation: Interpretation,
    change: RuleChange,
    *,
    facts_stored: bool,
) -> str:
    lines: list[str] = []
    changed = bool(
        change.added or change.removed or (interpretation.facts and facts_stored)
    )
    if not interpretation.understood:
        lines.append(
            "Matnni tahlil qila olmadim, shuning uchun yozganingizni aynan shu "
            "ko'rinishda qoida qilib saqladim."
        )
    elif not changed and not change.full:
        # Said outright. A summary that talks about a rule being cancelled,
        # over a list that did not change, is an admin told "done" about
        # something that was not -- the one thing this reply must not do.
        lines.append(
            "Hech narsa o'zgarmadi. "
            + (interpretation.summary or "Bu allaqachon qoidalarda bor.")
        )
    elif interpretation.summary:
        lines.append(f"Tushundim: {interpretation.summary}")

    if change.added:
        lines.append("Eslab qoldim:\n" + "\n".join(f"- {rule}" for rule in change.added))
    if change.removed:
        lines.append(
            "Endi amal qilmayman:\n" + "\n".join(f"- {rule}" for rule in change.removed)
        )
    if interpretation.facts:
        if facts_stored:
            lines.append(
                "Bilimlar bazasiga ham yozdim, bemor so'rasa shundan javob beraman:\n"
                + "\n".join(f"- {fact.question} → {fact.answer}" for fact in interpretation.facts)
            )
        else:
            lines.append("Bilimlar bazasiga yozib bo'lmadi, dashboard'dan qo'shib qo'ying.")
    if change.full:
        lines.append(
            f"Qoidalar soni {MAX_RULES} tadan oshmasligi kerak, ba'zilarini o'chiring "
            "(dashboard → Sozlamalar)."
        )
    lines.append(
        f"Jami {len(change.rules)} ta qoida. Ularni dashboard → Sozlamalar bo'limida "
        "ko'rish, tahrirlash yoki o'chirish mumkin."
    )
    return _clip("\n\n".join(lines))


async def handle_instruction(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    instruction: str,
    provider: LLMProvider,
    embedding_provider: EmbeddingProvider | None = None,
    now: datetime | None = None,
) -> str:
    """Understand an admin's instruction, apply it, and say what was done.

    Returns the reply for the admin and leaves the commit to the caller,
    which sends the reply only after the change is durable: an admin told
    "eslab qoldim" about something that was rolled back is worse than an
    admin told nothing.

    What is said back is built from what actually changed, not from what the
    model said it would change -- the summary is the model's understanding,
    the lists are the facts.
    """
    existing = await clinic_rules(session, tenant_id)
    try:
        interpretation = await interpret(
            instruction, existing, provider, now=now or datetime.now(CLINIC_TIMEZONE)
        )
    except Exception:
        # Never lose an instruction because the model was down or wrote
        # nonsense. The admin's own words are the rule, and they are told so.
        logger.exception("admin_instruction_not_interpreted")
        interpretation = verbatim(instruction)

    logger.info(
        "admin_instruction_interpreted",
        extra={
            "tenant_id": str(tenant_id),
            "kind": interpretation.kind,
            "understood": interpretation.understood,
            "rules": len(interpretation.rules),
            "removals": len(interpretation.remove),
            "facts": len(interpretation.facts),
            "instruction": instruction[:200],
        },
    )

    if interpretation.kind == "list":
        return _listing(existing)
    if interpretation.kind == "clarify":
        return _clip(interpretation.question)
    if interpretation.kind in ("refused", "not_an_instruction"):
        return _clip(
            interpretation.summary
            or "Buni qoida sifatida tushunmadim. Nima qilishim kerakligini aniqroq yozing."
        )

    change = await change_rules(
        session,
        tenant_id=tenant_id,
        add=interpretation.rules,
        remove=[existing[number - 1] for number in interpretation.remove],
    )
    # Each new rule is also filed in the knowledge base, inactive, so it is
    # visible on the screen the clinic reads and never retrieved. See
    # record_rule_in_knowledge_base for why it must stay inactive.
    for rule in change.added:
        try:
            async with session.begin_nested():
                await record_rule_in_knowledge_base(
                    session,
                    rule=rule,
                    position=change.rules.index(rule) + 1,
                    embedding_provider=embedding_provider,
                )
        except Exception:
            logger.exception("admin_rule_not_mirrored")
    facts_stored = await _store_facts(session, interpretation.facts, embedding_provider)
    return _confirmation(interpretation, change, facts_stored=facts_stored)
