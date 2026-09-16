r"""Ask the assistant what it would say, without saying it to anybody.

Everything else in scripts/ inspects the machinery: probe_retrieval.py shows
which rows a question reaches and how far away they are. Nothing showed the
thing a patient actually receives, which is the whole product -- and some of
what governs it cannot be measured in distances at all. Tone, empathy, the
refusal to give medical advice, whether a reply stayed in the patient's
alphabet: those are read, not computed, and until now the only way to read
one was to message the clinic's live Instagram and hope.

This runs the real path -- guardrail, retrieval, the real prompt, the real
model -- and prints the reply. It writes nothing: no message is stored, no
conversation is touched, no appointment is booked, and nothing is delivered
to any channel. The [[BOOK:...]] marker is printed as it comes, because
whether the model emitted one is part of what you are reading for.

    python scripts/dry_run_reply.py "buyragim ogriyapti"
    python scripts/dry_run_reply.py --history "Salom" "EKG qancha"

With --history, every argument but the last is treated as the conversation
so far, alternating patient and assistant starting with the patient. That is
what makes "va narxi qancha?" mean anything.

Each reply costs one model call and one embedding call, so this is a tool for
a handful of messages at a time, not a batch harness.
"""

import argparse
import asyncio
import os
import sys

from app.core.db import db_session
from app.core.faq_seeding import FaqSeedingError, resolve_faq_tenant_id
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.rag.llm import ChatMessage
from app.services.answer import generate_answer


def _history(turns: list[str]) -> list[ChatMessage]:
    """Earlier turns, oldest first, alternating patient then assistant."""
    roles: list[str] = ["user", "assistant"]
    return [
        ChatMessage(role=roles[index % 2], content=text)  # type: ignore[typeddict-item]
        for index, text in enumerate(turns)
    ]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("messages", nargs="+", help="the patient's message, last if --history")
    parser.add_argument(
        "--history",
        action="store_true",
        help="treat all but the last argument as the conversation so far",
    )
    args = parser.parse_args()

    if args.history:
        earlier, latest = _history(args.messages[:-1]), args.messages[-1]
    else:
        earlier, latest = [], args.messages[-1]
        if len(args.messages) > 1:
            sys.exit("Several messages given without --history; pass one, or add --history.")

    async with db_session() as session:
        try:
            tenant_id = await resolve_faq_tenant_id(session, os.environ.get("IG_ACCOUNT_ID"))
        except FaqSeedingError as exc:
            sys.exit(str(exc))

        token = set_current_tenant(tenant_id)
        try:
            for turn in earlier:
                who = "patient" if turn["role"] == "user" else "bot"
                print(f"  {who:8}| {turn['content']}")
            print(f"  {'patient':8}| {latest}")
            reply = await generate_answer(session, latest, history=earlier or None)
            print(f"  {'BOT':8}| {reply}")
            print(f"  {'':8}| ({len(reply)} chars)")
        finally:
            reset_current_tenant(token)


if __name__ == "__main__":
    asyncio.run(main())
