r"""Ask the knowledge base what it would retrieve for a patient's question.

The assistant refuses to quote a price whenever retrieve_relevant_faqs comes
back empty, and from the outside that refusal looks the same whatever caused
it: a missing row, a wrong tenant, an embedding built by a provider the rows
were not embedded with, or a real row the query simply lands too far from.
This prints the distances so the four can be told apart.

It reads; it writes nothing. Run it wherever the database and the embedding
provider are both reachable -- on a managed host that means the deployed
container's own shell, since the database is on a private network:

    python scripts/probe_retrieval.py "buyrak uzi qancha"

Several questions at once are fine, and are what makes the output worth
reading: asking the same service in Uzbek and in the Russian the price list
is written in shows whether the gap is the language or the row.

Distances are pgvector's cosine distance, lower being closer. The line the
assistant actually acts on is DEFAULT_MAX_DISTANCE -- matches beyond it are
printed too, marked, because "0.34, just past the cutoff" and "0.71, nothing
like it" call for opposite fixes.
"""

import asyncio
import os
import sys

from app.core.db import db_session
from app.core.faq_seeding import FaqSeedingError, resolve_faq_tenant_id
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.rag.embeddings import get_embedding_provider
from app.rag.retrieval import DEFAULT_MAX_DISTANCE, retrieve_relevant_faqs

LIMIT = 5


async def main() -> None:
    questions = sys.argv[1:]
    if not questions:
        sys.exit(f'Usage: {sys.argv[0]} "<question>" ["<question>" ...]')

    provider = get_embedding_provider()
    print(f"embedding provider: {type(provider).__name__}")
    print(f"cutoff: {DEFAULT_MAX_DISTANCE}\n")

    async with db_session() as session:
        try:
            tenant_id = await resolve_faq_tenant_id(session, os.environ.get("IG_ACCOUNT_ID"))
        except FaqSeedingError as exc:
            sys.exit(str(exc))

        token = set_current_tenant(tenant_id)
        try:
            for question in questions:
                # max_distance=None: the cutoff is what is being measured, so
                # it must not also be what hides the evidence.
                matches = await retrieve_relevant_faqs(
                    session, question, limit=LIMIT, max_distance=None
                )
                print(f"? {question}")
                if not matches:
                    print("  (the knowledge base is empty for this tenant)")
                for match in matches:
                    mark = " " if match.distance <= DEFAULT_MAX_DISTANCE else "x"
                    print(f"  {mark} {match.distance:.4f}  {match.knowledge_base.question}")
                print()
        finally:
            reset_current_tenant(token)


if __name__ == "__main__":
    asyncio.run(main())
