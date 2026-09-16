r"""Load a clinic's FAQ list from a JSON file into the knowledge_base table.

The JSON is an array of objects with "question", "answer", and an optional
"category" -- see data/faqs.json. The work itself lives in
app.core.faq_seeding, which app.main also calls on startup when SEED_FAQS_FROM
is set; this is the same operation driven from a shell instead, for wherever
one can reach the database.

Idempotent, because app.services.knowledge_base.ingest_faqs matches on the
exact question text: re-running after editing an answer updates that row in
place. Editing a *question* leaves the old row behind, since its text no
longer matches anything in the file -- delete it by hand if that matters.

Which tenant the rows belong to is resolved from the Instagram channel. With
exactly one channel in the database that needs no argument; with several,
name one via IG_ACCOUNT_ID.

Usage, from the repo root:

    python scripts/ingest_faqs.py data/faqs.json
"""

import asyncio
import os
import sys
from pathlib import Path

from app.core.db import db_session
from app.core.faq_seeding import FaqSeedingError, load_faqs, seed_faqs


async def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} <faqs.json>")

    path = Path(sys.argv[1])
    try:
        faqs = load_faqs(path)
    except FaqSeedingError as exc:
        sys.exit(str(exc))

    if not faqs:
        print(f"{path} is empty - nothing to ingest.")
        return

    async with db_session() as session:
        try:
            tenant_id, count = await seed_faqs(
                session, faqs, os.environ.get("IG_ACCOUNT_ID"), reembed_existing=True
            )
        except FaqSeedingError as exc:
            sys.exit(str(exc))

    print(f"Ingested {count} FAQ rows into tenant_id={tenant_id} from {path}.")


if __name__ == "__main__":
    asyncio.run(main())
