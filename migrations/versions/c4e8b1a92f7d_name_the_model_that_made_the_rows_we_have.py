"""name the model that made the rows we have

a7d3f21c9b04 added knowledge_base.embedding_model and left every existing row
NULL, on the reasoning that NULL means "made by something we can no longer
name" and should be repaired by re-embedding. On this deployment that reading
was wrong in the expensive direction: the rows were made by
text-embedding-3-small already, and the first startup after the column landed
tried to re-embed all 3522 of them and hit the seeding timeout:

    ERROR app.core.faq_seeding faq_seeding_timed_out path=data/faqs.json
    seconds=60.0

Timed out means nothing was written, so the NULLs survived and every
subsequent deploy would have spent the same minute and the same embedding
calls on work that was already done, and finished no further along.

The rows really are OpenAI's. Neither Railway service carries GEMINI_API_KEY
or any other Gemini credential, and the configuration this replaced refused
to start at all when MODEL_PROVIDER named a provider whose key was missing --
so the deployment that wrote these vectors was on OpenAI, and
text-embedding-3-small is the only embedding model this codebase has ever
asked OpenAI for.

That last part is what makes this safe to state as fact rather than guess, and
it is also the limit of the claim: it is about the rows in this deployment's
database. A deployment that did embed with something else must not run this
migration -- it should let ingest_faqs re-embed instead, out of band rather
than inside startup's timeout.

Revision ID: c4e8b1a92f7d
Revises: a7d3f21c9b04
"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c4e8b1a92f7d"
down_revision: str | Sequence[str] | None = "a7d3f21c9b04"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_MODEL = "text-embedding-3-small"


def upgrade() -> None:
    op.execute(
        f"UPDATE knowledge_base SET embedding_model = '{_MODEL}' WHERE embedding_model IS NULL"
    )


def downgrade() -> None:
    # Only the rows this migration claimed, so a row embedded after it -- by
    # ingest_faqs, which stamps the same name -- is not silently unclaimed and
    # then re-embedded for no reason.
    op.execute(f"UPDATE knowledge_base SET embedding_model = NULL WHERE embedding_model = '{_MODEL}'")
