"""record which model embedded each row

Vectors are only comparable to vectors from the same model. Until now nothing
recorded which model made a given knowledge_base row, so a deployment that
changed embedding provider kept every old vector and compared new queries
against a space those vectors do not share. Retrieval does not fail loudly
when that happens: every distance simply comes back too far, the bot answers
"I don't know that" to questions it has perfect rows for, and the only sign
is a clinic saying the assistant has gone stupid.

The column is filled in as rows are embedded (app.services.knowledge_base.
ingest_faqs), and a row whose model does not match the one now configured is
re-embedded rather than trusted. Existing rows are left NULL deliberately:
NULL means "made by something we can no longer name", which is exactly the
state of every row written before this migration, and is treated as a
mismatch so they are repaired on the next ingest.

Revision ID: a7d3f21c9b04
Revises: a7d31f0c5b92
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7d3f21c9b04"
down_revision: str | Sequence[str] | None = "a7d31f0c5b92"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_base",
        sa.Column("embedding_model", sa.String(length=100), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("knowledge_base", "embedding_model")
