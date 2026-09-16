"""add conversation identity constraints

Makes first contact safe to handle concurrently, now that the inbound
pipeline actually persists patients and conversations (see
app.services.conversation). Two webhook deliveries for a patient's first two
message bubbles can arrive at once, both find no existing row, and both
insert; without these the result is two User rows for one person and a
conversation history split across two transcripts.

Both tables are empty on any deployment upgrading through this: nothing in
the application wrote to users or conversations before the change that
accompanies this migration. If a future deployment ever does hold rows that
violate these, the migration will fail loudly on the duplicates rather than
silently keeping one — which is the right way round for patient records.

Revision ID: d47a1c9e6b30
Revises: f1c4a8b76d2e
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d47a1c9e6b30"
down_revision: str | Sequence[str] | None = "f1c4a8b76d2e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_unique_constraint(
        "uq_users_tenant_channel_external",
        "users",
        ["tenant_id", "channel_id", "external_id"],
    )
    # Partial, so a closed conversation never blocks the patient's next one
    # — only one conversation per patient may be open at a time.
    op.create_index(
        "uq_conversations_open_per_user",
        "conversations",
        ["tenant_id", "user_id"],
        unique=True,
        postgresql_where=sa.text("status = 'open'"),
    )


def downgrade() -> None:
    op.drop_index("uq_conversations_open_per_user", table_name="conversations")
    op.drop_constraint("uq_users_tenant_channel_external", "users", type_="unique")
