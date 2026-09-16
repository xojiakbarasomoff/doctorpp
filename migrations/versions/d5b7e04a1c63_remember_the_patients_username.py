"""remember the patient's username

The dashboard's conversation list had nothing to call a patient but the
platform id: "17841467434701445". Operators do not recognise that, cannot
search for it, and cannot match it to the person in front of them -- so the
one screen the clinic uses all day identified everybody by a number nobody
had ever seen.

Instagram gives the account's own messaging API the sender's username on
request, and Telegram puts it in the update. Neither is a fact about the
conversation, so it is stored on the patient row it belongs to.

Nullable, and it stays nullable. A patient whose lookup failed, or whose
Instagram account has no username set, is still a patient the clinic must be
able to see; the column filling in later (app.services.profile) is the normal
case rather than a repair.

The index is what the search box runs on. Lower(username) rather than
username, so "Asomov" finds "asomov" -- Instagram usernames are lowercase
but the operator typing one has no reason to know that.

Revision ID: d5b7e04a1c63
Revises: c4e8b1a92f7d
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d5b7e04a1c63"
down_revision: str | Sequence[str] | None = "c4e8b1a92f7d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("users", sa.Column("username", sa.String(length=255), nullable=True))
    op.create_index(
        "ix_users_tenant_username_lower",
        "users",
        ["tenant_id", sa.text("lower(username)")],
    )


def downgrade() -> None:
    op.drop_index("ix_users_tenant_username_lower", table_name="users")
    op.drop_column("users", "username")
