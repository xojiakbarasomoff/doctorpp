"""three named staff roles, and only those three

The permission check this accompanies used to read `if role == "doctor":
deny`. Everything else -- a typo, a role invented later, a value written
straight into the table -- was therefore an account with full rights over
the clinic: its knowledge base, its settings, its patient data. The
application now works from an allow-list (app.core.roles), and this puts
the same rule where it cannot be bypassed by anything that writes to the
database without going through it.

Existing rows are promoted, not reclassified. Every current `operator`
account already had every power the new `admin` role has, so mapping it to
`operator` -- which no longer edits the knowledge base or the clinic
settings -- would quietly take away access people are using today, and in
the worst case leave a clinic with nobody who can change its own settings.
Promoting preserves exactly what each account can do right now. Splitting
the front desk back out is then a deliberate act in the dashboard, one
account at a time, by someone who knows who does what.

`doctor` rows are left alone: read-only means the same thing before and
after.

Revision ID: a7f1c93be204
Revises: b93c5e17a204
Create Date: 2026-08-24

"""

from collections.abc import Sequence

from alembic import op

revision: str = "a7f1c93be204"
down_revision: str | Sequence[str] | None = "b93c5e17a204"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Bare "role": the metadata naming convention (app.models.base) expands a
# check constraint to ck_%(table_name)s_%(constraint_name)s, so this is
# what becomes ck_operators_role on the table.
CONSTRAINT = "role"
ROLES = ("admin", "operator", "doctor")


def upgrade() -> None:
    # Order matters: the data has to satisfy the constraint before the
    # constraint exists, or adding it fails on the first row it validates.
    op.execute("UPDATE operators SET role = 'admin' WHERE role = 'operator'")
    # Anything that is neither of the two roles this schema ever had is not
    # a role at all -- it is a value that was slipping past the old check
    # and holding full rights. Read-only is the safe place to put it: the
    # account keeps working, and whoever notices can set the right role.
    op.execute(f"UPDATE operators SET role = 'doctor' WHERE role NOT IN {ROLES}")
    op.create_check_constraint(CONSTRAINT, "operators", f"role IN {ROLES}")


def downgrade() -> None:
    op.drop_constraint("ck_operators_role", "operators", type_="check")
    # The promotion is not undone. Downgrading the schema is not a reason to
    # take an administrator's access away, and the pre-migration value of a
    # promoted row is not recorded anywhere to restore it from.
