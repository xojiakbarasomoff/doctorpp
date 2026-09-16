"""Which clinic staff account may do what.

Three roles, and the split between them follows what a mistake costs rather
than what a job title suggests:

* `doctor` reads. A doctor opens the dashboard to see the day, not to
  rebook it -- that is the front desk's work.
* `operator` is the front desk: it talks to patients, books and cancels
  appointments, and works the leads. This is the account most staff have and
  the one open all day.
* `admin` additionally changes what the clinic *is* -- its FAQ, its
  settings, its doctors, its staff accounts. Editing the knowledge base is
  the reason this is separated: a wrong price there is one row, and the
  assistant then quotes it to every patient who asks, unprompted and
  without a human in the loop. That is not a keystroke that belongs in the
  middle of a busy front-desk shift.

Reading is not a permission here. Any authenticated staff account may read
its own clinic's data, and the tenant scoping (app.api.admin.deps) is what
keeps that from meaning anyone else's.

The mapping is an allow-list, and deliberately so. The check this replaces
was `if role == "doctor": deny`, which meant every role nobody had thought
of -- a typo, a "viewer" invented later, a value written straight into the
database -- silently held full rights over the clinic. Here an unrecognised
role holds nothing, so the failure mode of a mistake is a locked-out
account rather than an unnoticed administrator.
"""

from enum import StrEnum


class Role(StrEnum):
    ADMIN = "admin"
    OPERATOR = "operator"
    DOCTOR = "doctor"


class Permission(StrEnum):
    # Talk to patients and manage their bookings: reply in a conversation,
    # take it off the bot, book or cancel an appointment, work a lead, and
    # export the appointments CSV (which carries patient names and phones).
    HANDLE_PATIENTS = "handle_patients"
    # Change what the clinic tells people: the knowledge base the assistant
    # answers from, the clinic settings, the list of doctors.
    MANAGE_CLINIC = "manage_clinic"
    # Create staff accounts and set their roles.
    MANAGE_STAFF = "manage_staff"


# ADMIN is spelled as "every permission there is" rather than as a list, so
# that a permission added later cannot accidentally leave the clinic's own
# owner unable to use it. Every other role has to be granted a new
# permission deliberately, which is the safe direction for that default.
_PERMISSIONS: dict[Role, frozenset[Permission]] = {
    Role.DOCTOR: frozenset(),
    Role.OPERATOR: frozenset({Permission.HANDLE_PATIENTS}),
    Role.ADMIN: frozenset(Permission),
}


def permissions_for(role: str) -> frozenset[Permission]:
    """What `role` may do. An unrecognised role may do nothing."""
    try:
        known = Role(role)
    except ValueError:
        return frozenset()
    return _PERMISSIONS[known]


def has_permission(role: str, permission: Permission) -> bool:
    return permission in permissions_for(role)
