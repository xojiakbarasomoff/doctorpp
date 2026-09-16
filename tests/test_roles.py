"""The permission mapping itself, without a request in the way."""

import pytest

from app.core.roles import Permission, Role, has_permission, permissions_for


def test_a_doctor_may_only_read() -> None:
    assert permissions_for(Role.DOCTOR) == frozenset()


def test_the_front_desk_handles_patients_but_not_the_clinic() -> None:
    """The whole reason for a third role: the account that is open all day
    can talk to patients and book them in, and cannot rewrite what the
    assistant tells every patient who asks.
    """
    assert has_permission(Role.OPERATOR, Permission.HANDLE_PATIENTS)
    assert not has_permission(Role.OPERATOR, Permission.MANAGE_CLINIC)
    assert not has_permission(Role.OPERATOR, Permission.MANAGE_STAFF)


def test_an_admin_holds_every_permission_including_ones_added_later() -> None:
    """Spelled as "all of them" in the mapping, so a permission introduced
    later cannot leave the clinic's owner unable to use it.
    """
    assert permissions_for(Role.ADMIN) == frozenset(Permission)


@pytest.mark.parametrize(
    "role",
    ["", "Admin", "ADMIN", "superuser", "viewer", "operator ", "registrator", "0"],
)
def test_an_unrecognised_role_holds_nothing(role: str) -> None:
    """The check this replaced was `if role == "doctor": deny`, so each of
    these — a typo, a case difference, a role invented later, a value
    written straight into the table — was an account with full rights over
    the clinic's data. The mapping is an allow-list precisely so that the
    cost of a mistake is a locked-out account instead.
    """
    assert permissions_for(role) == frozenset()
    for permission in Permission:
        assert not has_permission(role, permission)


def test_every_role_is_in_the_mapping() -> None:
    """A role added to the enum without a line in the mapping would raise a
    KeyError on the first request from an account holding it, rather than
    fail here.
    """
    for role in Role:
        assert isinstance(permissions_for(role), frozenset)
