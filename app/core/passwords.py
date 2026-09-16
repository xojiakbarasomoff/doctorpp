import bcrypt

# Long enough that a guess is not worth attempting against the login
# rate-limiter. The Telegram admin API this replaces accepted four
# characters, and shipped with a hardcoded admin/admin.
#
# Lives here rather than next to the one endpoint that first needed it,
# because the same floor has to hold on every way a password can be set:
# the dashboard's change-password form and the first account a deployment
# provisions for itself. A floor enforced in one place only is not a floor.
MIN_PASSWORD_LENGTH = 10


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time comparison against a stored bcrypt hash — bcrypt.checkpw
    itself is what makes this safe against timing attacks, not anything
    here.
    """
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
