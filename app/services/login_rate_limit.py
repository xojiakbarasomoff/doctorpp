from arq.connections import ArqRedis

# Keyed by the submitted username, not IP: the goal is protecting one
# operator account from having its password brute-forced, which is what
# matters regardless of how many IPs an attacker spreads attempts across.
# Locking out a non-existent username too (same as a real one) is
# deliberate — it keeps rate-limiting from becoming a second way to tell
# whether a username exists, on top of authenticate_operator's own
# same-error-either-way behavior.
MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 15 * 60


def _attempts_key(username: str) -> str:
    return f"login_attempts:{username}"


async def is_login_rate_limited(pool: ArqRedis, username: str) -> bool:
    count = await pool.get(_attempts_key(username))
    return count is not None and int(count) >= MAX_LOGIN_ATTEMPTS


async def record_failed_login(pool: ArqRedis, username: str) -> None:
    key = _attempts_key(username)
    count = await pool.incr(key)
    if count == 1:
        # Only set on the first failure in a window — an EXPIRE on every
        # call would keep pushing the lockout further out on each new
        # attempt, turning a 15-minute window into an indefinitely
        # sliding one under sustained brute-forcing.
        await pool.expire(key, LOGIN_LOCKOUT_SECONDS)


async def reset_login_attempts(pool: ArqRedis, username: str) -> None:
    await pool.delete(_attempts_key(username))
