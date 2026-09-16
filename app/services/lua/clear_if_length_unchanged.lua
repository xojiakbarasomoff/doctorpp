-- Best-effort, single-attempt cleanup used only on the emergency-bypass
-- path: if the buffer's length still matches what was observed a moment
-- ago (ARGV[1]), clear it and bump the generation counter (invalidating
-- any already-scheduled deferred fire job for this user). If the length
-- has changed -- another message raced in since it was observed -- this
-- no-ops rather than deleting a message it never saw: the leftover buffer
-- just fires normally later as an independent, harmless follow-up. Never
-- retried, because nothing here is allowed to delay or block the
-- emergency reply itself, which is enqueued unconditionally by the caller
-- before this runs.
--
-- KEYS[1] = messages key, KEYS[2] = generation key
-- ARGV[1] = expected length
local current_len = redis.call('LLEN', KEYS[1])
if tostring(current_len) ~= ARGV[1] then
    return 0
end
redis.call('DEL', KEYS[1])
redis.call('INCR', KEYS[2])
return 1
