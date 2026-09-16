-- Atomically: if `generation` (ARGV[1]) still equals the live generation
-- counter, claim the pending message buffer (pop + clear both keys) and
-- return it. Otherwise -- a newer message reset things since this job was
-- scheduled -- return nil and touch nothing. This is what makes a stale,
-- superseded deferred job a cheap no-op instead of double-processing or
-- clobbering state a newer message is still accumulating into.
--
-- KEYS[1] = generation key, KEYS[2] = messages key
-- ARGV[1] = expected generation
local current = redis.call('GET', KEYS[1])
if current == false or current ~= ARGV[1] then
    return false
end
local messages = redis.call('LRANGE', KEYS[2], 0, -1)
redis.call('DEL', KEYS[2])
redis.call('DEL', KEYS[1])
return messages
