-- Atomically put a claimed batch back, after the work that claimed it failed.
--
-- pop_if_current_generation deletes both keys, so once a batch is claimed the
-- only copy of the patient's words is in the worker's memory. If the job then
-- raises -- a rate-limited model, a dropped connection -- that copy dies with
-- it and nobody ever answers. This is the other half of that claim.
--
-- Two cases, and the difference matters:
--
--   * The generation key is gone, meaning nothing has arrived since. Recreate
--     it at the value this job was scheduled for, so the retry's pop matches
--     and claims the batch again.
--
--   * The generation key exists, meaning a newer message came in and started
--     a new generation with a job already scheduled for it. Leave the counter
--     alone and put these messages at the HEAD of the list: they were said
--     first, and the newer job will answer all of them together. This job's
--     own retry will then find a generation it does not match and no-op,
--     which is what stops the same words being answered twice.
--
-- KEYS[1] = generation key, KEYS[2] = messages key
-- ARGV[1] = generation to restore
-- ARGV[2] = ttl seconds
-- ARGV[3..] = the messages, oldest first
if redis.call('EXISTS', KEYS[1]) == 0 then
    redis.call('SET', KEYS[1], ARGV[1])
end

-- Pushed back to front so that ARGV[3] ends up at the head and the batch
-- keeps the order the patient typed it in.
for i = #ARGV, 3, -1 do
    redis.call('LPUSH', KEYS[2], ARGV[i])
end

-- Long enough to outlive the retry schedule; without it the restored batch
-- would inherit no expiry at all and outlive the conversation.
redis.call('EXPIRE', KEYS[1], ARGV[2])
redis.call('EXPIRE', KEYS[2], ARGV[2])

return redis.call('LLEN', KEYS[2])
