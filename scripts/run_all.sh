#!/bin/sh
# The web app and the worker in one container, for a host where this
# deployment has a single service to run in.
#
# If either process exits, the other is stopped and the container exits with
# the failing process's status, so the host's restart policy brings both back
# rather than leaving a web app that accepts webhooks nobody will answer.
set -u

arq app.workers.tasks.WorkerSettings &
worker=$!
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" &
web=$!

trap 'kill "$worker" "$web" 2>/dev/null' TERM INT

while kill -0 "$worker" 2>/dev/null && kill -0 "$web" 2>/dev/null; do
  sleep 5
done

if kill -0 "$worker" 2>/dev/null; then
  wait "$web"; status=$?
  echo "web process exited ($status); stopping worker" >&2
else
  wait "$worker"; status=$?
  echo "worker process exited ($status); stopping web" >&2
fi
kill "$worker" "$web" 2>/dev/null
exit "${status:-1}"
