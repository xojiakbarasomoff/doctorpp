#!/bin/sh
# Before each deploy: make sure this deployment's own database exists on the
# shared Postgres server, then bring its schema up to date.
set -e
python scripts/ensure_database.py
alembic upgrade head
