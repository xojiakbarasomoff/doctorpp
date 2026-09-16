.PHONY: install lint format typecheck test up down migrate

install:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	black .

typecheck:
	mypy app

test:
	pytest

up:
	docker compose up -d

down:
	docker compose down

migrate:
	alembic upgrade head
