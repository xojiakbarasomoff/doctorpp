# Dental Clinic Assistant

A multi-tenant SaaS AI assistant that talks to patients on behalf of dental
clinics, handling conversations, scheduling context, and retrieval-augmented
answers grounded in each clinic's own data.

This is the merged application. It serves **Instagram and Telegram** from
one codebase: one answer pipeline, one conversation store, one database.
Everything a platform knows about itself lives under `app/channels/`;
everything else is platform-neutral and shared. See
[Architecture](#architecture).

`../telegram/` still holds the original Telegram project. **Nothing there
should be deployed any more** — its bot, admin API, Mini App and dashboard
all run through the code here now, against a schema its own models no longer
match. It is kept only as a reference while the merge finishes; once it is
removed, this directory is renamed.

Everything below is relative to this directory — run `make`, `pytest`,
`alembic` and `docker compose` from `instagram/`, not from the repository
root. The one piece deliberately left at the root is
`.github/workflows/ci.yml`: GitHub reads workflows from there and nowhere
else, so it stays put and runs every step with
`working-directory: instagram`.

## Prerequisites

- Python 3.12
- Docker Desktop (with Docker Compose)

## Local setup

1. Clone the repo and create a virtual environment:

   ```bash
   python3.12 -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```

2. Install the project with dev dependencies:

   ```bash
   make install
   ```

3. Copy the environment template and fill in real values:

   ```bash
   cp .env.example .env
   ```

## Running with Docker

Bring up the full stack (api, worker, postgres with pgvector, redis):

```bash
make up
```

Docker Compose reads its own env file, `.env.docker` (copy it from
`.env.docker.example`) — separate from the `.env` you set up above, since
containers need `DATABASE_URL`/`REDIS_URL` to point at the `postgres`/`redis`
service names rather than `localhost`:

```bash
cp .env.docker.example .env.docker
```

Check the API is up:

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

Tear the stack down:

```bash
make down
```

## Deploying

See **[docs/DEPLOY.md](docs/DEPLOY.md)** for the full Railway walkthrough:
four services, the variables each one needs, and what every provisioning log
line means when a step does not do what you expected.

The short version: `scripts/generate_secrets.py` writes a ready-to-paste
variable block, and the app provisions its own tenant, channels and first
dashboard login on startup from those variables. That is not convenience —
on a managed host the database is reachable only from inside the cluster's
private network, so no script run from your machine can write the first rows.

## Connecting a channel

A channel is a clinic's account on one platform. Its credentials live
encrypted in the database, never in the environment — one deployment serves
many clinics, so a single `BOT_TOKEN` variable could only ever be right for
one of them.

The scripts below are the right tool wherever you have a shell onto the
database. Where you do not — Railway, Fly, most managed hosts — set
`PROVISION_TELEGRAM_BOT_TOKEN` / `PROVISION_IG_ACCOUNT_ID` instead and let
startup do the same work; see `app/core/provisioning.py`.

### Telegram

From this directory, with the bot token from @BotFather:

```powershell
$env:TELEGRAM_BOT_TOKEN = "<token from @BotFather>"
$env:PUBLIC_BASE_URL = "https://your-deployment.example.com"
$env:TENANT_NAME = "Smile Dental"
./../.venv/Scripts/python.exe scripts/setup_telegram_channel.py
Remove-Item Env:\TELEGRAM_BOT_TOKEN
```

That checks the token with `getMe`, creates the tenant and channel, and
registers the webhook at `/webhook/telegram/<bot id>` with a generated
secret. Re-running refreshes all three rather than duplicating anything.

The secret is what proves a delivery came from Telegram. It is stored in
`Channel.config` and verified on every update — a channel without one
**refuses every delivery**, deliberately: an endpoint that accepts
unauthenticated updates lets anyone write into a clinic's patient
transcript.

### Instagram

`scripts/bootstrap_tenant.py` creates the tenant and channel;
`scripts/set_channel_credentials.py` stores the access token once Meta
issues it. Meta's webhook is configured in the app dashboard, pointing at
`/webhook` and verified with `WEBHOOK_VERIFY_TOKEN` / `META_APP_SECRET`.

## Running tests

```bash
make test
```

Lint, format, and type-check:

```bash
make lint
make format
make typecheck
```

## The dashboard API

`/api/admin/*` is the JSON API the operator dashboard runs on, and
`/api/webapp/{bot_id}/book` is what the Telegram Mini App posts to. Both were
ported from the Telegram project; both had a hole that is closed here.

**Which clinic you are is never a request parameter.** Admin endpoints take
the tenant from the logged-in operator's own row, and every query filters on
it. The endpoints these replace took `tenant_id: int = Query(1)`, so any
authenticated operator could read and modify another clinic's patients,
appointments and conversations by changing a number in the URL.

Authentication is the session cookie the dashboard already used — signed,
`HttpOnly`, rate-limited at login. A client starts by calling
`GET /api/admin/session`, which returns who it is logged in as and a CSRF
token; every write echoes that token back in `X-CSRF-Token`. A `doctor`
account can read everything and change nothing.

| Area | Endpoints |
| --- | --- |
| Session | `GET /api/admin/session` |
| Conversations | list, detail, `POST .../bot` (take over / hand back), `POST .../reply` |
| Appointments | list by day, create, cancel, confirm |
| Doctors | list, create, patch |
| Leads | list, create, patch |
| Knowledge base | list, create (embeds it), delete (deactivates) |
| Settings | get, patch (merges) |
| Analytics | summary + bookings per day |
| Account | change your own password, export the day as CSV |

The dashboard itself is served at **`/admin`** from this same application,
which is what lets it authenticate with the session cookie rather than
carrying a token in JavaScript where any script on the page could read it.
It has no build step and no CDN: one HTML file, so it keeps working on a
clinic's patchy connection.

An operator's reply goes out through the same delivery service the bot uses,
so it reaches whichever platform the patient wrote in on, routed by the
context captured from their own last message — for a Telegram Business
conversation, back over that same connection.

### Reminders

The worker runs a cron pass every five minutes that reminds patients a day
before and two hours before their appointment, on whichever platform they
booked through. A reminder is marked sent only once it has actually gone
out, so one that could not be delivered is tried again rather than quietly
written off; a booking made at short notice gets the urgent reminder only,
not both at once.

### The Mini App

`POST /api/webapp/{bot_id}/book` verifies Telegram's signature over the
`initData` string before writing anything, and takes both the clinic and the
patient from it. The endpoint it replaces verified nothing at all: anyone who
found the URL could create appointments in any clinic's calendar, attributed
to any patient they named.

## Architecture

The codebase is split along one line: **what a messaging platform knows about
itself** versus **what the clinic's assistant does**. Only the first half is
Instagram-specific.

```
app/
├── channels/       # PLATFORM-SPECIFIC — the only code that names a platform
│   ├── base.py     #   ChannelAdapter contract + registry
│   └── instagram/  #   Graph API client, 24h window, placeholder tokens
├── api/            # FastAPI routers; webhook.py is Instagram's inbound edge
├── core/           # settings, db, encryption, logging, tenant context
├── models/         # SQLAlchemy models
├── repositories/   # tenant-scoped data access
├── services/       # SHARED business logic (see below)
├── rag/            # retrieval-augmented generation pipeline
├── workers/        # ARQ background jobs
└── main.py         # FastAPI app entrypoint

migrations/         # Alembic migrations
infra/              # infrastructure config
tests/              # pytest test suite
```

### The inbound path

```
Instagram webhook   (app/api/webhook.py)          <- platform-specific
Telegram  webhook   (app/api/telegram_webhook.py)  <- platform-specific
        |   verify signature, parse payload, resolve channel
        v
idempotency.claim_event                      <- shared: drop redeliveries
        |
        v
conversation.register_inbound_message        <- shared: user/conversation/message
        |   stops here if an operator has taken the conversation over
        v
debounce.handle_inbound_message              <- shared: batch bubbles, catch emergencies
        |
        v
workers.tasks.process_inbound_message        <- shared: the ARQ job
        |
        +--> answer.generate_answer          <- shared: guardrail, RAG, prompt, LLM
        |
        +--> delivery.send_reply             <- shared: dispatch by channel type
                     |
                     v
             ChannelAdapter.send_text        <- platform-specific again
```

Everything between the two platform-specific ends is written against a
channel id and a platform-issued user id, never against anything one
platform shaped. Both channels run this exact path; a third would too.

One thing does cross it: `reply_context`, an opaque mapping the inbound edge
captures and the adapter reads back, carried through the queue untouched by
everything in between. It exists because some platforms route a reply by
more than the recipient's id — a Telegram conversation reached through
Telegram Business must be answered over that same business connection, or
the reply goes out from the bot account instead of the clinic's own.

### Adding a channel

1. Implement `ChannelAdapter` (`app/channels/base.py`): `send_text`, plus
   `delivery_block_reason` if the platform restricts unsolicited replies.
2. Register it in `app/channels/__init__.py`.
3. Add an inbound route that verifies the platform's own authentication,
   resolves the channel with `tenant_resolution.resolve_channel`, claims the
   event id with `idempotency.claim_event`, records it with
   `conversation.register_inbound_message`, and hands it to
   `debounce.handle_inbound_message`.

Nothing else changes. `Channel.type` already carries the value, the ARQ jobs
already take a channel id, and the Redis keys are already namespaced per
channel.

### Multi-tenancy

Every tenant-scoped read and write goes through a repository that filters on
the tenant bound to the current request or job (`app/core/tenant_context.py`).
A tenant is established from the channel an inbound event names, or from the
logged-in operator — never from anything the caller supplies. See
`app/repositories/base.py` and `tests/test_tenant_isolation.py`.
