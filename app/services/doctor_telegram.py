"""The doctor's own Telegram bot: their appointments, and a day off in one line.

Not a patient channel. Patients write to the doctor on Instagram; this bot
has exactly one audience, the doctor (and whoever else is named in
DOCTOR_TELEGRAM_USERNAMES), and does three things:

* tells them the moment a new appointment is booked, from any source --
  the dashboard, the web app, the assistant;
* lists the appointments for today or tomorrow on request;
* when they write "bugun ishga bora olmayman", asks once to confirm, then
  cancels that day's appointments and tells every patient it can reach on
  Instagram. The ones it cannot reach -- booked by phone, or outside
  Instagram's 24-hour reply window -- are listed back with their numbers, so
  somebody rings them. A patient who travels across Tashkent to a closed
  door because a message silently failed is the failure this is built
  around.

Runs by long polling from the worker rather than by webhook: the deployment
this was written for sits behind a tunnel whose address changes, and a
webhook pointed at yesterday's address is a bot that has gone quiet without
saying so.
"""

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.db import db_session
from app.core.tenant_context import reset_current_tenant, set_current_tenant
from app.models.appointment import ACTIVE_STATUSES, Appointment, AppointmentStatus
from app.models.message import MessageSender
from app.models.tenant import Tenant
from app.models.user import User
from app.services.appointment import CLINIC_TIMEZONE
from app.services.conversation import last_inbound_at, record_outbound_message, reply_context_for
from app.services.delivery import send_reply

logger = logging.getLogger(__name__)

API = "https://api.telegram.org"
POLL_TIMEOUT_SECONDS = 25

CHATS_KEY = "doctor_telegram_chats"
OFFSET_KEY = "doctor_telegram_offset"
NOTIFIED_UNTIL_KEY = "doctor_telegram_notified_until"

BTN_TODAY = "📅 Bugungi qabullar"
BTN_TOMORROW = "📅 Ertangi qabullar"
BTN_OFF_TODAY = "🚫 Bugun kela olmayman"
BTN_OFF_TOMORROW = "🚫 Ertaga kela olmayman"

KEYBOARD = {
    "keyboard": [
        [{"text": BTN_TODAY}, {"text": BTN_TOMORROW}],
        [{"text": BTN_OFF_TODAY}, {"text": BTN_OFF_TOMORROW}],
    ],
    "resize_keyboard": True,
}

# "bugun ishga bora olmayman", "ertaga kelolmayman", "сегодня не смогу" --
# the ways a doctor actually types it on a phone. Matched loosely on purpose,
# because nothing happens on a match except a question with two buttons.
_ABSENT = re.compile(
    r"(bora?\s*olma|borolma|kela?\s*olma|kelolma|chiqa?\s*olma|chiqolma|ishlamayman|"
    r"kelmayman|bormayman|не\s*смогу|не\s*приду|не\s*выйду|не\s*буду)",
    re.IGNORECASE,
)
_TOMORROW = re.compile(r"(ertaga|завтра)", re.IGNORECASE)


@dataclass(frozen=True)
class Booking:
    appointment: Appointment
    patient: User | None

    @property
    def name(self) -> str:
        if self.appointment.patient_name:
            return self.appointment.patient_name
        if self.patient is not None:
            if self.patient.name:
                return self.patient.name
            if self.patient.username:
                return f"@{self.patient.username}"
        return "Ismi yo'q"

    @property
    def phone(self) -> str | None:
        return self.appointment.patient_phone or (self.patient.phone if self.patient else None)

    def line(self) -> str:
        local = self.appointment.scheduled_at.astimezone(CLINIC_TIMEZONE)
        phone = f" · {self.phone}" if self.phone else ""
        return f"🕐 {local:%H:%M} — {self.name}{phone}"


class BotAPI:
    """The four Bot API calls this needs, and nothing else."""

    def __init__(self, token: str, http: httpx.AsyncClient | None = None) -> None:
        self._base = f"{API}/bot{token}"
        self._http = http or httpx.AsyncClient(timeout=POLL_TIMEOUT_SECONDS + 10)

    async def _call(self, method: str, **payload: Any) -> Any:
        # The token is in the URL, so neither the URL nor the exception
        # (which quotes it) is logged -- only the method and Telegram's own
        # description.
        response = await self._http.post(f"{self._base}/{method}", json=payload)
        body = response.json()
        if not body.get("ok"):
            logger.warning(
                "doctor_telegram_call_failed",
                extra={"method": method, "description": body.get("description")},
            )
            return None
        return body.get("result")

    async def get_updates(self, offset: int | None) -> list[dict[str, Any]]:
        result = await self._call(
            "getUpdates",
            offset=offset,
            timeout=POLL_TIMEOUT_SECONDS,
            allowed_updates=["message", "callback_query"],
        )
        return result or []

    async def send(
        self, chat_id: int, text: str, reply_markup: dict[str, Any] | None = None
    ) -> None:
        payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        await self._call("sendMessage", **payload)

    async def answer_callback(self, callback_id: str) -> None:
        await self._call("answerCallbackQuery", callback_query_id=callback_id)

    async def delete_webhook(self) -> None:
        # getUpdates refuses to run while a webhook is set.
        await self._call("deleteWebhook")


def _allowed(username: str | None, settings: Settings) -> bool:
    wanted = {
        name.strip().lstrip("@").lower()
        for name in settings.doctor_telegram_usernames.split(",")
        if name.strip()
    }
    return bool(username) and username.lstrip("@").lower() in wanted


async def _tenant(session: AsyncSession, settings: Settings) -> Tenant | None:
    if not settings.provision_tenant_name:
        return None
    return (
        await session.execute(select(Tenant).where(Tenant.name == settings.provision_tenant_name))
    ).scalar_one_or_none()


def _day_bounds(day: date) -> tuple[datetime, datetime]:
    start = datetime.combine(day, time.min, tzinfo=CLINIC_TIMEZONE)
    return start, start + timedelta(days=1)


async def bookings_on(session: AsyncSession, tenant_id: uuid.UUID, day: date) -> list[Booking]:
    start, end = _day_bounds(day)
    rows = (
        await session.execute(
            select(Appointment, User)
            .outerjoin(User, Appointment.user_id == User.id)
            .where(
                Appointment.tenant_id == tenant_id,
                Appointment.status.in_(ACTIVE_STATUSES),
                Appointment.scheduled_at >= start,
                Appointment.scheduled_at < end,
            )
            .order_by(Appointment.scheduled_at)
        )
    ).all()
    return [Booking(appointment, patient) for appointment, patient in rows]


def _day_word(day: date, today: date) -> str:
    if day == today:
        return "Bugun"
    if day == today + timedelta(days=1):
        return "Ertaga"
    return f"{day:%d.%m.%Y}"


def _schedule_text(day: date, today: date, bookings: list[Booking]) -> str:
    head = f"{_day_word(day, today)} ({day:%d.%m.%Y})"
    if not bookings:
        return f"{head}: qabulga yozilganlar yo'q."
    lines = "\n".join(booking.line() for booking in bookings)
    return f"{head} — {len(bookings)} ta qabul:\n\n{lines}"


def _patient_notice(settings: Settings, booking: Booking) -> str:
    local = booking.appointment.scheduled_at.astimezone(CLINIC_TIMEZONE)
    doctor = settings.doctor_name or "Doktor"
    phone = settings.clinic_phone_numbers
    rebook = (
        f" Boshqa vaqtga yozilish uchun {phone} raqamiga qo'ng'iroq qiling."
        if phone
        else " Boshqa vaqtga yozilish uchun biz bilan bog'laning."
    )
    return (
        f"Assalomu alaykum! Afsuski, doktor {doctor} {local:%d.%m.%Y} kuni qabul qila "
        f"olmaydi, shuning uchun soat {local:%H:%M} dagi qabulingiz bekor qilindi."
        f"{rebook} Noqulaylik uchun uzr so'raymiz."
    )


async def cancel_day(
    session: AsyncSession, tenant_id: uuid.UUID, day: date, settings: Settings
) -> tuple[list[Booking], list[Booking]]:
    """Cancel the day and tell each patient. Returns (told, not told).

    Each appointment is cancelled whether or not its patient could be told:
    the doctor is not coming either way, and a booking left active would
    keep the slot looking taken in the dashboard. The ones not told are the
    list somebody has to ring.
    """
    told: list[Booking] = []
    untold: list[Booking] = []
    for booking in await bookings_on(session, tenant_id, day):
        appointment, patient = booking.appointment, booking.patient
        delivered = None
        if patient is not None:
            text = _patient_notice(settings, booking)
            conversation_id = appointment.conversation_id
            last_seen = await last_inbound_at(session, conversation_id) if conversation_id else None
            try:
                delivered = await send_reply(
                    session,
                    channel_id=patient.channel_id,
                    recipient_external_id=patient.external_id,
                    text=text,
                    last_user_message_at=last_seen or appointment.created_at,
                    reply_context=(
                        await reply_context_for(session, conversation_id)
                        if conversation_id
                        else None
                    ),
                )
            except Exception:  # noqa: BLE001 - one patient must not stop the rest
                logger.warning(
                    "doctor_day_off_notice_failed",
                    extra={"appointment_id": str(appointment.id)},
                    exc_info=True,
                )
            if delivered is not None and conversation_id is not None:
                await record_outbound_message(
                    session,
                    conversation_id=conversation_id,
                    channel_type=delivered,
                    text=text,
                    sender=MessageSender.BOT,
                )

        appointment.status = AppointmentStatus.CANCELLED
        note = "Doktor bu kuni qabul qila olmadi."
        appointment.notes = f"{appointment.notes}\n{note}" if appointment.notes else note
        await session.commit()
        (told if delivered is not None else untold).append(booking)
    return told, untold


def _day_off_report(day: date, today: date, told: list[Booking], untold: list[Booking]) -> str:
    word = _day_word(day, today).lower()
    if not told and not untold:
        return (
            f"{_day_word(day, today)} qabulga yozilganlar yo'q edi, hech kimga xabar yuborilmadi."
        )
    parts = [f"✅ {word} kungi {len(told) + len(untold)} ta qabul bekor qilindi."]
    if told:
        parts.append(
            f"\nInstagram orqali xabar yuborildi ({len(told)}):\n"
            + "\n".join(booking.line() for booking in told)
        )
    if untold:
        parts.append(
            f"\n⚠️ Xabar yetkazib bo'lmadi — ularga qo'ng'iroq qilish kerak ({len(untold)}):\n"
            + "\n".join(booking.line() for booking in untold)
        )
    return "\n".join(parts)


async def _remember_chat(session: AsyncSession, tenant: Tenant, chat_id: int) -> None:
    chats = [c for c in tenant.settings.get(CHATS_KEY, []) if isinstance(c, int)]
    if chat_id not in chats:
        tenant.settings = {**tenant.settings, CHATS_KEY: [*chats, chat_id]}
        await session.commit()


async def _ask_day_off(api: BotAPI, chat_id: int, day: date, today: date, count: int) -> None:
    word = _day_word(day, today).lower()
    if count == 0:
        await api.send(chat_id, f"{_day_word(day, today)} qabulga yozilganlar yo'q.", KEYBOARD)
        return
    await api.send(
        chat_id,
        f"{word.capitalize()} ({day:%d.%m.%Y}) {count} ta bemor yozilgan. Hammasining qabulini "
        f"bekor qilib, ularga Instagram orqali doktor kela olmasligini yozaymi?",
        {
            "inline_keyboard": [
                [
                    {"text": "✅ Ha, xabar yubor", "callback_data": f"off:{day.isoformat()}"},
                    {"text": "❌ Yo'q", "callback_data": "cancel"},
                ]
            ]
        },
    )


async def handle_update(
    api: BotAPI, session: AsyncSession, tenant: Tenant, update: dict[str, Any]
) -> None:
    settings = get_settings()
    today = datetime.now(CLINIC_TIMEZONE).date()

    callback = update.get("callback_query")
    if callback:
        await api.answer_callback(callback["id"])
        sender = callback.get("from", {})
        chat_id = callback.get("message", {}).get("chat", {}).get("id")
        if chat_id is None or not _allowed(sender.get("username"), settings):
            return
        data = callback.get("data", "")
        if data.startswith("off:"):
            day = date.fromisoformat(data.removeprefix("off:"))
            if day < today:
                await api.send(chat_id, "Bu kun o'tib ketgan.", KEYBOARD)
                return
            await api.send(chat_id, "Bekor qilinyapti, bemorlarga xabar yuborilyapti…")
            token = set_current_tenant(tenant.id)
            try:
                told, untold = await cancel_day(session, tenant.id, day, settings)
            finally:
                reset_current_tenant(token)
            logger.info(
                "doctor_day_off_applied",
                extra={"day": day.isoformat(), "told": len(told), "untold": len(untold)},
            )
            await api.send(chat_id, _day_off_report(day, today, told, untold), KEYBOARD)
        elif data == "cancel":
            await api.send(chat_id, "Yaxshi, hech narsa o'zgarmadi.", KEYBOARD)
        return

    message = update.get("message")
    if not message:
        return
    chat_id = message["chat"]["id"]
    username = message.get("from", {}).get("username")
    if not _allowed(username, settings):
        logger.warning("doctor_telegram_refused", extra={"username": username, "chat_id": chat_id})
        await api.send(chat_id, "Kechirasiz, bu bot faqat doktor uchun.")
        return

    await _remember_chat(session, tenant, chat_id)
    text = (message.get("text") or "").strip()

    if text in {BTN_TODAY, BTN_TOMORROW, "/bugun", "/ertaga"}:
        day = today + timedelta(days=1) if text in {BTN_TOMORROW, "/ertaga"} else today
        token = set_current_tenant(tenant.id)
        try:
            bookings = await bookings_on(session, tenant.id, day)
        finally:
            reset_current_tenant(token)
        await api.send(chat_id, _schedule_text(day, today, bookings), KEYBOARD)
        return

    if text in {BTN_OFF_TODAY, BTN_OFF_TOMORROW} or _ABSENT.search(text):
        tomorrow = text == BTN_OFF_TOMORROW or (text != BTN_OFF_TODAY and _TOMORROW.search(text))
        day = today + timedelta(days=1) if tomorrow else today
        token = set_current_tenant(tenant.id)
        try:
            count = len(await bookings_on(session, tenant.id, day))
        finally:
            reset_current_tenant(token)
        await _ask_day_off(api, chat_id, day, today, count)
        return

    await api.send(
        chat_id,
        "Assalomu alaykum, doktor! Yangi qabullar shu yerga keladi.\n\n"
        "Pastdagi tugmalar bilan qabullarni ko'rishingiz mumkin. Ishga chiqa olmasangiz, "
        "masalan «bugun ishga bora olmayman» deb yozing — bemorlarga o'zim xabar beraman.",
        KEYBOARD,
    )


async def announce_new_bookings(api: BotAPI, session: AsyncSession, tenant: Tenant) -> None:
    """Tell the doctor about every appointment created since the last look."""
    chats = [c for c in tenant.settings.get(CHATS_KEY, []) if isinstance(c, int)]
    raw = tenant.settings.get(NOTIFIED_UNTIL_KEY)
    now = datetime.now(UTC)
    if not raw:
        # First run: start from now rather than announcing the whole history.
        tenant.settings = {**tenant.settings, NOTIFIED_UNTIL_KEY: now.isoformat()}
        await session.commit()
        return
    since = datetime.fromisoformat(raw)
    rows = (
        await session.execute(
            select(Appointment, User)
            .outerjoin(User, Appointment.user_id == User.id)
            .where(
                Appointment.tenant_id == tenant.id,
                Appointment.created_at > since,
                Appointment.status.in_(ACTIVE_STATUSES),
            )
            .order_by(Appointment.created_at)
        )
    ).all()
    if not rows:
        return
    if chats:
        for appointment, patient in rows:
            booking = Booking(appointment, patient)
            local = appointment.scheduled_at.astimezone(CLINIC_TIMEZONE)
            text = f"🆕 Yangi qabul\n📅 {local:%d.%m.%Y} 🕐 {local:%H:%M}\n👤 {booking.name}"
            if booking.phone:
                text += f"\n📞 {booking.phone}"
            if appointment.notes:
                text += f"\n📝 {appointment.notes}"
            for chat_id in chats:
                await api.send(chat_id, text)
    latest = max(appointment.created_at for appointment, _ in rows)
    tenant.settings = {**tenant.settings, NOTIFIED_UNTIL_KEY: latest.isoformat()}
    await session.commit()


async def run_forever(stop: asyncio.Event) -> None:
    """The worker's background loop: poll Telegram, announce new bookings."""
    settings = get_settings()
    token = settings.doctor_telegram_bot_token
    if not token:
        return
    api = BotAPI(token)
    await api.delete_webhook()
    logger.info("doctor_telegram_started")
    while not stop.is_set():
        try:
            async with db_session() as session:
                tenant = await _tenant(session, settings)
                if tenant is None:
                    logger.warning("doctor_telegram_no_tenant")
                    await asyncio.sleep(30)
                    continue
                offset = tenant.settings.get(OFFSET_KEY)
            updates = await api.get_updates(offset)
            async with db_session() as session:
                tenant = await _tenant(session, settings)
                assert tenant is not None
                for update in updates:
                    tenant.settings = {**tenant.settings, OFFSET_KEY: update["update_id"] + 1}
                    await session.commit()
                    try:
                        await handle_update(api, session, tenant, update)
                    except Exception:  # noqa: BLE001 - one bad update must not stop the bot
                        await session.rollback()
                        logger.exception("doctor_telegram_update_failed")
                    await session.refresh(tenant)
                token_ctx = set_current_tenant(tenant.id)
                try:
                    await announce_new_bookings(api, session, tenant)
                finally:
                    reset_current_tenant(token_ctx)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - keep polling through a bad minute
            logger.exception("doctor_telegram_loop_failed")
            await asyncio.sleep(10)
