"""Mirroring leads into the clinic's own Google Sheet.

The people who own these clinics do not open dashboards. They open the
spreadsheet they already keep, and a lead that is only in a dashboard is,
to them, a lead nobody told them about. So every patient the bots talk to
lands in a sheet with four columns the owner asked for:

    ism familiya | telefon | manba | bemor haqida qisqacha

A mirror, never a source. Nothing is read back into the application, and a
deleted, renamed or broken sheet loses nothing the database does not still
hold -- which is what makes it safe for every failure here to be swallowed
rather than retried into the patient's face.

One tab for both channels, with the source column saying which. Two tabs
was the other reading of the request, and it makes "how many leads this
week" a sum the owner has to do by hand.

Google's REST API directly, over the httpx client this project already
uses, with google-auth for the token. gspread and google-api-python-client
each bring a stack of transitive dependencies to wrap two endpoints
(values.append and values.update), and one of those endpoints is the whole
integration.
"""

import base64
import binascii
import json
import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any, cast

import google.auth.transport.requests as google_requests
import httpx
from google.oauth2 import service_account

from app.core.config import Settings, get_settings
from app.services.appointment import CLINIC_TIMEZONE
from app.services.conversation_signals import looks_like_a_phone_number

logger = logging.getLogger(__name__)

SHEETS_API = "https://sheets.googleapis.com/v4/spreadsheets"
SCOPES = ("https://www.googleapis.com/auth/spreadsheets",)

# One sheet, and the date column is a real date rather than text.
#
# That is the whole mechanism: Sheets' own filter offers "Kecha / Bugun /
# Erta" and a list of days on a date column, and nothing else it offers is
# as good -- a hand-built dropdown cannot do "yesterday", and a tab per day
# cannot do "this week". A text column that merely looks like a date gets
# none of it.
VIEW_SHEET = "Lidlar"

HEADER = ("Sana", "Ism familiya", "Telefon", "Manba", "Bemor haqida qisqacha")

_LAST_COLUMN = "E"
_DATE_COLUMN = "A"
_PHONE_COLUMN = "C"

# How the date is shown. The value underneath is a real date whatever this
# says; this only decides what the clinic reads.
_DATE_FORMAT = "dd.MM.yyyy"

# The appointment book, on its own sheet.
#
# Separate from the lead list because they answer different questions. A
# lead is "somebody wrote, ring them"; an appointment is "this person is
# expected at this time" -- it has a doctor, a slot and a status that moves.
APPOINTMENT_SHEET = "Qabullar"

APPOINTMENT_HEADER = (
    "Qabul_ID",
    "Yozilgan_Vaqti",
    "Bemor_F_I_Sh",
    "Telefon_Raqami",
    "Qabul_Sanasi",
    "Qabul_Vaqti",
    "Shifokor",
    "Mutaxassislik",
    "Xizmat_Turi",
    "Xizmat_Narxi",
    "Kanal",
    "Mijoz_ID",
    "Status",
    "Bekor_Qilish_Sababi",
    "Izoh_Eslatma",
)

# H (Mutaxassislik) and J (Xizmat_Narxi) are never written.
#
# They hold one ARRAYFORMULA each, in row 2, spilling down every row --
# which is the only shape that reaches a row the bot has not created yet.
# Writing a value into a spilled cell does not overwrite it, it breaks it:
# the whole column turns into #REF! for every appointment at once. So the
# bot writes around them, in three ranges rather than one.
_APPOINTMENT_HEAD = "A:G"
_APPOINTMENT_SERVICE = "I"
_APPOINTMENT_TAIL = "K:O"

# The clinic reads Uzbek, and these are the words in its own dropdowns --
# the Sozlamalar sheet is the single source of both, so a status written
# here always matches one the sheet offers.
STATUS_LABELS = {
    "scheduled": "Kutilmoqda",
    "confirmed": "Tasdiqlandi",
    "cancelled": "Bekor qilindi",
    "completed": "Yakunlandi",
    "no_show": "Kelmadi",
}

CHANNEL_LABELS = {
    "telegram": "Telegram",
    "instagram": "Instagram",
    "web": "Web",
    "operator": "Qo'ng'iroq",
    "phone": "Qo'ng'iroq",
}

# How far down the number formats are applied.
#
# An open-ended range only reaches the rows that exist when it is sent,
# and appending makes a new row that inherits nothing -- so the date came
# out as 46265 and the phone as a number, on a sheet whose columns were
# correctly formatted. Formatting rows that do not exist yet is what makes
# the format meet the row when it arrives. A thousand leads is years for
# one clinic.
_FORMATTED_ROWS = 1000

# Enough for a sentence about a toothache, short enough that the column
# stays readable in a spreadsheet without anyone widening it.
MAX_COMMENT = 180

# Bounded so a slow Google never becomes a slow reply. This runs after the
# patient has already been answered, but it still holds a worker slot.
TIMEOUT_SECONDS = 20.0


# The look of a freshly created tab. Colours chosen for a document somebody
# reads on a phone between patients: one dark header that survives scrolling,
# quiet banding so the eye keeps its place across four columns, and the
# story column wide enough to hold a sentence.
_HEADER_COLOUR = "#0F766E"
_BAND_COLOUR = "#F1F5F9"
_COLUMN_WIDTHS = (110, 200, 160, 120, 520)


def _rgb(value: str) -> dict[str, float]:
    raw = value.lstrip("#")
    return {
        "red": int(raw[0:2], 16) / 255,
        "green": int(raw[2:4], 16) / 255,
        "blue": int(raw[4:6], 16) / 255,
    }


def _layout_requests(sheet_id: int) -> list[dict[str, Any]]:
    """The parts the sheet does not work without.

    Sent separately from the decoration below, because batchUpdate is
    atomic: one refused request takes the whole batch with it. A sheet that
    already had banding on it -- which is any sheet this has ever run on
    before -- made addBanding fail, and with it went the date format and the
    phone format, so the column showed 46262 and the phone came back as
    9.989E+11. The rules and the paint now travel separately.
    """
    white = {"red": 1.0, "green": 1.0, "blue": 1.0}
    ink = _rgb("#1F2937")
    full = {"sheetId": sheet_id, "startColumnIndex": 0, "endColumnIndex": len(HEADER)}
    requests: list[dict[str, Any]] = [
        # The header stays in view through a year of leads.
        {
            "updateSheetProperties": {
                "properties": {"sheetId": sheet_id, "gridProperties": {"frozenRowCount": 1}},
                "fields": "gridProperties.frozenRowCount",
            }
        },
        {
            "repeatCell": {
                "range": {**full, "startRowIndex": 0, "endRowIndex": 1},
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": _rgb(_HEADER_COLOUR),
                        "textFormat": {"foregroundColor": white, "bold": True, "fontSize": 11},
                        "verticalAlignment": "MIDDLE",
                        "horizontalAlignment": "LEFT",
                        "padding": {"left": 10, "right": 10, "top": 6, "bottom": 6},
                    }
                },
                "fields": (
                    "userEnteredFormat(backgroundColor,textFormat,verticalAlignment,"
                    "horizontalAlignment,padding)"
                ),
            }
        },
        {
            "updateDimensionProperties": {
                "range": {"sheetId": sheet_id, "dimension": "ROWS", "startIndex": 0, "endIndex": 1},
                "properties": {"pixelSize": 40},
                "fields": "pixelSize",
            }
        },
        {
            "repeatCell": {
                "range": {**full, "startRowIndex": 1},
                "cell": {
                    "userEnteredFormat": {
                        # Without this the story runs on under the next
                        # column and the owner reads half a sentence.
                        "wrapStrategy": "WRAP",
                        "verticalAlignment": "MIDDLE",
                        "textFormat": {"foregroundColor": ink, "fontSize": 10},
                        "padding": {"left": 10, "right": 10, "top": 6, "bottom": 6},
                    }
                },
                "fields": "userEnteredFormat(wrapStrategy,verticalAlignment,textFormat,padding)",
            }
        },
        {
            "repeatCell": {
                # A phone number is a label, not a quantity. Left as a number
                # it loses a leading zero and renders as 9.989E+11.
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": _FORMATTED_ROWS,
                    "startColumnIndex": HEADER.index("Telefon"),
                    "endColumnIndex": HEADER.index("Telefon") + 1,
                },
                "cell": {"userEnteredFormat": {"numberFormat": {"type": "TEXT"}}},
                "fields": "userEnteredFormat.numberFormat",
            }
        },
        {
            # The filter itself. This is what puts the dropdown on every
            # column header, and on a real date column Sheets fills it with
            # "Kecha / Bugun / Erta" and the days on record -- which is the
            # whole feature, not decoration.
            "setBasicFilter": {"filter": {"range": {**full, "startRowIndex": 0}}}
        },
        {
            # The date column, shown the way the clinic writes dates. The
            # value underneath stays a real date, which is what lets Sheets'
            # own filter offer "Kecha / Bugun / Erta" on this column.
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 1,
                    "endRowIndex": _FORMATTED_ROWS,
                    "startColumnIndex": HEADER.index("Sana"),
                    "endColumnIndex": HEADER.index("Sana") + 1,
                },
                "cell": {
                    "userEnteredFormat": {"numberFormat": {"type": "DATE", "pattern": _DATE_FORMAT}}
                },
                "fields": "userEnteredFormat.numberFormat",
            }
        },
    ]
    return requests


def _decoration_requests(sheet_id: int) -> list[dict[str, Any]]:
    """Banding and column widths: nice to have, and allowed to fail."""
    white = {"red": 1.0, "green": 1.0, "blue": 1.0}
    full = {"sheetId": sheet_id, "startColumnIndex": 0, "endColumnIndex": len(HEADER)}
    requests: list[dict[str, Any]] = [
        {
            "addBanding": {
                "bandedRange": {
                    "range": {**full, "startRowIndex": 1},
                    "rowProperties": {
                        "firstBandColor": white,
                        "secondBandColor": _rgb(_BAND_COLOUR),
                    },
                }
            }
        },
    ]
    requests.extend(
        {
            "updateDimensionProperties": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": index,
                    "endIndex": index + 1,
                },
                "properties": {"pixelSize": width},
                "fields": "pixelSize",
            }
        }
        for index, width in enumerate(_COLUMN_WIDTHS)
    )
    return requests


class SheetsError(Exception):
    """Raised when the sheet cannot be written. Never reaches a patient."""


@dataclass(frozen=True)
class LeadRow:
    """One patient, as the clinic's owner reads them."""

    name: str | None
    phone: str | None
    source: str
    comment: str | None
    # The day this lead belongs on. The appointment's day when there is one,
    # because a patient who writes on the 28th to be seen on the 31st belongs
    # in the 31st's list -- that is the list somebody works from in the
    # morning. Only when nothing is booked does it fall back to today.
    day: date

    def as_cells(self) -> list[str]:
        comment = (self.comment or "").strip().replace(chr(10), " ")
        if len(comment) > MAX_COMMENT:
            comment = comment[: MAX_COMMENT - 1].rstrip() + "…"
        # ISO in, whatever _DATE_FORMAT says out: Sheets parses this shape
        # regardless of the spreadsheet's locale, which "28.08.2026" is not
        # guaranteed to be.
        return [f"{self.day:%Y-%m-%d}", self.name or "", self.phone or "", self.source, comment]


def _load_credentials(raw: str) -> service_account.Credentials:
    """The service-account key, given verbatim or base64-encoded.

    Both, because the key's private_key field contains newlines and whether
    a host can carry those in an environment variable varies -- Railway can,
    a .env file read line by line cannot. Guessing between the two costs one
    try/except and removes a class of "it works locally" bug.
    """
    text = raw.strip()
    if not text.startswith("{"):
        try:
            text = base64.b64decode(text, validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise SheetsError("GOOGLE_SERVICE_ACCOUNT_JSON is neither JSON nor base64") from exc
    try:
        info = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SheetsError(f"GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON: {exc}") from exc
    try:
        # google-auth ships without complete annotations, so this reads as
        # an untyped call under --strict. Silenced at the call rather than
        # for the module, so the rest of this file stays fully checked.
        credentials = service_account.Credentials.from_service_account_info(  # type: ignore[no-untyped-call]
            info, scopes=list(SCOPES)
        )
        return cast("service_account.Credentials", credentials)
    except Exception as exc:  # noqa: BLE001 - google's message names the missing field
        raise SheetsError(f"Not a usable service account key: {exc}") from exc


@dataclass(frozen=True)
class AppointmentRow:
    """One booking, as the appointment book records it."""

    appointment_id: uuid.UUID
    created_at: datetime
    scheduled_at: datetime
    patient_name: str | None
    phone: str | None
    doctor: str
    channel: str
    client_id: str | None
    status: str
    service: str | None = None
    cancel_reason: str | None = None
    note: str | None = None

    @property
    def reference(self) -> str:
        """The clinic-facing code, derived rather than counted.

        A running number -- MED-1001, MED-1002 -- would have to be read off
        the sheet and incremented, which two bookings a second apart can do
        at the same time and land on the same code. This one falls out of
        the appointment's own id, so the same booking always produces the
        same reference: a retry updates its row instead of adding a second.
        """
        return f"MED-{self.appointment_id.hex[:6].upper()}"

    def head_cells(self) -> list[str]:
        """A-G. Written as text Sheets parses: the date and time columns are
        formatted as a real date and a real time, so "2026-09-02" and "14:30"
        become values the filter and the Dashboard can count."""
        local = self.scheduled_at.astimezone(CLINIC_TIMEZONE)
        return [
            self.reference,
            f"{self.created_at.astimezone(CLINIC_TIMEZONE):%Y-%m-%d %H:%M}",
            self.patient_name or "",
            # Leading apostrophe: Sheets reads a cell that starts with "+"
            # as a formula, exactly as it reads "=", so USER_ENTERED turned
            # +998771997272 into the number 998771997272 and the clinic lost
            # the country code. The apostrophe is the sheet's own way of
            # saying "this is text"; it is consumed on the way in and never
            # appears in the cell.
            f"'{_e164(self.phone)}" if self.phone else "",
            f"{local:%Y-%m-%d}",
            f"{local:%H:%M}",
            self.doctor,
        ]

    def service_cell(self) -> list[str]:
        """I. Empty for a booking the assistant took: it agrees a time, not a
        procedure, and guessing a billable service from a complaint is how a
        patient is charged for something nobody offered them."""
        return [self.service or ""]

    def tail_cells(self) -> list[str]:
        comment = (self.note or "").strip().replace(chr(10), " ")
        if len(comment) > MAX_COMMENT:
            comment = comment[: MAX_COMMENT - 1].rstrip() + "\u2026"
        return [
            CHANNEL_LABELS.get(self.channel.lower(), "Web"),
            str(self.client_id or ""),
            STATUS_LABELS.get(self.status, self.status),
            self.cancel_reason or "",
            comment,
        ]


def _e164(phone: str | None) -> str:
    """+998XXXXXXXXX, which is the only shape the column is documented to hold.

    Stored as text, so the leading + survives -- a number-formatted cell
    drops it, and a number nobody can dial is worse than an empty one.
    """
    digits = "".join(character for character in (phone or "") if character.isdigit())
    if not digits:
        return ""
    if len(digits) == 9:
        digits = f"998{digits}"
    return f"+{digits}"


class SheetsMirror:
    """Appends leads to one worksheet of one spreadsheet."""

    def __init__(self, settings: Settings | None = None) -> None:
        resolved = settings or get_settings()
        if resolved.google_service_account_json is None or (
            resolved.google_sheets_spreadsheet_id is None
        ):
            raise SheetsError(
                "GOOGLE_SERVICE_ACCOUNT_JSON and GOOGLE_SHEETS_SPREADSHEET_ID are both required"
            )
        self._credentials = _load_credentials(resolved.google_service_account_json)
        self._spreadsheet_id = resolved.google_sheets_spreadsheet_id
        # Whether the sheet has been checked this process. A flag rather
        # than a request, because this runs before every lead.
        self._ready = False
        self._appointments_ready = False

    def _token(self) -> str:
        # google-auth refreshes only when the current token is close to
        # expiry, so this is a cheap call on all but the first use.
        if not self._credentials.valid:
            self._credentials.refresh(google_requests.Request())  # type: ignore[no-untyped-call]
        token = self._credentials.token
        if not isinstance(token, str):  # pragma: no cover - defensive
            raise SheetsError("Google returned no access token")
        return token

    async def _call(
        self, client: httpx.AsyncClient, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any]:
        response = await client.request(
            method,
            f"{SHEETS_API}/{self._spreadsheet_id}{path}",
            headers={"Authorization": f"Bearer {self._token()}"},
            **kwargs,
        )
        if response.is_error:
            # The body names the tab or the id that was wrong, which is the
            # whole diagnostic value; it carries no credential.
            raise SheetsError(f"Sheets API {response.status_code}: {response.text[:300]}")
        body: dict[str, Any] = response.json()
        return body

    async def _ensure_sheet(self, client: httpx.AsyncClient) -> None:
        """Create and lay out the sheet, once per process."""
        if self._ready:
            return
        sheets = await self.worksheet_ids(client)
        if VIEW_SHEET not in sheets:
            sheets[VIEW_SHEET] = await self._create_worksheet(client, VIEW_SHEET)
        existing = await self._call(
            client,
            "GET",
            f"/values/{VIEW_SHEET}!A1:{_LAST_COLUMN}1",
            params={"majorDimension": "ROWS"},
        )
        # Matching, not merely present.
        #
        # "Is there anything in row 1?" was the check, and it let a sheet
        # left over from an older layout through untouched: the header stayed
        # as it was, the styling was skipped, and a real patient's lead
        # landed two columns to the right of where anyone would look for it.
        # Any header that is not this one is a sheet from a different design,
        # and it is rewritten.
        header = [str(cell).strip() for cell in (existing.get("values") or [[]])[0]]
        if header != list(HEADER):
            if header:
                logger.warning("sheets_header_replaced found=%s", header)
            await self._call(
                client,
                "PUT",
                f"/values/{VIEW_SHEET}!A1:{_LAST_COLUMN}1",
                params={"valueInputOption": "RAW"},
                json={"values": [list(HEADER)]},
            )
            await self._style(client, sheets[VIEW_SHEET])
        self._ready = True

    async def worksheet_ids(self, client: httpx.AsyncClient) -> dict[str, int]:
        body = await self._call(
            client, "GET", "", params={"fields": "sheets.properties(sheetId,title)"}
        )
        return {
            sheet["properties"]["title"]: sheet["properties"]["sheetId"]
            for sheet in body.get("sheets", [])
            if "properties" in sheet
        }

    async def _create_worksheet(self, client: httpx.AsyncClient, title: str) -> int:
        body = await self._call(
            client,
            "POST",
            ":batchUpdate",
            json={
                "requests": [
                    {
                        "addSheet": {
                            "properties": {
                                "title": title,
                                "gridProperties": {"columnCount": len(HEADER)},
                            }
                        }
                    }
                ]
            },
        )
        sheet_id: int = body["replies"][0]["addSheet"]["properties"]["sheetId"]
        logger.warning("sheets_created title=%s", title)
        return sheet_id

    async def _style(self, client: httpx.AsyncClient, gid: int) -> None:
        """Lay the sheet out, once, when it is created.

        Two calls, not one: batchUpdate is atomic, so a refused decoration
        would otherwise take the date and phone formats down with it -- and
        those are what make the filter and the phone numbers work at all.

        Both are swallowed on failure. A sheet that works but looks plain is
        a far better outcome than a lead that was not written because
        styling it went wrong.
        """
        for name, requests in (
            ("layout", _layout_requests(gid)),
            ("decoration", _decoration_requests(gid)),
        ):
            try:
                await self._call(client, "POST", ":batchUpdate", json={"requests": requests})
            except SheetsError as exc:
                logger.warning("sheets_styling_skipped part=%s error=%s", name, exc)

    async def append(self, client: httpx.AsyncClient, row: LeadRow) -> None:
        await self._call(
            client,
            "POST",
            f"/values/{VIEW_SHEET}!A:{_LAST_COLUMN}:append",
            # USER_ENTERED, so the date column holds a real date and Sheets'
            # own filter can offer "Kecha / Bugun / Erta" on it. The phone
            # column is formatted as text (see _style_requests), which is
            # what stops the same setting turning a number into 9.989E+11.
            # No insertDataOption: the default fills the blank rows that are
            # already there, and those already carry the date and phone
            # formats. INSERT_ROWS makes a new row instead, which inherits
            # its format from the header above it -- so the date came out as
            # 46265 and the phone as a number on a correctly formatted sheet.
            params={"valueInputOption": "USER_ENTERED"},
            json={"values": [row.as_cells()]},
        )

    async def update_row(self, client: httpx.AsyncClient, row_number: int, row: LeadRow) -> None:
        """Rewrite one row in place, for a lead the sheet already has.

        A name or a phone usually arrives a few turns after the first
        message. Appending again would give the owner the same person twice,
        once without their number.
        """
        await self._call(
            client,
            "PUT",
            f"/values/{VIEW_SHEET}!A{row_number}:{_LAST_COLUMN}{row_number}",
            params={"valueInputOption": "USER_ENTERED"},
            json={"values": [row.as_cells()]},
        )

    async def find_row(self, client: httpx.AsyncClient, phone: str, day: date) -> int | None:
        """The row holding this phone on this day, if any.

        Matched on both. The same patient booking a second visit is a second
        line, on the day of that visit -- which is what the list is for.
        """
        if not phone:
            return None
        body = await self._call(
            client,
            "GET",
            f"/values/{VIEW_SHEET}!{_DATE_COLUMN}:{_PHONE_COLUMN}",
            params={"majorDimension": "ROWS", "valueRenderOption": "UNFORMATTED_VALUE"},
        )
        wanted = _sheets_serial(day)
        for index, values in enumerate(body.get("values") or [], start=1):
            cells = [*values, "", ""]
            if cells[0] == wanted and str(cells[2]).strip() == phone:
                return index
        return None

    async def _ensure_appointment_sheet(self, client: httpx.AsyncClient) -> None:
        if self._appointments_ready:
            return
        sheets = await self.worksheet_ids(client)
        if APPOINTMENT_SHEET not in sheets:
            sheets[APPOINTMENT_SHEET] = await self._create_worksheet(client, APPOINTMENT_SHEET)
        existing = await self._call(
            client,
            "GET",
            f"/values/{APPOINTMENT_SHEET}!A1:O1",
            params={"majorDimension": "ROWS"},
        )
        header = [str(cell).strip() for cell in (existing.get("values") or [[]])[0]]
        if header != list(APPOINTMENT_HEADER):
            if header:
                logger.warning("sheets_appointment_header_replaced found=%s", header)
            await self._call(
                client,
                "PUT",
                f"/values/{APPOINTMENT_SHEET}!A1:O1",
                params={"valueInputOption": "RAW"},
                json={"values": [list(APPOINTMENT_HEADER)]},
            )
        self._appointments_ready = True

    async def find_appointment_row(self, client: httpx.AsyncClient, reference: str) -> int | None:
        """The row holding this reference, so a status change rewrites the
        booking the clinic already has rather than adding a second one."""
        body = await self._call(
            client,
            "GET",
            f"/values/{APPOINTMENT_SHEET}!A:A",
            params={"majorDimension": "ROWS"},
        )
        for index, values in enumerate(body.get("values") or [], start=1):
            if values and str(values[0]).strip() == reference:
                return index
        return None

    async def _write_appointment(
        self, client: httpx.AsyncClient, row_number: int, row: AppointmentRow
    ) -> None:
        """The three ranges the bot owns, in one request.

        Three rather than one because H and J hold spilled formulas; one
        request rather than three because a booking half-written is a
        booking the front desk cannot act on.
        """
        await self._call(
            client,
            "POST",
            "/values:batchUpdate",
            json={
                "valueInputOption": "USER_ENTERED",
                "data": [
                    {
                        "range": f"{APPOINTMENT_SHEET}!A{row_number}:G{row_number}",
                        "values": [row.head_cells()],
                    },
                    {
                        "range": f"{APPOINTMENT_SHEET}!I{row_number}",
                        "values": [row.service_cell()],
                    },
                    {
                        "range": f"{APPOINTMENT_SHEET}!K{row_number}:O{row_number}",
                        "values": [row.tail_cells()],
                    },
                ],
            },
        )

    async def upsert_appointment(self, row: AppointmentRow) -> None:
        """Put this booking in the appointment book, once.

        The row is allocated by appending the head columns and reading back
        which row Google used, rather than by counting rows ourselves: two
        bookings taken in the same second would otherwise pick the same one
        and the second would overwrite the first.
        """
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            await self._ensure_appointment_sheet(client)
            existing = await self.find_appointment_row(client, row.reference)
            if existing is None:
                appended = await self._call(
                    client,
                    "POST",
                    f"/values/{APPOINTMENT_SHEET}!{_APPOINTMENT_HEAD}:append",
                    params={"valueInputOption": "USER_ENTERED"},
                    json={"values": [row.head_cells()]},
                )
                updated = appended.get("updates", {}).get("updatedRange", "")
                existing = _row_number(updated)
                if existing is None:  # pragma: no cover - Google always says
                    logger.error("sheets_appointment_row_unknown range=%s", updated)
                    return
            await self._write_appointment(client, existing, row)

    async def upsert(self, row: LeadRow) -> None:
        """Put this lead on the list, once."""
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            await self._ensure_sheet(client)
            existing = await self.find_row(client, row.phone or "", row.day)
            if existing is not None:
                await self.update_row(client, existing, row)
                return
            await self.append(client, row)


# Sheets counts days from 1899-12-30, and an unformatted read gives that
# number rather than the text on screen. Matching on it avoids having to
# guess how the spreadsheet's locale renders a date back to us.
_SHEETS_EPOCH = date(1899, 12, 30)


def _row_number(updated_range: str) -> int | None:
    """The row out of "Qabullar!A5:G5"."""
    match = re.search(r"![A-Z]+(\d+)", updated_range)
    return int(match.group(1)) if match else None


def _sheets_serial(day: date) -> int:
    return (day - _SHEETS_EPOCH).days


@lru_cache
def get_mirror() -> SheetsMirror | None:
    """The configured mirror, or None when the clinic has no sheet.

    Cached because building it parses the key and sets up a credentials
    object; the token inside refreshes itself.
    """
    settings = get_settings()
    if settings.google_sheets_spreadsheet_id is None:
        return None
    try:
        return SheetsMirror(settings)
    except SheetsError:
        logger.exception("sheets_disabled_bad_configuration")
        return None


async def mirror_lead(row: LeadRow) -> None:
    """Send one lead to the clinic's sheet, if there is one.

    Never raises. The patient has already been answered by the time this
    runs, and the row is already in the database — a spreadsheet that is
    briefly behind is not worth failing a job over, and a retry would
    duplicate the reply, not the row.
    """
    mirror = get_mirror()
    if mirror is None:
        return
    try:
        await mirror.upsert(row)
    except SheetsError as exc:
        logger.error("sheets_mirror_failed error=%s", exc)
    except Exception:
        logger.exception("sheets_mirror_failed")


async def mirror_appointment(row: AppointmentRow) -> None:
    """Send one booking to the clinic's appointment book, if there is one.

    Never raises, for the same reason mirror_lead does not: the patient has
    been told their time and the row is already in the database. A
    spreadsheet that is briefly behind is not worth failing a job over.
    """
    mirror = get_mirror()
    if mirror is None:
        return
    try:
        await mirror.upsert_appointment(row)
    except SheetsError as exc:
        logger.error("sheets_appointment_failed error=%s", exc)
    except Exception:
        logger.exception("sheets_appointment_failed")


# Openings, acknowledgements and bare numbers — the messages that say
# nothing about why the patient wrote. Matched as whole words against a
# stripped message, so "salom" is skipped and "salom, tishim og'riyapti" is
# not.
_SMALL_TALK = frozenset(
    {
        "salom",
        "assalom",
        "assalomu",
        "alaykum",
        "assalomu alaykum",
        "assalom alaykum",
        "ассалому алайкум",
        "салом",
        "здравствуйте",
        "привет",
        "hi",
        "hello",
        "ok",
        "ha",
        "yoq",
        "rahmat",
        "спасибо",
        "/start",
    }
)


def summarise_problem(patient_messages: Sequence[str]) -> str:
    """One line saying what this patient wrote in about.

    Their own words, not a generated summary. A summary would mean another
    model call per lead, and this deployment's daily allowance is better
    spent answering patients than describing them — and the patient's own
    "tishim og'riyapti, chap tomonda" is more useful to a clinic than
    anything a paraphrase would produce.

    The first message that is not a greeting or a phone number, because that
    is the one that says why they wrote.

    It was the longest such message until a real conversation showed why
    that fails: "ha, shu vaqt to'g'ri keladi" is longer than "implant
    qancha turadi?", so the clinic's column filled with the patient
    agreeing to a time instead of the reason they came. Confirmations and
    logistics are reliably wordier than the question underneath them.

    The first one cannot have that problem. Nothing has been offered yet
    when it arrives, so it cannot be agreement to anything -- it is the
    reason, every time. It sometimes costs a detail the patient added a
    message later ("chap tomonda, sovuqdan achishadi"), and that is the
    cheaper mistake: a column saying "implant qancha turadi?" is useful to
    whoever rings them, and one saying "ha, shu vaqt to'g'ri keladi" is not.
    """
    candidates = [
        text.strip()
        for text in patient_messages
        if text.strip()
        and text.strip().lower().strip("!?.,") not in _SMALL_TALK
        and not looks_like_a_phone_number(text)
    ]
    return candidates[0] if candidates else ""
