"""Mirroring leads into the clinic's own spreadsheet.

No test here talks to Google. The transport is an httpx.MockTransport, so
what is checked is what this code sends and how it behaves when Google says
no -- which is the part that decides whether a patient's reply survives a
broken sheet.
"""

import base64
import json
import uuid
from datetime import UTC, date, datetime
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from app.services.sheets import (
    APPOINTMENT_HEADER,
    HEADER,
    MAX_COMMENT,
    VIEW_SHEET,
    AppointmentRow,
    LeadRow,
    SheetsError,
    SheetsMirror,
    _decoration_requests,
    _layout_requests,
    _load_credentials,
    _sheets_serial,
    mirror_lead,
    summarise_problem,
)
from tests.conftest import isolated_settings

# A structurally valid key with a real (throwaway) RSA private key would be
# needed to build Credentials, so the credential path is tested through
# _load_credentials' failure modes instead, and the request-shaping tests
# drive SheetsMirror with its credentials replaced.
NOT_A_KEY = json.dumps({"type": "service_account", "project_id": "p"})


class _FakeCredentials:
    valid = True
    token = "test-token"

    def refresh(self, request: Any) -> None:  # pragma: no cover - never called
        raise AssertionError("a valid credential should not be refreshed")


def _mirror(handler: Any) -> tuple[SheetsMirror, list[httpx.Request]]:
    """A mirror whose credentials are fake and whose transport is recorded."""
    settings = isolated_settings(
        google_service_account_json=NOT_A_KEY,
        google_sheets_spreadsheet_id="sheet-1",
    )
    mirror = SheetsMirror.__new__(SheetsMirror)
    mirror._credentials = _FakeCredentials()  # type: ignore[assignment]
    mirror._spreadsheet_id = settings.google_sheets_spreadsheet_id  # type: ignore[assignment]
    mirror._ready = False  # type: ignore[assignment]
    return mirror, []


def _install_mirror(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    """Point the module-level mirror and its HTTP client at a fake transport.

    The real AsyncClient is captured before the patch: building one inside
    the replacement would call the replacement again, forever.
    """
    real_client = httpx.AsyncClient
    mirror, _ = _mirror(handler)
    monkeypatch.setattr("app.services.sheets.get_mirror", lambda: mirror)
    monkeypatch.setattr(
        "app.services.sheets.httpx.AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler)),
    )


# --- the row the clinic reads ----------------------------------------------


TODAY = date(2026, 8, 28)


def test_the_columns_are_the_ones_the_clinic_asked_for() -> None:
    row = LeadRow("Nodira Karimova", "998901234567", "telegram", "tish", TODAY)

    assert row.as_cells() == [
        "2026-08-28",
        "Nodira Karimova",
        "998901234567",
        "telegram",
        "tish",
    ]
    assert HEADER == ("Sana", "Ism familiya", "Telefon", "Manba", "Bemor haqida qisqacha")


def test_a_missing_name_or_number_is_an_empty_cell_not_the_word_none() -> None:
    """A lead usually arrives before the name does. "None" in a spreadsheet
    column reads as a person called None.
    """
    assert LeadRow(None, None, "instagram", None, TODAY).as_cells() == [
        "2026-08-28",
        "",
        "",
        "instagram",
        "",
    ]


def test_a_long_story_is_trimmed_so_the_column_stays_readable() -> None:
    row = LeadRow(None, "1", "telegram", "a" * 500, TODAY)

    cell = row.as_cells()[4]

    assert len(cell) == MAX_COMMENT
    assert cell.endswith("…")


def test_newlines_are_flattened_so_one_lead_stays_one_row() -> None:
    row = LeadRow(None, "1", "telegram", "tishim\nog'riyapti", TODAY)

    assert row.as_cells()[4] == "tishim og'riyapti"


# --- what the comment says --------------------------------------------------


def test_the_comment_is_what_the_patient_actually_wrote() -> None:
    """Their own words rather than a generated summary: another model call
    per lead is an allowance better spent answering patients, and "chap
    tomonda tishim og'riyapti" is more use to a clinic than a paraphrase.
    """
    said = ["Salom", "chap tomonda tishim og'riyapti, sovuqdan achishadi", "ok"]

    assert summarise_problem(said) == "chap tomonda tishim og'riyapti, sovuqdan achishadi"


def test_agreeing_to_a_time_is_not_the_reason_they_wrote() -> None:
    """From a real conversation: taking the longest message filled the
    clinic's column with "ha, shu vaqt to'g'ri keladi", because a
    confirmation is wordier than the question under it. The first
    substantive message cannot be agreement — nothing has been offered when
    it arrives.
    """
    said = [
        "Assalomu alaykum",
        "implant qancha turadi?",
        "qimmat ekan",
        "mayli, qabulga yozing",
        "ha, shu vaqt to'g'ri keladi",
    ]

    assert summarise_problem(said) == "implant qancha turadi?"


@pytest.mark.parametrize(
    "said",
    [
        ["Salom", "assalomu alaykum", "Здравствуйте", "/start", "ok", "rahmat"],
        ["+998 90 123 45 67"],
        [],
        ["   "],
    ],
)
def test_greetings_and_bare_numbers_say_nothing_about_the_problem(said: list[str]) -> None:
    assert summarise_problem(said) == ""


def test_a_greeting_with_a_question_attached_is_not_small_talk() -> None:
    assert summarise_problem(["salom, implant qancha turadi?"]) == "salom, implant qancha turadi?"


# --- what is sent to Google -------------------------------------------------


async def test_a_lead_is_written_to_the_hidden_sheet_with_its_date() -> None:
    """The date is what the picker filters on, so it goes in with the row
    rather than being worked out later.
    """
    written: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and ":append" in str(request.url):
            written.append(json.loads(request.content)["values"][0])
        return httpx.Response(200, json={})

    mirror, _ = _mirror(handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await mirror.append(client, LeadRow("N", "998901234567", "telegram", "tish", TODAY))

    assert written == [["2026-08-28", "N", "998901234567", "telegram", "tish"]]


async def test_the_same_patient_on_the_same_day_is_one_row() -> None:
    """Their name usually arrives a few turns after their number. Appending
    again would give the owner the same person twice.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "!A:C" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "values": [
                        ["Sana", "Ism familiya", "Telefon"],
                        [_sheets_serial(TODAY), "", "998901234567"],
                    ]
                },
            )
        return httpx.Response(200, json={})

    mirror, _ = _mirror(handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await mirror.find_row(client, "998901234567", TODAY) == 2


async def test_the_same_patient_on_a_different_day_is_a_new_row() -> None:
    """Somebody writing again next week is a separate lead on a separate day,
    which is the whole point of being able to pick a date.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "!A:C" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "values": [
                        ["Sana", "Ism", "Telefon"],
                        [_sheets_serial(date(2026, 8, 20)), "", "998901234567"],
                    ]
                },
            )
        return httpx.Response(200, json={})

    mirror, _ = _mirror(handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await mirror.find_row(client, "998901234567", TODAY) is None


async def test_a_lead_with_no_number_never_matches_an_existing_row() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"values": [["Sana", "Ism", "Telefon"]]})

    mirror, _ = _mirror(handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        assert await mirror.find_row(client, "", TODAY) is None


async def test_googles_refusal_is_raised_with_enough_to_act_on() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, text='{"error":{"message":"The caller does not have permission"}}'
        )

    mirror, _ = _mirror(handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SheetsError, match="403"):
            # A read, not the styling: _style swallows what it is handed, and
            # what is being checked here is that the error carries the status
            # and Google's own message.
            await mirror.find_row(client, "998901234567", TODAY)


# --- never at the patient's expense -----------------------------------------


async def test_a_broken_sheet_never_reaches_the_patient(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The reply has already gone out and the row is already in the
    database. A spreadsheet that is briefly behind costs nothing; a job that
    fails here would be retried and send the reply twice.
    """

    class _Exploding:
        async def upsert(self, row: LeadRow) -> None:
            raise SheetsError("Sheets API 404: no such tab")

    monkeypatch.setattr("app.services.sheets.get_mirror", lambda: _Exploding())

    with caplog.at_level("ERROR", logger="app"):
        await mirror_lead(LeadRow("N", "1", "telegram", "x", TODAY))

    assert "sheets_mirror_failed" in caplog.text


async def test_no_sheet_configured_is_silence_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.sheets.get_mirror", lambda: None)

    await mirror_lead(LeadRow("N", "1", "telegram", "x", TODAY))


# --- the key ----------------------------------------------------------------


def test_a_base64_key_is_accepted_because_a_dotenv_cannot_hold_newlines() -> None:
    """private_key contains newlines. Railway can carry those in a variable
    and a line-by-line .env cannot, so both spellings are accepted rather
    than making it work on one host and not the other.
    """
    encoded = base64.b64encode(NOT_A_KEY.encode()).decode()

    with pytest.raises(SheetsError, match="Not a usable service account key"):
        _load_credentials(encoded)  # decoded fine, then rejected for its contents


@pytest.mark.parametrize("raw", ["not json at all", "!!!!", "{oops}"])
def test_a_key_that_is_neither_json_nor_base64_says_so(raw: str) -> None:
    with pytest.raises(SheetsError):
        _load_credentials(raw)


# --- how a new tab is laid out ----------------------------------------------


def test_a_new_tab_is_made_readable_rather_than_left_as_grey_columns() -> None:
    """This sheet exists because the owner will not open a dashboard, so
    looking like something they will read is not decoration.
    """
    kinds = [next(iter(request)) for request in _layout_requests(0)]

    assert "updateSheetProperties" in kinds  # frozen header
    # The filter is the mechanism the whole feature rests on, so it travels
    # with the layout rather than with the paint.
    assert "setBasicFilter" in kinds
    assert "addBanding" in [next(iter(r)) for r in _decoration_requests(0)]


def test_the_phone_column_is_text_so_it_is_not_shown_as_9_989e_11() -> None:
    formats = {
        request["repeatCell"]["range"]["startColumnIndex"]: request["repeatCell"]["cell"][
            "userEnteredFormat"
        ]["numberFormat"]
        for request in _layout_requests(0)
        if "repeatCell" in request and "numberFormat" in request["repeatCell"]["fields"]
    }

    assert formats[HEADER.index("Telefon")]["type"] == "TEXT"
    # And the date is a real date, which is what puts "Kecha / Bugun / Erta"
    # in the filter menu.
    assert formats[HEADER.index("Sana")]["type"] == "DATE"


def test_the_story_column_wraps_instead_of_running_under_the_next_one() -> None:
    wrapping = [
        request["repeatCell"]
        for request in _layout_requests(0)
        if "repeatCell" in request
        and request["repeatCell"]["cell"]["userEnteredFormat"].get("wrapStrategy") == "WRAP"
    ]

    assert wrapping


async def test_styling_a_tab_never_costs_the_clinic_a_lead(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A tab that works but looks plain is a far better outcome than a lead
    that was not written because the decoration failed.
    """
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if str(request.url).endswith(":batchUpdate"):
            return httpx.Response(500, text="styling exploded")
        return httpx.Response(200, json={})

    mirror, _ = _mirror(handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with caplog.at_level("WARNING", logger="app"):
            await mirror._style(client, 0)
        await mirror.append(client, LeadRow("N", "1", "telegram", "x", TODAY))

    assert "sheets_styling_skipped" in caplog.text
    assert calls[-1] == "POST"  # the row still went in


def test_the_row_carries_the_day_it_belongs_to() -> None:
    """Not the day the patient wrote. Somebody who messages on the 28th to be
    seen on the 31st belongs in the 31st's list, because that is the list the
    front desk works from that morning.
    """
    booked = LeadRow("Kelajak", "998901110031", "telegram", "31 avgustga", date(2026, 8, 31))

    assert booked.as_cells()[0] == "2026-08-31"


def test_the_date_goes_in_as_iso_whatever_it_is_shown_as() -> None:
    """ISO is the shape Sheets parses regardless of the spreadsheet's locale.
    "31.08.2026" is not, and would land as text on a sheet set to English —
    which would take the whole date filter with it.
    """
    cells = LeadRow(None, "1", "telegram", None, date(2026, 8, 31)).as_cells()

    assert cells[0] == "2026-08-31"


def test_a_day_becomes_the_serial_sheets_actually_stores() -> None:
    """find_row reads unformatted values, so it compares against the number
    Sheets keeps rather than whatever the locale renders it back as.
    """
    assert _sheets_serial(date(1899, 12, 31)) == 1
    assert _sheets_serial(date(2026, 8, 31)) == 46265


async def test_a_sheet_left_over_from_an_older_layout_is_rewritten() -> None:
    """A real patient was lost to this.

    The check used to be "is row 1 empty?", so a sheet still carrying the
    old picker layout -- "Sana:" in A1 -- was treated as already set up. The
    header was left alone, the styling was skipped, and the lead landed two
    columns right of where anyone reads. Any header that is not the current
    one belongs to a design that is gone.
    """
    written: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and "/values/" in str(request.url):
            return httpx.Response(200, json={"values": [["Sana:", "", "Telefon"]]})
        if request.method == "GET":
            return httpx.Response(
                200, json={"sheets": [{"properties": {"sheetId": 0, "title": VIEW_SHEET}}]}
            )
        if request.method == "PUT":
            written.extend(json.loads(request.content)["values"])
        return httpx.Response(200, json={})

    mirror, _ = _mirror(handler)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await mirror._ensure_sheet(client)

    assert written == [list(HEADER)]


# --- the appointment book -------------------------------------------------

BOOKING = uuid.UUID("3f2504e0-4f89-11d3-9a0c-0305e82c3301")


def _appointment(**over: object) -> AppointmentRow:
    fields: dict[str, object] = {
        "appointment_id": BOOKING,
        "created_at": datetime(2026, 8, 29, 9, 5, tzinfo=UTC),
        "scheduled_at": datetime(2026, 9, 2, 9, 30, tzinfo=UTC),  # 14:30 Toshkent
        "patient_name": "Erkin Shodmonov",
        "phone": "771997272",
        "doctor": "Dr. Aliyev A.A.",
        "channel": "instagram",
        "client_id": "17841400000",
        "status": "scheduled",
    }
    fields.update(over)
    return AppointmentRow(**fields)  # type: ignore[arg-type]


def test_the_booking_carries_the_time_it_was_agreed_for() -> None:
    """The whole point of the appointment book. A date alone tells the front
    desk somebody is coming; the time tells them when to expect them.
    """
    head = _appointment().head_cells()

    assert head[4] == "2026-09-02"
    assert head[5] == "14:30"  # 09:30 UTC in the clinic's own time zone
    assert head[6] == "Dr. Aliyev A.A."


def test_the_reference_is_the_same_every_time_for_one_booking() -> None:
    """A retry has to find the row it already wrote. A counted number --
    MED-1001, MED-1002 -- would be read off the sheet and could be handed to
    two bookings a second apart; this one falls out of the booking's own id.
    """
    assert _appointment().reference == _appointment(status="confirmed").reference
    assert _appointment().reference.startswith("MED-")


def test_the_phone_is_written_in_the_shape_the_column_documents() -> None:
    """+998XXXXXXXXX. A patient types nine digits; the clinic dials twelve."""
    # The apostrophe is Sheets' own escape: without it USER_ENTERED reads a
    # leading "+" as a formula and the country code is lost to arithmetic.
    assert _appointment().head_cells()[3] == "'+998771997272"
    assert _appointment(phone="+998 90 111 22 33").head_cells()[3] == "'+998901112233"
    assert _appointment(phone=None).head_cells()[3] == ""


def test_statuses_and_channels_are_written_in_the_sheets_own_words() -> None:
    """The sheet's dropdown offers "Kutilmoqda", not "scheduled". A status
    outside the list reads as invalid to the person filtering on it.
    """
    tail = _appointment().tail_cells()

    assert tail[0] == "Instagram"
    assert tail[2] == "Kutilmoqda"
    assert _appointment(status="cancelled").tail_cells()[2] == "Bekor qilindi"
    assert _appointment(status="no_show").tail_cells()[2] == "Kelmadi"


def test_the_service_column_is_left_for_a_person_to_fill() -> None:
    """The assistant agrees a time, not a procedure. Guessing a billable
    service from "tishim og'riyapti" is how somebody is charged for
    something nobody offered them.
    """
    assert _appointment().service_cell() == [""]
    assert _appointment(service="EKG").service_cell() == ["EKG"]


async def test_the_formula_columns_are_never_written_over() -> None:
    """H and J hold one spilled ARRAYFORMULA each. A value written into a
    spilled cell does not overwrite it -- it turns the whole column into
    #REF!, for every appointment at once. So the bot writes around them.
    """
    ranges: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if request.method == "GET" and "/values/" in url and "A1:O1" in url:
            return httpx.Response(200, json={"values": [list(APPOINTMENT_HEADER)]})
        if request.method == "GET" and "/values/" in url:
            return httpx.Response(200, json={"values": [["Qabul_ID"]]})
        if request.method == "GET":
            return httpx.Response(
                200, json={"sheets": [{"properties": {"sheetId": 0, "title": "Qabullar"}}]}
            )
        if "append" in url:
            return httpx.Response(200, json={"updates": {"updatedRange": "Qabullar!A2:G2"}})
        if "values:batchUpdate" in url:
            body = json.loads(request.content)
            ranges.extend(entry["range"] for entry in body["data"])
        return httpx.Response(200, json={})

    mirror, _ = _mirror(handler)
    mirror._appointments_ready = False  # type: ignore[attr-defined]
    real_client = httpx.AsyncClient

    def fake(*args: object, **kwargs: object) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler))

    with patch("app.services.sheets.httpx.AsyncClient", fake):
        await mirror.upsert_appointment(_appointment())

    assert ranges == ["Qabullar!A2:G2", "Qabullar!I2", "Qabullar!K2:O2"]
    assert not any("H" in written or "J" in written for written in ranges)
