"""The appointment book, month by month: where a booking goes, when a month
line is drawn, and how an old unordered book is put in order.

Days are Sheets' own day numbers: 46286 is Monday 21 September 2026.
"""

from datetime import date
from typing import Any

import pytest

from app.services.sheet_months import (
    MARKER,
    BookRow,
    Placement,
    block_month,
    booking_cells,
    insert_requests,
    label,
    line_requests,
    line_rule,
    month_of,
    parse,
    place,
    plan,
    serial,
)

HEADER = ["Bemor", "Telefon", "Sana", "Vaqti", "Kanal", "Status", "Izoh", "Kod"]
SEP_21 = serial(date(2026, 9, 21))
SEP_28 = serial(date(2026, 9, 28))
OCT_3 = serial(date(2026, 10, 3))
AUG_30 = serial(date(2026, 8, 30))
NOON, MORNING = 0.5, 0.375


def booking(day: float, time: float = NOON, code: str = "MED-000001") -> list[Any]:
    return ["Bemor", "+998900000000", day, time, "Instagram", "Kutilmoqda", "", code]


def line(month: str) -> list[Any]:
    return [month, "", "", "", "", "", "", MARKER]


# --- reading the book ---------------------------------------------------------------------


def test_serials_match_sheets() -> None:
    assert SEP_21 == 46286
    assert month_of(46286) == (2026, 9)


def test_lines_bookings_and_gaps_are_told_apart() -> None:
    rows = parse([HEADER, line("Sentabr 2026"), booking(SEP_21), [], ["", "", "", ""]])
    assert rows[0] == BookRow(2, month=(2026, 9))
    assert rows[1].day == SEP_21 and not rows[1].is_line
    assert rows[2].empty and rows[3].empty


@pytest.mark.parametrize(
    "cells",
    [
        ["Sentabr 2026", "", "", "", "", "", "", ""],  # typed by hand, no marker
        ["  oktabr 2026 ", "", "", "", "", "", "", ""],
        ["Nimadir", "", "", "", "", "", "", MARKER],  # marker without a month: not a line
    ],
)
def test_a_line_typed_by_hand_is_recognised_and_a_bare_marker_is_not(cells: list[Any]) -> None:
    [row] = parse([HEADER, cells])
    assert row.is_line == (cells[0].strip().split()[0].lower() in ("sentabr", "oktabr"))


def test_a_patient_called_like_a_month_is_still_a_booking() -> None:
    """ "May 2026" in the name column of a real booking has a date beside it."""
    [row] = parse([HEADER, ["May 2026", "+1", SEP_21, NOON, "", "", "", "MED-1"]])
    assert not row.is_line and row.day == SEP_21


@pytest.mark.parametrize("text", ["2026-09-26", "26.09.2026", "Shanba 26.09.2026"])
def test_a_date_typed_as_text_is_read_as_a_date(text: str) -> None:
    [row] = parse([HEADER, ["A", "", text, "12:20", "", "", "", "MED-1"]])
    assert row.day == serial(date(2026, 9, 26))
    assert row.time == pytest.approx((12 * 60 + 20) / 1440)


# --- where a booking goes -------------------------------------------------------------------


BOOK = parse(
    [
        HEADER,
        line("Sentabr 2026"),  # 2
        booking(SEP_21, MORNING),  # 3
        booking(SEP_28),  # 4
        line("Oktabr 2026"),  # 5
        booking(OCT_3),  # 6
    ]
)


def test_a_booking_goes_under_its_month_in_the_order_of_the_day() -> None:
    assert place(BOOK, SEP_21, NOON) == Placement(row=4)  # after the 21st's morning
    assert place(BOOK, SEP_21, 0.3) == Placement(row=3)  # before it
    assert place(BOOK, SEP_28, 0.9) == Placement(row=5)  # end of September, above October
    assert place(BOOK, OCT_3, 0.9) == Placement(row=7)


def test_a_new_month_line_goes_where_the_calendar_puts_it() -> None:
    november = serial(date(2026, 11, 2))
    assert place(BOOK, november, NOON) == Placement(row=8, line_at=7)  # after October
    assert place(BOOK, AUG_30, NOON) == Placement(row=3, line_at=2)  # before September
    december_last_year = serial(date(2025, 12, 31))
    assert place(BOOK, december_last_year, NOON) == Placement(row=3, line_at=2)


def test_the_first_booking_of_an_empty_book_opens_its_month() -> None:
    assert place(parse([HEADER]), SEP_21, NOON) == Placement(row=3, line_at=2)


def test_a_month_with_a_line_and_no_bookings_yet_takes_the_row_under_it() -> None:
    book = parse([HEADER, line("Sentabr 2026"), line("Oktabr 2026"), booking(OCT_3)])
    assert place(book, SEP_28, NOON) == Placement(row=3)


def test_gaps_left_in_the_book_do_not_pull_a_booking_away_from_its_month() -> None:
    book = parse([HEADER, line("Sentabr 2026"), booking(SEP_21), [], [], booking(SEP_28)])
    assert place(book, SEP_28, 0.9) == Placement(row=7)
    assert place(book, SEP_21, 0.9) == Placement(row=6)


def test_which_month_a_row_sits_under() -> None:
    assert block_month(BOOK, 4) == (2026, 9)
    assert block_month(BOOK, 6) == (2026, 10)
    assert block_month(parse([HEADER, booking(SEP_21)]), 2) is None


# --- the requests -------------------------------------------------------------------------


CELLS = booking_cells(
    "Ali", "+998901234567", SEP_21, NOON, "Instagram", "Kutilmoqda", "", "MED-1", '"X" dd'
)


def test_a_booking_is_written_as_a_real_date_time_and_text_phone() -> None:
    assert CELLS[1] == {"userEnteredValue": {"stringValue": "+998901234567"}}
    assert CELLS[2]["userEnteredValue"] == {"numberValue": SEP_21}
    assert CELLS[2]["userEnteredFormat"]["numberFormat"]["pattern"] == '"X" dd'
    assert CELLS[3]["userEnteredFormat"]["numberFormat"] == {"type": "TIME", "pattern": "HH:mm"}
    assert len(CELLS) == 8


def test_an_insert_is_one_batch_that_inserts_before_it_writes() -> None:
    requests = insert_requests(7, BOOK, Placement(row=4), CELLS, template_row=999, new_month=None)
    kinds = [next(iter(r)) for r in requests]
    assert kinds == ["insertDimension", "copyPaste", "copyPaste", "updateCells"]
    assert requests[0]["insertDimension"]["range"]["startIndex"] == 3
    # The template row was pushed down by the insert.
    assert requests[1]["copyPaste"]["source"]["startRowIndex"] == 1000
    assert {r["copyPaste"]["pasteType"] for r in requests[1:3]} == {
        "PASTE_FORMAT",
        "PASTE_DATA_VALIDATION",
    }
    assert requests[3]["updateCells"]["range"]["startRowIndex"] == 3


def test_a_new_month_draws_its_line_then_the_booking_under_it() -> None:
    requests = insert_requests(
        7, BOOK, Placement(row=8, line_at=7), CELLS, template_row=999, new_month=(2026, 11)
    )
    inserts = [
        r["insertDimension"]["range"]["startIndex"] for r in requests if "insertDimension" in r
    ]
    assert inserts == [6, 7]
    written = [r["updateCells"] for r in requests if "updateCells" in r]
    assert written[0]["rows"][0]["values"][0]["userEnteredValue"]["stringValue"] == "Noyabr 2026"
    assert written[0]["rows"][0]["values"][7]["userEnteredValue"]["stringValue"] == MARKER
    assert written[1]["range"]["startRowIndex"] == 7
    copies = [r["copyPaste"]["source"]["startRowIndex"] for r in requests if "copyPaste" in r]
    assert copies == [1001, 1001]  # two inserts above it


def test_a_month_line_has_no_dropdown_and_spans_the_visible_columns() -> None:
    requests = line_requests(7, 5, 2026, 10)
    [validation] = [r["setDataValidation"] for r in requests if "setDataValidation" in r]
    assert "rule" not in validation and validation["range"]["startRowIndex"] == 4
    [merge] = [r["mergeCells"]["range"] for r in requests if "mergeCells" in r]
    assert (merge["startColumnIndex"], merge["endColumnIndex"]) == (0, 7)
    assert label(2026, 10) == "Oktabr 2026"


def test_the_line_colour_rule_carries_no_separator() -> None:
    formula = line_rule(7)["addConditionalFormatRule"]["rule"]["booleanRule"]["condition"][
        "values"
    ][0]["userEnteredValue"]
    assert "," not in formula and ";" not in formula and MARKER in formula


# --- putting an old book in order ----------------------------------------------------------


def test_the_live_book_is_planned_as_it_was_found() -> None:
    """The shape of the clinic's book on the day: bookings in 2-9, an empty
    stretch, two more at the bottom, one date typed as text."""
    raw: list[list[Any]] = [HEADER]
    raw += [booking(SEP_21 + i) for i in range(7)]
    raw.append(["Азиз", "+998999116282", "2026-09-26", 0.51, "Instagram", "Kutilmoqda", "", "M"])
    raw += [[] for _ in range(220)]
    raw += [booking(SEP_28, 0.48), booking(SEP_28 + 1, 0.4)]

    tidy = plan(parse(raw), raw)

    assert tidy.delete == list(range(229, 9, -1))  # the gap, bottom first
    assert tidy.text_dates == {9: float(serial(date(2026, 9, 26)))}
    assert tidy.months == [(2026, 9)]


def test_old_lines_are_taken_out_to_be_drawn_again_and_trailing_blanks_left() -> None:
    raw = [
        HEADER,
        line("Sentabr 2026"),
        booking(SEP_21),
        [],
        line("Oktabr 2026"),
        booking(OCT_3),
        [],
    ]
    tidy = plan(parse(raw), raw)
    assert tidy.delete == [5, 4, 2]
    assert tidy.months == [(2026, 9), (2026, 10)]
