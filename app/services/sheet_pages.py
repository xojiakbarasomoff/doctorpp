"""The front page of the clinic's spreadsheet: the year, and each month in it.

The appointment book (Qabullar) is one long list, and the doctor asked for
what a paper diary has: open the year, pick the month, see that month's
patients -- on the same page, with an ✕ to go back. So everything lives on
one tab, Home:

    top      the year and twelve month tiles
    below    one section per month the clinic has asked for, far enough
             down to be out of sight until a tile is clicked

A tile is a link to the top of its month's section, and the ✕ at the top of
a section is a link back to the top of the page. Plain links, because a
spreadsheet without scripts has no clicks of its own -- and a link to a cell
is the one kind of navigation Sheets does inside a page.

A section holds no copy of anything. It is a single formula over Qabullar:
that month's bookings, in the order the day goes. A booking the assistant
writes at three in the morning is on Home when the doctor opens it, and
nothing here has to run for that.

Built on request (`python -m app.services.sheet_pages 2026 9`), not on every
booking: the page is layout, and a clinic's layout changes when a person
decides it should.
"""

import asyncio
import logging
import sys
from collections.abc import Sequence
from typing import Any

import httpx

from app.services.sheets import (
    _STATUS_COLOURS as STATUS_COLOURS,
)
from app.services.sheets import (
    APPOINTMENT_SHEET,
    STATUS_CHOICES,
    SheetsMirror,
    _rgb,
)

logger = logging.getLogger(__name__)

HOME_SHEET = "Home"
MONTHS_UZ = (
    "Yanvar", "Fevral", "Mart", "Aprel", "May", "Iyun",
    "Iyul", "Avgust", "Sentabr", "Oktabr", "Noyabr", "Dekabr",
)  # fmt: skip
WEEKDAYS = "Dushanba,Seshanba,Chorshanba,Payshanba,Juma,Shanba,Yakshanba"
EMPTY_MONTH = "Bu oyda hali qabul yo‘q"
CLOSE = "✕  Yopish"

# The palette of the rest of the spreadsheet: the teal of the Lidlar and
# Qabullar headers, a slate ink, and quiet greys for what is not there yet.
TEAL = "#0F766E"
TEAL_SOFT = "#CCFBF1"
INK = "#0F172A"
MUTED = "#64748B"
FAINT = "#94A3B8"
LINE = "#E2E8F0"
BAND = "#F8FAFC"
NOT_YET = "#F1F5F9"
WHITE = "#FFFFFF"

# --- the grid --------------------------------------------------------------------------
#
# One set of columns for the tiles and the tables under them, so a month's
# table sits exactly under the tiles: four wide columns with narrow gaps,
# and the last column -- a margin beside the tiles -- wide enough to hold
# the comment in a table row.
#
#   A      B     C   D     E   F       G   H       I
#   margin tile  gap tile  gap tile    gap tile    margin / Izoh
_WIDTHS = (32, 190, 18, 190, 18, 190, 18, 190, 420)
_TILE_COLUMNS = (1, 3, 5, 7)
_YEAR_ROW, _FIRST_TILE_ROW = 4, 6
# Where the first month's section starts (0-based), and how many rows each
# month is given: a busy month is twenty patients a day, six days a week.
FIRST_SECTION_ROW = 40
SECTION_ROWS = 700
# Section rows, relative to its first row.
_TITLE, _KPI_LABELS, _KPI_VALUES, _HEADER, _DATA = 0, 2, 3, 5, 6
# The table's columns in the grid: when, patient, phone, status, comment.
_WHEN, _PATIENT, _PHONE, _STATUS, _NOTE = 1, 3, 5, 7, 8
TABLE_HEADER = {
    _WHEN: "Kun · sana · vaqt",
    _PATIENT: "Bemor",
    _PHONE: "Telefon",
    _STATUS: "Status",
    _NOTE: "Izoh",
}

# --- formulas ----------------------------------------------------------------------


def localise(formula: str, separator: str) -> str:
    """Written with commas; stored with the spreadsheet's own separator.

    The API parses a formula the way the spreadsheet's locale would: on the
    clinic's ru_RU spreadsheet every argument separator is ";" and a formula
    with commas is #ERROR!. Commas inside a quoted string are text, not
    separators, and stay.
    """
    if separator == ",":
        return formula
    out: list[str] = []
    quoted = False
    for character in formula:
        if character == '"':
            quoted = not quoted
        out.append(separator if character == "," and not quoted else character)
    return "".join(out)


def separator_for(locale: str | None) -> str:
    """ "," where the decimal point is a dot, ";" where it is a comma."""
    dot_decimal = ("en", "ja", "zh", "ko", "th", "he", "iw", "hi", "ms", "fil", "ga")
    return "," if (locale or "en").split("_")[0].lower() in dot_decimal else ";"


def _month_dates(year: int, month: int) -> tuple[str, str]:
    """DATE() of the first day, and of the first day after -- December's
    "after" is DATE(year, 13, 1), which Sheets rolls into January."""
    return f"DATE({year}, {month}, 1)", f"DATE({year}, {month + 1}, 1)"


# Qabullar's Sana as a real day: a date cell as it is, a date typed as text
# ("2026-09-26" -- two rows on the live sheet) read as one, anything else 0,
# which no month matches. Inside ARRAYFORMULA because LET does not spread
# IF over a range by itself.
_DAYS = (
    f"ARRAYFORMULA(IF(ISNUMBER({APPOINTMENT_SHEET}!C2:C), {APPOINTMENT_SHEET}!C2:C, "
    f"IFERROR(DATEVALUE({APPOINTMENT_SHEET}!C2:C), 0)))"
)


def month_formula(year: int, month: int) -> str:
    """The month's bookings as eight grid columns, B to I.

    Sorted by day and time. UNIQUE because the live book holds hand-made
    copies of rows; a copy that matches exactly is one booking, and a copy
    that differs stays visible so the difference can be seen. The gap
    columns get empty strings, so the table lines up under the tiles.
    """
    first, after = _month_dates(year, month)
    book = APPOINTMENT_SHEET
    when = (
        f'INDEX(SPLIT("{WEEKDAYS}", ","), 1, WEEKDAY(day, 2)) & ", " & '
        f'TEXT(day, "dd.mm") & "  ·  " & TEXT(slot, "hh:mm")'
    )
    return (
        f"=LET(d, {_DAYS}, "
        f"s, SORT(UNIQUE(FILTER(HSTACK(d, {book}!D2:D, {book}!A2:A, {book}!B2:B, {book}!F2:G), "
        f"d >= {first}, d < {after})), 1, TRUE, 2, TRUE), "
        f'gap, MAP(CHOOSECOLS(s, 1), LAMBDA(x, "")), '
        f"IFERROR(HSTACK("
        f"MAP(CHOOSECOLS(s, 1), CHOOSECOLS(s, 2), LAMBDA(day, slot, {when})), gap, "
        f"CHOOSECOLS(s, 3), gap, CHOOSECOLS(s, 4), gap, CHOOSECOLS(s, 5), CHOOSECOLS(s, 6)"
        f'), "{EMPTY_MONTH}"))'
    )


def month_title(year: int, month: int) -> str:
    return f"{MONTHS_UZ[month - 1]} {year}"


def _a1(row: int, column: int) -> str:
    """0-based (row, column) as an A1 reference."""
    return f"{chr(ord('A') + column)}{row + 1}"


def _jump(home_gid: int, row: int, column: int, text: str) -> str:
    return f'=HYPERLINK("#gid={home_gid}&range={_a1(row, column)}", "{text}")'


# --- what goes where -------------------------------------------------------------------


def tile_position(month: int) -> tuple[int, int]:
    """(row of the month's name, column) for a month's tile, 0-based."""
    index = month - 1
    return _FIRST_TILE_ROW + (index // 4) * 3, _TILE_COLUMNS[index % 4]


def section_rows(months: Sequence[int]) -> dict[int, int]:
    """The first row of each month's section, in calendar order."""
    return {
        month: FIRST_SECTION_ROW + index * SECTION_ROWS
        for index, month in enumerate(sorted(set(months)))
    }


def _status_range(top: int) -> str:
    first = top + _DATA
    column = chr(ord("A") + _STATUS)
    return f"{column}{first + 1}:{column}{first + SECTION_ROWS - _DATA}"


def home_values(year: int, months: Sequence[int], home_gid: int) -> dict[tuple[int, int], str]:
    """Cell (row, column) -> what it holds, for the whole of Home."""
    sections = section_rows(months)
    cells: dict[tuple[int, int], str] = {
        (1, 1): "Dr. Temur — qabullar",
        (2, 1): "Oyni tanlang: o‘sha oyning qabullari shu sahifada ochiladi.",
        (_YEAR_ROW, 1): str(year),
    }
    for month in range(1, 13):
        row, column = tile_position(month)
        name = MONTHS_UZ[month - 1]
        top = sections.get(month)
        if top is None:
            cells[(row, column)] = name
            cells[(row + 1, column)] = "Tez orada"
            continue
        cells[(row, column)] = _jump(home_gid, top, 0, name)
        # The section's own "Jami", so the tile and the table cannot disagree.
        total = _a1(top + _KPI_VALUES, _WHEN)
        cells[(row + 1, column)] = (
            f'=HYPERLINK("#gid={home_gid}&range={_a1(top, 0)}", {total} & " ta qabul  →")'
        )
    footer = tile_position(12)[0] + 3
    cells[(footer, 1)] = (
        "Yangi qabullar bot orqali Qabullar varag‘iga tushadi va shu yerda o‘zi paydo bo‘ladi."
    )

    for month, top in sections.items():
        statuses = _status_range(top)
        cells[(top + _TITLE, _WHEN)] = f"{month_title(year, month)} — qabullar"
        cells[(top + _TITLE, _STATUS)] = _jump(home_gid, 0, 0, CLOSE)
        cells[(top + _KPI_LABELS, _WHEN)] = "Jami"
        cells[(top + _KPI_LABELS, _PATIENT)] = "Kutilmoqda"
        cells[(top + _KPI_LABELS, _PHONE)] = "Keldi · Kelmadi"
        cells[(top + _KPI_LABELS, _STATUS)] = "Bekor qilindi"
        # Counted on Status: every booking has one, and an empty month's
        # one line of text is in the first column, not this one.
        cells[(top + _KPI_VALUES, _WHEN)] = f"=COUNTA({statuses})"
        cells[(top + _KPI_VALUES, _PATIENT)] = f'=COUNTIF({statuses}, "Kutilmoqda")'
        cells[(top + _KPI_VALUES, _PHONE)] = (
            f'=COUNTIF({statuses}, "Keldi") & "  ·  " & COUNTIF({statuses}, "Kelmadi")'
        )
        cells[(top + _KPI_VALUES, _STATUS)] = f'=COUNTIF({statuses}, "Bekor qilindi")'
        for column, label in TABLE_HEADER.items():
            cells[(top + _HEADER, column)] = label
        cells[(top + _DATA, _WHEN)] = month_formula(year, month)
    return cells


# --- the look -------------------------------------------------------------------------


def _cell_range(gid: int, row: int, col: int, rows: int = 1, cols: int = 1) -> dict[str, int]:
    return {
        "sheetId": gid,
        "startRowIndex": row,
        "endRowIndex": row + rows,
        "startColumnIndex": col,
        "endColumnIndex": col + cols,
    }


def _format(range_: dict[str, int], fmt: dict[str, Any], fields: str) -> dict[str, Any]:
    return {"repeatCell": {"range": range_, "cell": {"userEnteredFormat": fmt}, "fields": fields}}


def _text(
    size: int = 10, bold: bool = False, colour: str = INK, underline: bool = False
) -> dict[str, Any]:
    return {
        "fontFamily": "Inter",
        "fontSize": size,
        "bold": bold,
        "underline": underline,
        "foregroundColorStyle": {"rgbColor": _rgb(colour)},
    }


def _height(gid: int, start: int, end: int, pixels: int) -> dict[str, Any]:
    return {
        "updateDimensionProperties": {
            "range": {"sheetId": gid, "dimension": "ROWS", "startIndex": start, "endIndex": end},
            "properties": {"pixelSize": pixels},
            "fields": "pixelSize",
        }
    }


def _width(gid: int, column: int, pixels: int) -> dict[str, Any]:
    return {
        "updateDimensionProperties": {
            "range": {
                "sheetId": gid,
                "dimension": "COLUMNS",
                "startIndex": column,
                "endIndex": column + 1,
            },
            "properties": {"pixelSize": pixels},
            "fields": "pixelSize",
        }
    }


_TEXT_FIELDS = "userEnteredFormat(textFormat,horizontalAlignment,verticalAlignment)"
_FILL_FIELDS = (
    "userEnteredFormat(backgroundColorStyle,textFormat,horizontalAlignment,verticalAlignment)"
)
_TILE_FIELDS = (
    "userEnteredFormat(backgroundColorStyle,textFormat,horizontalAlignment,"
    "verticalAlignment,padding)"
)


def _tile_requests(gid: int, pages: set[int], current: int | None) -> list[dict[str, Any]]:
    """Twelve tiles. The current month solid teal; a month with a section
    white with a teal edge; a month without one grey."""
    requests: list[dict[str, Any]] = []
    for month in range(1, 13):
        row, column = tile_position(month)
        built = month in pages
        if month == current and built:
            fill, name_colour, count_colour, edge = TEAL, WHITE, TEAL_SOFT, TEAL
        elif built:
            fill, name_colour, count_colour, edge = WHITE, TEAL, MUTED, TEAL
        else:
            fill, name_colour, count_colour, edge = NOT_YET, FAINT, FAINT, LINE
        background = {"backgroundColorStyle": {"rgbColor": _rgb(fill)}}
        requests += [
            _format(
                _cell_range(gid, row, column),
                {
                    **background,
                    "textFormat": _text(15, True, name_colour),
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "BOTTOM",
                    "padding": {"left": 14, "top": 6},
                },
                _TILE_FIELDS,
            ),
            _format(
                _cell_range(gid, row + 1, column),
                {
                    **background,
                    "textFormat": _text(10, False, count_colour),
                    "horizontalAlignment": "LEFT",
                    "verticalAlignment": "TOP",
                    "padding": {"left": 14, "bottom": 6},
                },
                _TILE_FIELDS,
            ),
            {
                "updateBorders": {
                    "range": _cell_range(gid, row, column, rows=2),
                    **{
                        side: {"style": "SOLID", "colorStyle": {"rgbColor": _rgb(edge)}}
                        for side in ("top", "bottom", "left", "right")
                    },
                }
            },
        ]
    for block in range(3):
        top = _FIRST_TILE_ROW + block * 3
        requests += [
            _height(gid, top, top + 1, 46),
            _height(gid, top + 1, top + 2, 30),
            _height(gid, top + 2, top + 3, 18),
        ]
    return requests


def _section_requests(gid: int, top: int) -> list[dict[str, Any]]:
    """A month's section: title and ✕, the numbers, then the table."""
    data = top + _DATA
    rows = SECTION_ROWS - _DATA
    table = _cell_range(gid, data, 1, rows=rows, cols=8)
    requests: list[dict[str, Any]] = [
        {
            "mergeCells": {
                "range": _cell_range(gid, top + _TITLE, 1, cols=5),
                "mergeType": "MERGE_ALL",
            }
        },
        _format(
            _cell_range(gid, top + _TITLE, 1),
            {"textFormat": _text(20, True), "verticalAlignment": "MIDDLE"},
            _TEXT_FIELDS,
        ),
        # The ✕: a pill, at the top right of the table.
        _format(
            _cell_range(gid, top + _TITLE, _STATUS),
            {
                "backgroundColorStyle": {"rgbColor": _rgb(NOT_YET)},
                "textFormat": _text(11, True, INK),
                "horizontalAlignment": "CENTER",
                "verticalAlignment": "MIDDLE",
            },
            _FILL_FIELDS,
        ),
        {
            "updateBorders": {
                "range": _cell_range(gid, top + _TITLE, _STATUS),
                **{
                    side: {"style": "SOLID", "colorStyle": {"rgbColor": _rgb(LINE)}}
                    for side in ("top", "bottom", "left", "right")
                },
            }
        },
        {
            "updateBorders": {
                "range": _cell_range(gid, top + _TITLE, 1, cols=8),
                "top": {"style": "SOLID_MEDIUM", "colorStyle": {"rgbColor": _rgb(TEAL)}},
            }
        },
        _format(
            _cell_range(gid, top + _KPI_LABELS, 1, cols=8),
            {"textFormat": _text(9, True, MUTED), "horizontalAlignment": "LEFT"},
            _TEXT_FIELDS,
        ),
        _format(
            _cell_range(gid, top + _KPI_VALUES, 1, cols=8),
            {"textFormat": _text(18, True), "horizontalAlignment": "LEFT"},
            _TEXT_FIELDS,
        ),
        _format(
            _cell_range(gid, top + _HEADER, 1, cols=8),
            {
                "backgroundColorStyle": {"rgbColor": _rgb(TEAL)},
                "textFormat": _text(10, True, WHITE),
                "horizontalAlignment": "LEFT",
                "verticalAlignment": "MIDDLE",
            },
            _FILL_FIELDS,
        ),
        _format(
            table,
            {
                "textFormat": _text(10),
                "horizontalAlignment": "LEFT",
                "verticalAlignment": "MIDDLE",
                "wrapStrategy": "CLIP",
            },
            "userEnteredFormat(textFormat,horizontalAlignment,verticalAlignment,wrapStrategy)",
        ),
        _format(
            _cell_range(gid, data, _WHEN, rows=rows),
            {"textFormat": _text(10, True, TEAL)},
            "userEnteredFormat.textFormat",
        ),
        _height(gid, top + _TITLE, top + _TITLE + 1, 56),
        _height(gid, top + _TITLE + 1, top + _KPI_LABELS, 10),
        _height(gid, top + _KPI_VALUES, top + _KPI_VALUES + 1, 36),
        _height(gid, top + _HEADER, top + _HEADER + 1, 32),
        _height(gid, data, data + rows, 28),
        # Quiet stripes on filled rows only; a formula with no separator in
        # it, so the locale cannot break it.
        {
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [table],
                    "booleanRule": {
                        "condition": {
                            "type": "CUSTOM_FORMULA",
                            "values": [{"userEnteredValue": f'=($H{data + 1}<>"")*ISEVEN(ROW())'}],
                        },
                        "format": {"backgroundColorStyle": {"rgbColor": _rgb(BAND)}},
                    },
                },
                "index": 0,
            }
        },
    ]
    for name in STATUS_CHOICES:
        fill, text = STATUS_COLOURS[name]
        requests.append(
            {
                "addConditionalFormatRule": {
                    "rule": {
                        "ranges": [_cell_range(gid, data, _STATUS, rows=rows)],
                        "booleanRule": {
                            "condition": {
                                "type": "TEXT_EQ",
                                "values": [{"userEnteredValue": name}],
                            },
                            "format": {
                                "backgroundColorStyle": {"rgbColor": _rgb(fill)},
                                "textFormat": {
                                    "foregroundColorStyle": {"rgbColor": _rgb(text)},
                                    "bold": True,
                                },
                            },
                        },
                    },
                    "index": 0,
                }
            }
        )
    # The numbers in their status colours, read at a glance.
    for column, name in ((_PATIENT, "Kutilmoqda"), (_STATUS, "Bekor qilindi")):
        requests.append(
            _format(
                _cell_range(gid, top + _KPI_VALUES, column),
                {"textFormat": _text(18, True, STATUS_COLOURS[name][1])},
                "userEnteredFormat.textFormat",
            )
        )
    return requests


def home_requests(
    gid: int, year: int, months: Sequence[int], current: int | None
) -> list[dict[str, Any]]:
    """The look of Home, top to bottom."""
    sections = section_rows(months)
    requests: list[dict[str, Any]] = [
        {
            "updateSheetProperties": {
                "properties": {
                    "sheetId": gid,
                    "gridProperties": {"hideGridlines": True, "frozenRowCount": 0},
                    "tabColorStyle": {"rgbColor": _rgb(TEAL)},
                },
                "fields": (
                    "gridProperties.hideGridlines,gridProperties.frozenRowCount,tabColorStyle"
                ),
            }
        },
        {"mergeCells": {"range": _cell_range(gid, 1, 1, cols=7), "mergeType": "MERGE_ALL"}},
        {"mergeCells": {"range": _cell_range(gid, 2, 1, cols=7), "mergeType": "MERGE_ALL"}},
        {
            "mergeCells": {
                "range": _cell_range(gid, _YEAR_ROW, 1, cols=7),
                "mergeType": "MERGE_ALL",
            }
        },
        _format(_cell_range(gid, 1, 1), {"textFormat": _text(22, True)}, _TEXT_FIELDS),
        _format(_cell_range(gid, 2, 1), {"textFormat": _text(10, colour=MUTED)}, _TEXT_FIELDS),
        _format(
            _cell_range(gid, _YEAR_ROW, 1, cols=7),
            {
                "textFormat": _text(26, True, TEAL),
                "horizontalAlignment": "LEFT",
                "verticalAlignment": "BOTTOM",
            },
            _TEXT_FIELDS,
        ),
        {
            "updateBorders": {
                "range": _cell_range(gid, _YEAR_ROW, 1, cols=7),
                "bottom": {"style": "SOLID_MEDIUM", "colorStyle": {"rgbColor": _rgb(TEAL)}},
            }
        },
        _height(gid, 0, 1, 24),
        _height(gid, 1, 2, 48),
        _height(gid, 3, 4, 12),
        _height(gid, _YEAR_ROW, _YEAR_ROW + 1, 52),
        _height(gid, _YEAR_ROW + 1, _YEAR_ROW + 2, 18),
    ]
    requests.extend(_width(gid, index, width) for index, width in enumerate(_WIDTHS))
    requests += _tile_requests(gid, set(sections), current)
    footer = tile_position(12)[0] + 3
    requests.append(
        _format(_cell_range(gid, footer, 1), {"textFormat": _text(9, colour=FAINT)}, _TEXT_FIELDS)
    )
    for top in sections.values():
        requests += _section_requests(gid, top)
    return requests


# --- publishing ----------------------------------------------------------------------


def _grid(cells: dict[tuple[int, int], str], separator: str) -> list[list[str]]:
    rows = max(row for row, _ in cells) + 1
    columns = max(column for _, column in cells) + 1
    grid = [["" for _ in range(columns)] for _ in range(rows)]
    for (row, column), value in cells.items():
        grid[row][column] = localise(value, separator) if value.startswith("=") else value
    return grid


def rows_needed(months: Sequence[int]) -> int:
    sections = section_rows(months)
    return max(sections.values(), default=FIRST_SECTION_ROW) + SECTION_ROWS


class SheetPages:
    """Builds Home in the clinic's spreadsheet."""

    def __init__(self, mirror: SheetsMirror | None = None) -> None:
        self._mirror = mirror or SheetsMirror()

    async def _batch(
        self, client: httpx.AsyncClient, requests: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return await self._mirror._call(client, "POST", ":batchUpdate", json={"requests": requests})

    async def publish(self, year: int, months: Sequence[int], current: int | None = None) -> int:
        """Rebuild Home with a section for each of `months`. Returns Home's id.

        Home keeps its id -- links point at it, and somebody may have
        bookmarked it -- and is cleared in place: values, formats, merges,
        colour rules. A month tab from the earlier design, one tab per month,
        is removed: the month lives on Home now.
        """
        for month in months:
            if not 1 <= month <= 12:
                raise ValueError(f"no month {month}")
        async with httpx.AsyncClient(timeout=30) as client:
            body = await self._mirror._call(
                client,
                "GET",
                "",
                params={
                    "fields": "properties.locale,sheets(properties(sheetId,title,gridProperties),"
                    "merges,conditionalFormats)"
                },
            )
            separator = separator_for(body.get("properties", {}).get("locale"))
            sheets = {s["properties"]["title"]: s for s in body.get("sheets", [])}

            old_tabs = [
                sheets[title]["properties"]["sheetId"]
                for title in (month_title(year, m) for m in range(1, 13))
                if title in sheets
            ]
            if old_tabs:
                await self._batch(client, [{"deleteSheet": {"sheetId": gid}} for gid in old_tabs])

            if HOME_SHEET in sheets:
                home = sheets[HOME_SHEET]
                gid = home["properties"]["sheetId"]
            else:
                reply = await self._batch(
                    client, [{"addSheet": {"properties": {"title": HOME_SHEET, "index": 0}}}]
                )
                home = {}
                gid = reply["replies"][0]["addSheet"]["properties"]["sheetId"]

            have = home.get("properties", {}).get("gridProperties", {}).get("rowCount", 1000)
            reset: list[dict[str, Any]] = [
                {"deleteConditionalFormatRule": {"sheetId": gid, "index": 0}}
                for _ in home.get("conditionalFormats") or []
            ]
            reset += [{"unmergeCells": {"range": merge}} for merge in home.get("merges") or []]
            reset.append(
                {
                    "updateCells": {
                        "range": {"sheetId": gid},
                        "fields": "userEnteredValue,userEnteredFormat",
                    }
                }
            )
            need = rows_needed(months)
            if have < need:
                reset.append(
                    {
                        "appendDimension": {
                            "sheetId": gid,
                            "dimension": "ROWS",
                            "length": need - have,
                        }
                    }
                )
            await self._batch(client, reset)

            await self._mirror._call(
                client,
                "PUT",
                f"/values/'{HOME_SHEET}'!A1",
                params={"valueInputOption": "USER_ENTERED"},
                json={"values": _grid(home_values(year, months, gid), separator)},
            )
            await self._batch(client, home_requests(gid, year, months, current))
            logger.info("sheet_home_built months=%s removed_tabs=%d", sorted(months), len(old_tabs))
            return int(gid)


async def _main(argv: Sequence[str]) -> None:
    year, *months = (int(value) for value in argv)
    await SheetPages().publish(year, months, current=months[-1] if months else None)


if __name__ == "__main__":  # pragma: no cover - a command, run by hand
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 3:
        print("usage: python -m app.services.sheet_pages YEAR MONTH [MONTH ...]")
        sys.exit(2)
    asyncio.run(_main(sys.argv[1:]))
    print("done")
