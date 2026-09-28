"""Home in the clinic's spreadsheet: the year and a tile per month, each a
link to its month's line in Qabullar.

The formulas were worked out against the live spreadsheet (a ru_RU one,
where "," is #ERROR!); these pin down what was learned there.
"""

from typing import Any

import pytest

from app.services.sheet_pages import (
    HOME_SHEET,
    SheetPages,
    home_requests,
    home_values,
    localise,
    separator_for,
    tile_count,
    tile_name,
    tile_position,
)

HOME, BOOK = 1412175323, 1798833211

# --- formulas -------------------------------------------------------------------------


def test_separators_become_the_locales_but_commas_in_text_stay() -> None:
    formula = '=IFERROR(SPLIT("a,b", ","), "x, y")'
    assert localise(formula, ";") == '=IFERROR(SPLIT("a,b"; ","); "x, y")'
    assert localise(formula, ",") == formula


@pytest.mark.parametrize(
    ("locale", "separator"),
    [("ru_RU", ";"), ("uz_UZ", ";"), ("de_DE", ";"), ("en_US", ","), ("en_GB", ","), (None, ",")],
)
def test_the_separator_follows_the_decimal_mark(locale: str | None, separator: str) -> None:
    assert separator_for(locale) == separator


def test_a_tile_finds_its_months_line_wherever_it_has_moved_to() -> None:
    """MATCH, not a row number: bookings inserted above October move its line."""
    assert tile_name(BOOK, 2026, 9) == (
        f'=IFERROR(HYPERLINK("#gid={BOOK}&range=A" & MATCH("Sentabr 2026", Qabullar!A:A, 0), '
        '"Sentabr"), "Sentabr")'
    )


def test_a_tile_counts_its_months_bookings_and_says_tez_orada_before_it_has_a_line() -> None:
    count = tile_count(BOOK, 2026, 9)
    assert (
        'COUNTIFS(Qabullar!C:C, ">=" & DATE(2026, 9, 1), Qabullar!C:C, "<" & DATE(2026, 10, 1))'
        in count
    )
    assert count.endswith(', "Tez orada")')
    assert "DATE(2026, 13, 1)" in tile_count(BOOK, 2026, 12)


def test_quotes_and_brackets_are_balanced_in_every_formula() -> None:
    for formula in (v for v in home_values(2026, BOOK).values() if v.startswith("=")):
        assert formula.count('"') % 2 == 0, formula
        assert formula.count("(") == formula.count(")"), formula


# --- where things are -------------------------------------------------------------------


def test_twelve_tiles_four_to_a_row_never_overlapping() -> None:
    positions = [tile_position(month) for month in range(1, 13)]
    taken = {(row + dy, column) for row, column in positions for dy in (0, 1)}
    assert len(taken) == 24
    assert tile_position(1) == (6, 1) and tile_position(12) == (12, 7)


def test_every_month_has_a_tile_that_links_once_its_line_exists() -> None:
    cells = home_values(2026, BOOK)
    assert cells[(4, 1)] == "2026"
    for month in range(1, 13):
        row, column = tile_position(month)
        assert cells[(row, column)] == tile_name(BOOK, 2026, month)
        assert cells[(row + 1, column)] == tile_count(BOOK, 2026, month)


# --- the look -----------------------------------------------------------------------------


def _formulas(requests: list[dict[str, Any]]) -> list[str]:
    return [
        r["addConditionalFormatRule"]["rule"]["booleanRule"]["condition"]["values"][0][
            "userEnteredValue"
        ]
        for r in requests
        if "addConditionalFormatRule" in r
    ]


def test_a_tile_lights_up_when_its_month_opens_and_goes_solid_in_its_month() -> None:
    formulas = _formulas(home_requests(HOME, 2026, ";"))
    assert len(formulas) == 12 * 4
    # Another sheet is reachable from a colour rule only through INDIRECT.
    assert all('INDIRECT("Qabullar!A:A")' in f for f in formulas)
    september = [f for f in formulas if "Sentabr 2026" in f]
    assert len(september) == 4
    assert sum("MONTH(TODAY()) = 9" in f for f in september) == 2
    # Stored with the spreadsheet's separator.
    assert all("," not in f for f in formulas)


def test_the_this_month_rule_is_checked_before_the_has_a_line_rule() -> None:
    """Rules are added at index 0 one after another: the last one added is
    the first one checked, so it must be the more specific."""
    first_tile = _formulas(home_requests(HOME, 2026, ","))[:4]
    assert "TODAY()" not in first_tile[0]
    assert "TODAY()" in first_tile[3]


# --- publishing, against a fake Sheets API ------------------------------------------------


class FakeMirror:
    """Answers the handful of calls SheetPages makes, and remembers them."""

    def __init__(self, sheets: dict[str, int], locale: str = "ru_RU", rules: int = 0) -> None:
        self.sheets = dict(sheets)
        self.locale = locale
        self.rules = rules
        self.calls: list[tuple[str, str, dict[str, Any]]] = []
        self._next = 5000

    async def _call(self, client: Any, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((method, path, kwargs))
        if method == "GET":
            return {
                "properties": {"locale": self.locale},
                "sheets": [
                    {
                        "properties": {
                            "sheetId": gid,
                            "title": title,
                            "gridProperties": {"rowCount": 1000},
                        },
                        "merges": [{"sheetId": gid, "startRowIndex": 1}] if title == "Home" else [],
                        "conditionalFormats": [{}] * self.rules if title == "Home" else [],
                    }
                    for title, gid in self.sheets.items()
                ],
            }
        if path == ":batchUpdate":
            replies: list[dict[str, Any]] = []
            for request in kwargs["json"]["requests"]:
                if "addSheet" in request:
                    self._next += 1
                    self.sheets[request["addSheet"]["properties"]["title"]] = self._next
                    replies.append({"addSheet": {"properties": {"sheetId": self._next}}})
                elif "deleteSheet" in request:
                    gid = request["deleteSheet"]["sheetId"]
                    self.sheets = {t: g for t, g in self.sheets.items() if g != gid}
                    replies.append({})
                else:
                    replies.append({})
            return {"replies": replies}
        return {}

    def requests(self) -> list[dict[str, Any]]:
        return [
            request
            for _, path, kw in self.calls
            if path == ":batchUpdate"
            for request in kw["json"]["requests"]
        ]

    def home(self) -> list[list[str]]:
        [values] = [kw["json"]["values"] for method, _, kw in self.calls if method == "PUT"]
        return values


async def test_publishing_links_the_tiles_to_qabullar_and_removes_old_month_tabs() -> None:
    mirror = FakeMirror({"Home": HOME, "Sentabr 2026": 77, "Lidlar": 0, "Qabullar": BOOK})

    gid = await SheetPages(mirror).publish(2026)  # type: ignore[arg-type]

    assert gid == HOME and mirror.sheets["Home"] == HOME
    assert "Sentabr 2026" not in mirror.sheets and "Qabullar" in mirror.sheets
    row, column = tile_position(9)
    assert mirror.home()[row][column] == localise(tile_name(BOOK, 2026, 9), ";")
    [put] = [kw for method, _, kw in mirror.calls if method == "PUT"]
    assert put["params"]["valueInputOption"] == "USER_ENTERED"


async def test_publishing_again_leaves_one_set_of_rules_not_two() -> None:
    mirror = FakeMirror({"Home": HOME, "Qabullar": BOOK}, rules=48)

    await SheetPages(mirror).publish(2026)  # type: ignore[arg-type]

    requests = mirror.requests()
    assert sum("deleteConditionalFormatRule" in r for r in requests) == 48
    assert any("unmergeCells" in r for r in requests)
    last_delete = max(i for i, r in enumerate(requests) if "deleteConditionalFormatRule" in r)
    first_add = next(i for i, r in enumerate(requests) if "addConditionalFormatRule" in r)
    assert last_delete < first_add


async def test_a_missing_home_is_created_and_a_comma_locale_keeps_commas() -> None:
    mirror = FakeMirror({"Qabullar": BOOK}, locale="en_US")

    gid = await SheetPages(mirror).publish(2026)  # type: ignore[arg-type]

    assert mirror.sheets[HOME_SHEET] == gid
    row, column = tile_position(9)
    assert "Qabullar!A:A, 0" in mirror.home()[row][column]


async def test_without_an_appointment_book_there_is_nothing_to_point_at() -> None:
    mirror = FakeMirror({"Home": HOME})
    with pytest.raises(ValueError):
        await SheetPages(mirror).publish(2026)  # type: ignore[arg-type]
    assert not any(method == "PUT" for method, _, _ in mirror.calls)
