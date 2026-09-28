"""Home in the clinic's spreadsheet: the year, the months, and each month's
bookings further down the same page, with an ✕ back to the top.

The formulas were worked out against the live spreadsheet (a ru_RU one,
where "," is #ERROR!); these pin down what was learned there.
"""

from typing import Any

import pytest

from app.services.sheet_pages import (
    CLOSE,
    EMPTY_MONTH,
    FIRST_SECTION_ROW,
    HOME_SHEET,
    MONTHS_UZ,
    SECTION_ROWS,
    TABLE_HEADER,
    SheetPages,
    home_requests,
    home_values,
    localise,
    month_formula,
    month_title,
    rows_needed,
    section_rows,
    separator_for,
    tile_position,
)
from app.services.sheets import STATUS_CHOICES

HOME = 1412175323

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


def test_the_month_formula_bounds_the_month_and_rolls_december_into_january() -> None:
    september = month_formula(2026, 9)
    assert "DATE(2026, 9, 1)" in september and "DATE(2026, 10, 1)" in september
    assert "DATE(2026, 13, 1)" in month_formula(2026, 12)


def test_the_month_formula_reads_text_dates_hides_exact_copies_and_says_when_empty() -> None:
    formula = month_formula(2026, 9)
    # "2026-09-26" typed as text is on the live sheet.
    assert "DATEVALUE(Qabullar!C2:C)" in formula
    # LET does not spread IF over a range by itself; without this it was #ERROR!.
    assert formula.count("ARRAYFORMULA(") == 1
    assert "UNIQUE(FILTER(" in formula
    assert f'"{EMPTY_MONTH}"' in formula
    assert "1, TRUE, 2, TRUE" in formula


def test_the_table_fills_eight_columns_under_the_tiles() -> None:
    """When, gap, patient, gap, phone, gap, status, comment: B to I, so the
    table lines up with the four tile columns."""
    formula = month_formula(2026, 9)
    table = formula[formula.index("IFERROR(HSTACK(") :]
    assert table.count("gap") == 3
    assert table.count("CHOOSECOLS(s,") == 6
    assert sorted(TABLE_HEADER) == [1, 3, 5, 7, 8]


def test_quotes_and_brackets_are_balanced_in_every_formula() -> None:
    for formula in (v for v in home_values(2026, [9, 10], HOME).values() if v.startswith("=")):
        assert formula.count('"') % 2 == 0, formula
        assert formula.count("(") == formula.count(")"), formula


# --- where things are -------------------------------------------------------------------


def test_twelve_tiles_four_to_a_row_never_overlapping() -> None:
    positions = [tile_position(month) for month in range(1, 13)]
    taken = {(row + dy, column) for row, column in positions for dy in (0, 1)}
    assert len(taken) == 24
    assert tile_position(1) == (6, 1) and tile_position(12) == (12, 7)


def test_sections_start_below_the_tiles_in_calendar_order_and_never_overlap() -> None:
    assert section_rows([10, 9, 9]) == {
        9: FIRST_SECTION_ROW,
        10: FIRST_SECTION_ROW + SECTION_ROWS,
    }
    assert tile_position(12)[0] + 10 < FIRST_SECTION_ROW
    assert rows_needed([9]) == FIRST_SECTION_ROW + SECTION_ROWS
    assert rows_needed([1, 2, 3]) == FIRST_SECTION_ROW + 3 * SECTION_ROWS


# --- the links ----------------------------------------------------------------------------


def test_a_tile_jumps_down_to_its_section_on_the_same_page() -> None:
    cells = home_values(2026, [9], HOME)
    row, column = tile_position(9)
    top = f"A{FIRST_SECTION_ROW + 1}"
    assert cells[(row, column)] == f'=HYPERLINK("#gid={HOME}&range={top}", "Sentabr")'
    assert cells[(row + 1, column)] == (
        f'=HYPERLINK("#gid={HOME}&range={top}", B{FIRST_SECTION_ROW + 4} & " ta qabul  →")'
    )


def test_the_close_button_at_the_top_of_a_section_goes_back_to_the_top() -> None:
    cells = home_values(2026, [9], HOME)
    assert cells[(FIRST_SECTION_ROW, 7)] == f'=HYPERLINK("#gid={HOME}&range=A1", "{CLOSE}")'
    assert cells[(FIRST_SECTION_ROW, 1)] == "Sentabr 2026 — qabullar"
    assert "✕" in CLOSE


def test_only_months_with_a_section_link_anywhere() -> None:
    cells = home_values(2026, [9], HOME)
    for month in (1, 8, 10, 12):
        r, c = tile_position(month)
        assert cells[(r, c)] == MONTHS_UZ[month - 1]
        assert cells[(r + 1, c)] == "Tez orada"
    # Two on the tile, one ✕.
    assert sum("HYPERLINK" in value for value in cells.values()) == 3


def test_the_numbers_count_the_status_column_of_their_own_section() -> None:
    cells = home_values(2026, [9, 10], HOME)
    for top in section_rows([9, 10]).values():
        statuses = f"H{top + 7}:H{top + SECTION_ROWS}"
        assert cells[(top + 3, 1)] == f"=COUNTA({statuses})"
        assert cells[(top + 3, 3)] == f'=COUNTIF({statuses}, "Kutilmoqda")'
        assert cells[(top + 3, 7)] == f'=COUNTIF({statuses}, "Bekor qilindi")'
        assert cells[(top + 6, 1)].startswith("=LET(")


# --- the look -----------------------------------------------------------------------------


def _rules(requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r["addConditionalFormatRule"]["rule"] for r in requests if "addConditionalFormatRule" in r
    ]


def test_every_status_is_coloured_and_no_rule_carries_a_separator() -> None:
    rules = _rules(home_requests(HOME, 2026, [9], current=9))
    colours = [
        rule["booleanRule"]["condition"]["values"][0]["userEnteredValue"]
        for rule in rules
        if rule["booleanRule"]["condition"]["type"] == "TEXT_EQ"
    ]
    assert sorted(colours) == sorted(STATUS_CHOICES)
    [stripe] = [r for r in rules if r["booleanRule"]["condition"]["type"] == "CUSTOM_FORMULA"]
    formula = stripe["booleanRule"]["condition"]["values"][0]["userEnteredValue"]
    assert "," not in formula and ";" not in formula


def test_the_current_month_is_solid_teal_and_the_rest_quiet() -> None:
    requests = home_requests(HOME, 2026, [8, 9], current=9)

    def fill_at(month: int) -> dict[str, float]:
        row, column = tile_position(month)
        for request in requests:
            cell = request.get("repeatCell")
            if (
                cell
                and cell["range"]["startRowIndex"] == row
                and cell["range"]["startColumnIndex"] == column
            ):
                return dict(cell["cell"]["userEnteredFormat"]["backgroundColorStyle"]["rgbColor"])
        raise AssertionError(month)

    teal, white, grey = fill_at(9), fill_at(8), fill_at(1)
    assert teal != white and white != grey and grey != teal


# --- publishing, against a fake Sheets API ------------------------------------------------


class FakeMirror:
    """Answers the handful of calls SheetPages makes, and remembers them."""

    def __init__(
        self,
        sheets: dict[str, int],
        locale: str = "ru_RU",
        rules: int = 0,
        rows: int = 1000,
    ) -> None:
        self.sheets = dict(sheets)
        self.locale = locale
        self.rules = rules
        self.rows = rows
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
                            "gridProperties": {"rowCount": self.rows},
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


async def test_publishing_removes_the_month_tab_and_builds_everything_on_home() -> None:
    mirror = FakeMirror({"Home": HOME, "Sentabr 2026": 77, "Lidlar": 0, "Qabullar": 2})

    gid = await SheetPages(mirror).publish(2026, [9], current=9)  # type: ignore[arg-type]

    assert gid == HOME and mirror.sheets["Home"] == HOME
    assert "Sentabr 2026" not in mirror.sheets
    assert {"Lidlar", "Qabullar"} <= set(mirror.sheets)
    home = mirror.home()
    row, column = tile_position(9)
    assert home[row][column] == f'=HYPERLINK("#gid={HOME}&range=A41"; "Sentabr")'
    assert home[FIRST_SECTION_ROW][7] == f'=HYPERLINK("#gid={HOME}&range=A1"; "{CLOSE}")'
    assert "DATE(2026; 9; 1)" in home[FIRST_SECTION_ROW + 6][1]
    [put] = [kw for method, _, kw in mirror.calls if method == "PUT"]
    assert put["params"]["valueInputOption"] == "USER_ENTERED"


async def test_publishing_again_leaves_one_set_of_rules_not_two() -> None:
    """Home is cleared in place, colour rules included, before it is redrawn."""
    mirror = FakeMirror({"Home": HOME, "Qabullar": 2}, rules=5)

    await SheetPages(mirror).publish(2026, [9])  # type: ignore[arg-type]

    requests = mirror.requests()
    deletes = [r for r in requests if "deleteConditionalFormatRule" in r]
    assert len(deletes) == 5
    assert any("unmergeCells" in r for r in requests)
    clear = next(r for r in requests if "updateCells" in r)
    assert clear["updateCells"]["fields"] == "userEnteredValue,userEnteredFormat"
    first_add = next(i for i, r in enumerate(requests) if "addConditionalFormatRule" in r)
    assert max(i for i, r in enumerate(requests) if "deleteConditionalFormatRule" in r) < first_add


async def test_home_grows_when_there_are_more_months_than_rows() -> None:
    mirror = FakeMirror({"Home": HOME, "Qabullar": 2}, rows=1000)

    await SheetPages(mirror).publish(2026, [8, 9, 10])  # type: ignore[arg-type]

    [grow] = [r["appendDimension"] for r in mirror.requests() if "appendDimension" in r]
    assert grow["length"] == rows_needed([8, 9, 10]) - 1000


async def test_home_that_is_big_enough_is_not_grown() -> None:
    mirror = FakeMirror({"Home": HOME, "Qabullar": 2}, rows=5000)

    await SheetPages(mirror).publish(2026, [9])  # type: ignore[arg-type]

    assert not any("appendDimension" in r for r in mirror.requests())


async def test_a_missing_home_is_created_and_a_comma_locale_keeps_commas() -> None:
    mirror = FakeMirror({"Qabullar": 2}, locale="en_US")

    gid = await SheetPages(mirror).publish(2026, [9])  # type: ignore[arg-type]

    assert mirror.sheets[HOME_SHEET] == gid
    assert ", " in mirror.home()[FIRST_SECTION_ROW + 6][1]


@pytest.mark.parametrize("month", [0, 13, -1])
async def test_there_is_no_thirteenth_month(month: int) -> None:
    mirror = FakeMirror({"Home": HOME})
    with pytest.raises(ValueError):
        await SheetPages(mirror).publish(2026, [month])  # type: ignore[arg-type]
    assert mirror.calls == []


def test_month_titles_are_uzbek() -> None:
    assert month_title(2026, 9) == "Sentabr 2026"
    assert MONTHS_UZ[0] == "Yanvar" and len(MONTHS_UZ) == 12
