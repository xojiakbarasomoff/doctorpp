from app.core.redaction import preview


def test_preview_returns_short_text_unchanged() -> None:
    assert preview("Hi, do you have an opening tomorrow?", limit=40) == (
        "Hi, do you have an opening tomorrow?"
    )


def test_preview_truncates_long_text_with_ellipsis() -> None:
    text = "x" * 100
    result = preview(text, limit=40)
    assert result == "x" * 40 + "…"
    assert len(result) == 41


def test_preview_default_limit_is_forty() -> None:
    text = "y" * 41
    assert preview(text) == "y" * 40 + "…"
