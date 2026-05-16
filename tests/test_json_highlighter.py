"""Sanity tests for the JSON syntax highlighter."""
from __future__ import annotations

import pytest


@pytest.fixture
def highlighter(qapp: object):
    from PySide6.QtGui import QTextDocument  # noqa: PLC0415

    from posecascade.ui.json_highlighter import JsonHighlighter  # noqa: PLC0415

    doc = QTextDocument()
    doc.setPlainText('{"name": "test", "count": 3, "ok": true, "z": null}')
    return JsonHighlighter(doc), doc


def test_highlighter_applies_per_line_formats(highlighter) -> None:
    """At least one key, string, number, and literal should pick up a colour."""
    _, doc = highlighter
    block = doc.firstBlock()
    layout = block.layout()
    # ``QTextLayout.FormatRange`` items survive the highlight pass; we
    # just check we got at least four distinct foreground colours
    # painted (key, value-string, number, literal).
    formats = block.layout().formats() if hasattr(layout, "formats") else []
    colours = {fmt.format.foreground().color().name() for fmt in formats}
    assert len(colours) >= _MIN_DISTINCT_COLOURS


_MIN_DISTINCT_COLOURS = 4
