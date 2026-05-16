"""Qt smoke tests for the in-editor JSON animation dock (MVP-1)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _good_doc() -> dict:
    return {
        "schema_version": 1,
        "name": "smoke",
        "loop_sec": 2.0,
        "phases": [{"name": "still", "duration_sec": 2.0, "body": {"yaw_rad": 0.0}}],
    }


@pytest.fixture
def dock(qapp: object):
    """Create the dock against a real QApplication."""
    from posecascade.ui.animation_json_dock import AnimationJsonDock  # noqa: PLC0415

    return AnimationJsonDock()


def test_load_file_populates_editor(dock, tmp_path: Path) -> None:
    """Opening a JSON file fills the editor and shows OK status."""
    path = tmp_path / "anim.json"
    path.write_text(json.dumps(_good_doc()), encoding="utf-8")
    assert dock.load_file(path) is True
    assert "schema_version" in dock.current_text()
    # Live validation runs on load; OK means the parser was happy.
    dock._validate_now()  # noqa: SLF001 — direct invocation skips the debounce
    assert "OK" in dock._status.text()  # noqa: SLF001


def test_validation_flags_json_syntax_error(dock) -> None:
    """A malformed JSON body surfaces a parse-error message."""
    dock._editor.setPlainText("{not json")  # noqa: SLF001
    dock._validate_now()  # noqa: SLF001
    assert "JSON parse error" in dock._status.text()  # noqa: SLF001


def test_validation_flags_parser_error_on_missing_phases(dock) -> None:
    """Schema-shaped but parser-rejected docs surface the parser message."""
    dock._editor.setPlainText('{"schema_version": 1}')  # noqa: SLF001
    dock._validate_now()  # noqa: SLF001
    assert "phases" in dock._status.text()  # noqa: SLF001


def test_save_writes_text_back_to_disk(dock, tmp_path: Path) -> None:
    """Save persists the buffer to the originally-loaded file."""
    path = tmp_path / "anim.json"
    path.write_text(json.dumps(_good_doc()), encoding="utf-8")
    dock.load_file(path)
    edited_doc = {
        "schema_version": 1, "name": "edited",
        "phases": [{"name": "p", "duration_sec": 1.0}],
    }
    dock._editor.setPlainText(json.dumps(edited_doc))  # noqa: SLF001
    dock._on_save_clicked()  # noqa: SLF001
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["name"] == "edited"


def test_reload_blocked_on_invalid_buffer(dock) -> None:
    """An invalid buffer must not emit reload_requested — the runtime would crash."""
    received: list[str] = []
    dock.reload_requested.connect(received.append)
    dock._editor.setPlainText("{not json")  # noqa: SLF001
    dock._on_reload_clicked()  # noqa: SLF001
    assert received == []
    assert "blocked" in dock._status.text().lower()  # noqa: SLF001


def test_reload_emits_signal_when_valid(dock) -> None:
    """A clean buffer emits the reload signal with the current text."""
    received: list[str] = []
    dock.reload_requested.connect(received.append)
    dock._editor.setPlainText(json.dumps(_good_doc()))  # noqa: SLF001
    dock._on_reload_clicked()  # noqa: SLF001
    assert len(received) == 1
    assert "schema_version" in received[0]
