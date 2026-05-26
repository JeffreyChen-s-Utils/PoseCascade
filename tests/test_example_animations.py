"""Parse + schema validation for every bundled declarative animation.

Runs every ``.json`` under ``examples/scripts/`` through both the
JSON-Schema validator and the runtime parser. Catches the easy class
of breakage where a schema update or parser tightening makes a
bundled example stop loading.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from posecascade.scripting.declarative import parse_animation, resolve_extends

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "scripts"
_SCHEMA_PATH = Path(__file__).resolve().parent.parent / "schemas" / "animation_v1.json"


def _bundled_animations() -> list[Path]:
    """Every standalone ``.json`` file under ``examples/scripts/``.

    Files whose name starts with ``_`` are treated as inheritable
    profiles (loaded only via ``extends`` from a sibling JSON) and
    skipped — they intentionally omit fields like ``phases`` that
    every standalone animation must provide. Files ending in
    ``.drape.json`` are baked pose-drape snapshots, not animation
    scripts, and are also skipped.
    """
    if not _EXAMPLES_DIR.is_dir():
        return []
    return sorted(
        p for p in _EXAMPLES_DIR.glob("*.json")
        if not p.name.startswith("_") and not p.name.endswith(".drape.json")
    )


@pytest.fixture(scope="module")
def schema() -> dict:
    """Load the v1 schema once for all parametrised tests."""
    if not _SCHEMA_PATH.is_file():
        pytest.skip(f"schema missing at {_SCHEMA_PATH}")
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", _bundled_animations(), ids=lambda p: p.name)
def test_bundled_animation_passes_schema(path: Path, schema: dict) -> None:
    """Each bundled animation conforms to ``schemas/animation_v1.json``."""
    jsonschema = pytest.importorskip("jsonschema")
    doc = json.loads(path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(doc), key=lambda e: e.absolute_path)
    if errors:
        joined = "\n".join(
            f"  {'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}"
            for e in errors
        )
        pytest.fail(f"{path.name} fails schema:\n{joined}")


@pytest.mark.parametrize("path", _bundled_animations(), ids=lambda p: p.name)
def test_bundled_animation_parses_through_runtime(path: Path) -> None:
    """Each bundled animation parses without raising :class:`DeclarativeAnimationError`.

    Goes through :func:`resolve_extends` first so animations declaring an
    ``extends`` profile see the merged document, matching how
    ``load_animation`` drives them in production.
    """
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc = resolve_extends(doc, path.parent)
    parsed = parse_animation(doc)
    assert parsed.name, f"{path.name}: parser yielded an empty animation name"
    assert parsed.phases, f"{path.name}: parser yielded zero phases"


def test_showcase_reel_runs_the_canonical_demo_sequence() -> None:
    """``showcase.json`` ships the canonical 3D-model demo reel:
    idle → turntable → walk → wave → V-pose → hip-pop → bow → return."""
    path = _EXAMPLES_DIR / "showcase.json"
    if not path.is_file():
        pytest.skip("showcase.json not bundled in this checkout")
    doc = resolve_extends(json.loads(path.read_text(encoding="utf-8")), path.parent)
    parsed = parse_animation(doc)
    phase_names = [phase.name for phase in parsed.phases]
    # Demo reel must visit the canonical landmarks. Order matters — the
    # turntable comes before the walk so the viewer sees every angle
    # before any locomotion; the V-pose finale comes after the wave so
    # the reel has a clear "high-energy moment".
    expected_order = (
        "intro_idle", "turntable", "walk_forward_and_back", "wave_hello",
        "v_pose_finale", "hip_pop_stance", "bow", "return_to_neutral",
    )
    assert tuple(phase_names) == expected_order


def test_idle_loop_is_minimal_single_phase() -> None:
    """``idle.json`` is the simplest possible breathing-idle demo."""
    path = _EXAMPLES_DIR / "idle.json"
    if not path.is_file():
        pytest.skip("idle.json not bundled in this checkout")
    doc = resolve_extends(json.loads(path.read_text(encoding="utf-8")), path.parent)
    parsed = parse_animation(doc)
    assert len(parsed.phases) == 1
    assert parsed.phases[0].name == "breathe"
    assert parsed.loop_sec == 4.0
