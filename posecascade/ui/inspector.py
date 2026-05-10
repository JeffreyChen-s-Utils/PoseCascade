"""Inspector — right dock that edits the selected node's transform + key components.

Shows the node's name, three editable Vec3 rows (translation / Euler-rotation /
scale), the list of attached components, and inline tuners for the engine's
two main behaviour components: :class:`SpringChainComponent` and
:class:`ClothComponent`. Edits write straight back to the live node /
component, so the renderer (driven by the version counter on Transform) and
the physics hosts (which read the live params each substep) pick them up on
the next frame.

Rotation is presented as Tait-Bryan ZYX Euler degrees; round-tripping through
quaternion may shift the displayed values when the same node is reselected,
which is the standard editor compromise.
"""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDockWidget,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from posecascade.scene.component import (
    ClothComponent,
    SpringChainComponent,
)
from posecascade.scene.node import Node
from posecascade.scripting.api import quat_from_euler
from posecascade.utils.math3d import quat_normalize, quat_to_euler, vec3

_TRANSLATION_RANGE = 1.0e6
_SCALE_MIN = 1.0e-4
_SCALE_MAX = 1.0e4
_ROTATION_RANGE_DEG = 720.0  # display in degrees with sign; 720° lets users wrap past one turn
_DEFAULT_SPIN_DECIMALS = 4
_PARAM_DECIMALS = 3


class InspectorDock(QDockWidget):
    """Right dock that edits the selected node + tunes its physics components.

    Holds an optional reference to the engine ``services`` bundle so the
    component tuners can mutate the LIVE :class:`SpringChain` / :class:`ClothPiece`
    rather than the stale snapshot stored on the component — without this the
    inspector's stiffness/damping sliders would have no observable effect on a
    running simulation.
    """

    def __init__(self, services: object | None = None, parent: object = None) -> None:
        super().__init__("Inspector", parent)  # type: ignore[arg-type]
        self.setObjectName("InspectorDock")
        self._services = services
        self._node: Node | None = None
        self._suppress_signals = False

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setSpacing(6)

        self._name_label = QLabel("(no selection)")
        self._name_label.setStyleSheet("font-weight: bold; padding: 2px;")
        self._layout.addWidget(self._name_label)

        self._t_x, self._t_y, self._t_z = self._add_vec3_row(
            "Translation", -_TRANSLATION_RANGE, _TRANSLATION_RANGE,
        )
        self._r_x, self._r_y, self._r_z = self._add_vec3_row(
            "Rotation (°)", -_ROTATION_RANGE_DEG, _ROTATION_RANGE_DEG,
        )
        self._s_x, self._s_y, self._s_z = self._add_vec3_row(
            "Scale", _SCALE_MIN, _SCALE_MAX,
        )

        components_label = QLabel("Components:")
        components_label.setToolTip(
            "Components attached to this node — physics chains, cloth pieces, "
            "skin references etc. Tuners for each component appear below.",
        )
        self._layout.addWidget(components_label)
        self._components_list = QListWidget()
        self._components_list.setMaximumHeight(120)
        self._components_list.setToolTip(
            "Read-only list of component types on this node.",
        )
        self._layout.addWidget(self._components_list)

        # Container for component-specific tuners. Re-created each time selection
        # changes so the form layout stays in sync with the visible components.
        self._params_container = QWidget()
        self._params_layout = QVBoxLayout(self._params_container)
        self._params_layout.setContentsMargins(0, 0, 0, 0)
        self._params_layout.setSpacing(6)
        self._layout.addWidget(self._params_container)

        self._layout.addStretch()
        scroll.setWidget(container)
        self.setWidget(scroll)

        self._connect_transform_signals()
        self._refresh()

    def set_node(self, node: Node | None) -> None:
        """Bind the inspector to ``node``. Pass ``None`` to clear."""
        self._node = node
        self._refresh()

    def _add_vec3_row(self, label: str, lo: float, hi: float) -> tuple[
        QDoubleSpinBox, QDoubleSpinBox, QDoubleSpinBox,
    ]:
        row = QHBoxLayout()
        row.addWidget(QLabel(label))
        spinboxes = tuple(_make_spinbox(lo, hi) for _ in range(3))
        # Tooltip per axis so hovering each spinbox makes clear which
        # axis it edits — important because the row's "Translation"
        # label only appears at the start, easily off-screen on
        # narrow docks.
        axis_hints = (
            f"{label} — X axis", f"{label} — Y axis", f"{label} — Z axis",
        )
        for sb, hint in zip(spinboxes, axis_hints, strict=True):
            sb.setToolTip(hint)
            row.addWidget(sb)
        wrapper = QWidget()
        wrapper.setLayout(row)
        self._layout.addWidget(wrapper)
        return spinboxes  # type: ignore[return-value]

    def _connect_transform_signals(self) -> None:
        for sb in self._all_spinboxes():
            sb.valueChanged.connect(self._on_transform_edited)

    def _all_spinboxes(self) -> tuple[QDoubleSpinBox, ...]:
        return (
            self._t_x, self._t_y, self._t_z,
            self._r_x, self._r_y, self._r_z,
            self._s_x, self._s_y, self._s_z,
        )

    def _refresh(self) -> None:
        self._suppress_signals = True
        try:
            if self._node is None:
                self._populate_empty()
                return
            self._populate_from_node(self._node)
        finally:
            self._suppress_signals = False

    def _populate_empty(self) -> None:
        self._name_label.setText("(no selection)")
        for sb in self._all_spinboxes():
            sb.setEnabled(False)
            sb.setValue(0.0)
        self._components_list.clear()
        self._clear_params_container()

    def _populate_from_node(self, node: Node) -> None:
        self._name_label.setText(node.name or "<unnamed>")
        for sb in self._all_spinboxes():
            sb.setEnabled(True)
        t = node.transform.translation
        self._t_x.setValue(float(t[0]))
        self._t_y.setValue(float(t[1]))
        self._t_z.setValue(float(t[2]))
        yaw, pitch, roll = quat_to_euler(node.transform.rotation)
        self._r_x.setValue(math.degrees(roll))
        self._r_y.setValue(math.degrees(pitch))
        self._r_z.setValue(math.degrees(yaw))
        s = node.transform.scale
        self._s_x.setValue(float(s[0]))
        self._s_y.setValue(float(s[1]))
        self._s_z.setValue(float(s[2]))

        self._components_list.clear()
        for comp in node.components:
            self._components_list.addItem(type(comp).__name__)

        self._clear_params_container()
        for comp in node.components:
            editor = self._build_component_editor(comp)
            if editor is not None:
                self._params_layout.addWidget(editor)

    def _on_transform_edited(self) -> None:
        if self._suppress_signals or self._node is None:
            return
        translation = vec3(self._t_x.value(), self._t_y.value(), self._t_z.value())
        roll = math.radians(self._r_x.value())
        pitch = math.radians(self._r_y.value())
        yaw = math.radians(self._r_z.value())
        rotation = quat_normalize(quat_from_euler(yaw, pitch, roll))
        scale = vec3(self._s_x.value(), self._s_y.value(), self._s_z.value())
        self._node.transform.set_translation(translation.astype(np.float32))
        self._node.transform.set_rotation(rotation.astype(np.float32))
        self._node.transform.set_scale(scale.astype(np.float32))

    def _clear_params_container(self) -> None:
        while self._params_layout.count() > 0:
            item = self._params_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _build_component_editor(self, component: object) -> QWidget | None:
        if isinstance(component, SpringChainComponent):
            chain = self._lookup_live_chain(component.chain_name)
            return _spring_chain_editor(component, chain)
        if isinstance(component, ClothComponent):
            piece = self._lookup_live_cloth(component.cloth_name)
            return _cloth_editor(component, piece)
        return None

    def _lookup_live_chain(self, name: str) -> object | None:
        host = getattr(self._services, "physics_host", None)
        if host is None:
            return None
        return host.find_chain(name)

    def _lookup_live_cloth(self, name: str) -> object | None:
        host = getattr(self._services, "cloth_host", None)
        if host is None:
            return None
        return host.find_piece(name)


def _make_spinbox(lo: float, hi: float) -> QDoubleSpinBox:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(_DEFAULT_SPIN_DECIMALS)
    sb.setSingleStep(0.01)
    sb.setKeyboardTracking(False)  # only fire on commit (Enter / focus-out)
    sb.setAlignment(Qt.AlignmentFlag.AlignRight)
    return sb


def _spring_chain_editor(
    component: SpringChainComponent,
    live_chain: object | None,
) -> QGroupBox:
    """Inline form to tune a SpringChain's stiffness/damping/inertia. Edits the live chain
    if registered (so changes take effect immediately); falls back to mutating the
    component for offline scenes."""
    box = QGroupBox(f"SpringChain: {component.chain_name}")
    form = QFormLayout(box)
    form.addRow("Joints", QLabel(str(len(component.joints))))

    def stiffness_setter(value: float) -> None:
        component.stiffness = float(value)
        if live_chain is not None:
            live_chain.stiffness = float(value)  # type: ignore[attr-defined]

    def damping_setter(value: float) -> None:
        component.damping = float(value)
        if live_chain is not None:
            live_chain.damping = float(value)  # type: ignore[attr-defined]

    def inertia_setter(value: float) -> None:
        component.inertia = float(value)
        if live_chain is not None:
            for joint in live_chain.joints:  # type: ignore[attr-defined]
                joint.inertia = float(value)

    _add_param_row(
        form, "Stiffness", component.stiffness, 0.0, 200.0, stiffness_setter,
        tooltip=(
            "Spring constant pulling each joint back to its rest pose. "
            "Higher = stiffer hair / cloth, less sway. Typical 8–20."
        ),
    )
    _add_param_row(
        form, "Damping", component.damping, 0.0, 50.0, damping_setter,
        tooltip=(
            "Velocity damping each frame. Higher = less ringing after an "
            "impulse. Typical 0.3–0.8."
        ),
    )
    _add_param_row(
        form, "Inertia", component.inertia, 1.0e-4, 10.0, inertia_setter,
        tooltip=(
            "Per-joint mass — heavier joints respond more slowly to wind / "
            "head motion. Typical 0.01–0.05 for hair."
        ),
    )
    return box


def _cloth_editor(
    component: ClothComponent,
    live_piece: object | None,
) -> QGroupBox:
    """Inline form to tune cloth PBD params. Mutates the live piece directly when present."""
    box = QGroupBox(f"Cloth: {component.cloth_name}")
    form = QFormLayout(box)
    form.addRow("Mesh index", QLabel(str(component.mesh_index)))

    def make_setter(attr: str) -> Callable[[float], None]:
        def setter(value: float) -> None:
            setattr(component, attr, float(value))
            if live_piece is not None:
                setattr(live_piece.params, attr, float(value))  # type: ignore[attr-defined]
        return setter

    _add_param_row(
        form, "Stiffness", component.structural_stiffness, 0.0, 1.0,
        make_setter("structural_stiffness"),
        tooltip=(
            "How rigidly the cloth resists stretching along its weave (PBD "
            "structural constraint). 0 = rubbery, 1 = rigid. Typical 0.6–0.9."
        ),
    )
    _add_param_row(
        form, "Bend", component.bend_stiffness, 0.0, 1.0,
        make_setter("bend_stiffness"),
        tooltip=(
            "Resistance to folding / wrinkling. 0 = limp, 1 = card-stiff. "
            "Typical 0.1–0.3 for cloth, higher for leather."
        ),
    )
    _add_param_row(
        form, "Damping", component.linear_damping, 0.5, 1.0,
        make_setter("linear_damping"),
        tooltip=(
            "Per-frame velocity multiplier (1 = no damping, 0.5 = aggressive "
            "damp). Lower = faster settle after an impulse."
        ),
    )
    _add_param_row(
        form, "Rest pull", component.rest_pull, 0.0, 200.0,
        make_setter("rest_pull"),
        tooltip=(
            "Force pulling vertices back toward their rest position each "
            "frame. Prevents drift / sag accumulation. Typical 5–40."
        ),
    )
    return box


def _add_param_row(
    form: QFormLayout,
    label: str,
    initial: float,
    lo: float,
    hi: float,
    setter: Callable[[float], None],
    *,
    tooltip: str | None = None,
) -> None:
    sb = QDoubleSpinBox()
    sb.setRange(lo, hi)
    sb.setDecimals(_PARAM_DECIMALS)
    sb.setSingleStep(0.05)
    sb.setKeyboardTracking(False)
    sb.setValue(float(initial))
    sb.valueChanged.connect(setter)
    if tooltip is not None:
        sb.setToolTip(tooltip)
    form.addRow(label, sb)
