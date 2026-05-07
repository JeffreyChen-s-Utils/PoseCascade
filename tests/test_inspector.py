"""Qt smoke tests for :class:`posecascade.ui.inspector.InspectorDock`."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from posecascade.animation.cloth_host import ClothHost
from posecascade.animation.physics_host import PhysicsHost
from posecascade.animation.spring import SpringChain, SpringParams
from posecascade.scene.component import (
    ClothComponent,
    SpringChainComponent,
)
from posecascade.scene.node import Node
from posecascade.scene.transform import Transform
from posecascade.ui.inspector import InspectorDock
from posecascade.utils.math3d import quat_from_axis_angle, vec3


class _StubServices:
    """Minimal duck-typed Services replacement so InspectorDock can look up live objects."""

    def __init__(self, physics_host: PhysicsHost | None = None,
                 cloth_host: ClothHost | None = None) -> None:
        self.physics_host = physics_host
        self.cloth_host = cloth_host


def test_inspector_empty_state(qapp: Any) -> None:
    del qapp
    dock = InspectorDock(services=_StubServices())
    # Initial: no node bound.
    assert dock._node is None  # noqa: SLF001 — direct access for white-box test


def test_inspector_set_node_populates_transform_fields(qapp: Any) -> None:
    del qapp
    dock = InspectorDock(services=_StubServices())
    node = Node(
        name="probe",
        transform=Transform(
            translation=vec3(1.0, 2.0, 3.0),
            rotation=quat_from_axis_angle(vec3(0.0, 1.0, 0.0), math.pi / 4),
            scale=vec3(2.0, 2.0, 2.0),
        ),
    )
    dock.set_node(node)
    assert math.isclose(dock._t_x.value(), 1.0, abs_tol=1.0e-4)  # noqa: SLF001
    assert math.isclose(dock._t_y.value(), 2.0, abs_tol=1.0e-4)  # noqa: SLF001
    assert math.isclose(dock._t_z.value(), 3.0, abs_tol=1.0e-4)  # noqa: SLF001
    assert math.isclose(dock._s_x.value(), 2.0, abs_tol=1.0e-4)  # noqa: SLF001


def test_inspector_translation_edit_writes_back_to_node(qapp: Any) -> None:
    del qapp
    dock = InspectorDock(services=_StubServices())
    node = Node(name="probe", transform=Transform())
    dock.set_node(node)
    dock._t_x.setValue(5.0)  # noqa: SLF001
    dock._t_y.setValue(-3.0)  # noqa: SLF001
    dock._t_z.setValue(2.5)  # noqa: SLF001
    np.testing.assert_allclose(node.transform.translation, [5.0, -3.0, 2.5], atol=1.0e-5)


def test_inspector_clear_resets_label_and_disables_editors(qapp: Any) -> None:
    del qapp
    dock = InspectorDock(services=_StubServices())
    dock.set_node(Node(name="probe", transform=Transform()))
    dock.set_node(None)
    assert "no selection" in dock._name_label.text().lower()  # noqa: SLF001
    assert not dock._t_x.isEnabled()  # noqa: SLF001


def test_inspector_components_list_populated(qapp: Any) -> None:
    del qapp
    dock = InspectorDock(services=_StubServices())
    node = Node(name="anchor")
    node.add_component(SpringChainComponent(chain_name="hair_C", joints=()))
    node.add_component(ClothComponent(cloth_name="cape", mesh_index=0))
    dock.set_node(node)
    items = [dock._components_list.item(i).text()  # noqa: SLF001
             for i in range(dock._components_list.count())]  # noqa: SLF001
    assert "SpringChainComponent" in items
    assert "ClothComponent" in items


def test_inspector_spring_chain_editor_updates_live_chain(qapp: Any) -> None:
    del qapp
    # Build a real PhysicsHost with a registered chain so the inspector can find it live.
    anchor = Node(name="head_anchor")
    parent = anchor
    joints = []
    for i in range(3):
        joint = Node(name=f"hair_C_{i}", transform=Transform(translation=vec3(0.1, 0.0, 0.0)))
        parent.add_child(joint)
        joints.append(joint)
        parent = joint
    chain = SpringChain.from_node_chain(
        "hair_C", anchor, joints, params=SpringParams(stiffness=10.0, damping=2.0),
    )
    physics = PhysicsHost()
    physics.simulator.add_chain(chain)

    component = SpringChainComponent(
        chain_name="hair_C",
        joints=tuple(joints),
        stiffness=10.0,
        damping=2.0,
        inertia=1.0,
    )
    anchor.add_component(component)

    dock = InspectorDock(services=_StubServices(physics_host=physics))
    dock.set_node(anchor)

    # Find the spinbox for "Stiffness" and bump it. (The editor wires lambdas
    # into spinbox.valueChanged, so setValue triggers the live update.)
    from PySide6.QtWidgets import QDoubleSpinBox  # noqa: PLC0415

    spinboxes = dock._params_container.findChildren(QDoubleSpinBox)  # noqa: SLF001
    assert spinboxes, "no parameter spinboxes built for SpringChainComponent"
    spinboxes[0].setValue(50.0)  # first param row: Stiffness

    assert math.isclose(chain.stiffness, 50.0, abs_tol=1.0e-4)
    assert math.isclose(component.stiffness, 50.0, abs_tol=1.0e-4)


def test_inspector_cloth_editor_updates_live_piece(qapp: Any) -> None:
    del qapp
    from posecascade.animation.cloth import anchor_by_top_axis, cloth_from_mesh  # noqa: PLC0415
    from posecascade.utils.math3d import mat4_identity  # noqa: PLC0415

    positions = np.zeros((9, 3), dtype=np.float32)
    for r in range(3):
        for c in range(3):
            positions[r * 3 + c] = [c * 0.1, -r * 0.1, 0.0]
    indices = np.array([
        0, 3, 4, 0, 4, 1,
        1, 4, 5, 1, 5, 2,
        3, 6, 7, 3, 7, 4,
        4, 7, 8, 4, 8, 5,
    ], dtype=np.uint32)
    piece = cloth_from_mesh(
        "cape",
        positions, indices,
        world_matrix=mat4_identity(),
        anchor_mask=anchor_by_top_axis(positions, fraction=0.30),
    )
    cloth_host = ClothHost()
    cloth_host.solver.add_piece(piece)

    holder = Node(name="cape_holder")
    holder.add_component(ClothComponent(cloth_name="cape", mesh_index=0))

    dock = InspectorDock(services=_StubServices(cloth_host=cloth_host))
    dock.set_node(holder)

    from PySide6.QtWidgets import QDoubleSpinBox  # noqa: PLC0415

    spinboxes = dock._params_container.findChildren(QDoubleSpinBox)  # noqa: SLF001
    # First spinbox is Stiffness. Bump it.
    spinboxes[0].setValue(0.5)
    assert math.isclose(piece.params.structural_stiffness, 0.5, abs_tol=1.0e-4)
