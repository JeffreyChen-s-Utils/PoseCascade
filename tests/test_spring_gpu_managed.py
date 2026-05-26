"""End-to-end test for GPU-managed spring chains: the simulator skips
them, the chain stays in place, and clearing the flag resumes CPU
integration."""
from __future__ import annotations

import numpy as np

from posecascade.animation.spring import (
    Gravity,
    SpringChain,
    SpringParams,
    SpringSimulator,
)
from posecascade.scene.node import Node
from posecascade.utils.math3d import vec3


def _chain() -> SpringChain:
    """Build a tiny 3-joint chain anchored at the origin.

    Bones point along +X so gravity (-Y) produces a perpendicular force
    that swings the chain — gives a clean CPU-vs-static signal in the
    skip-mode test.
    """
    anchor = Node(name="anchor")
    j0 = Node(name="j0")
    j1 = Node(name="j1")
    j2 = Node(name="j2")
    j0.transform.set_translation(vec3(0, 0, 0))
    j1.transform.set_translation(vec3(0.1, 0, 0))
    j2.transform.set_translation(vec3(0.1, 0, 0))
    anchor.add_child(j0)
    j0.add_child(j1)
    j1.add_child(j2)
    return SpringChain.from_node_chain(
        "hair_C", anchor, [j0, j1, j2],
        params=SpringParams(stiffness=10.0, damping=1.5, inertia=1.0),
    )


def test_gpu_managed_chain_is_skipped_by_cpu_simulator() -> None:
    sim = SpringSimulator()
    sim.add_force(Gravity(force=vec3(0.0, -9.8, 0.0)))
    chain = _chain()
    sim.add_chain(chain)

    # Snapshot initial rotation before any integration.
    j0 = chain.joints[0].node
    initial_rotation = j0.transform.rotation.copy()

    # First baseline: WITHOUT gpu_managed, gravity rotates the joint.
    sim.step(0.5)  # half a second — plenty of time for visible motion
    cpu_rotation = j0.transform.rotation.copy()
    assert not np.allclose(cpu_rotation, initial_rotation, atol=1e-3), (
        "Expected CPU spring solver to rotate the joint under gravity"
    )

    # Reset chain by re-creating and marking it GPU-managed BEFORE step.
    sim2 = SpringSimulator()
    sim2.add_force(Gravity(force=vec3(0.0, -9.8, 0.0)))
    chain2 = _chain()
    sim2.add_chain(chain2)
    sim2.mark_gpu_managed(chain2)
    initial2 = chain2.joints[0].node.transform.rotation.copy()
    sim2.step(0.5)
    after = chain2.joints[0].node.transform.rotation.copy()
    assert np.allclose(after, initial2, atol=1e-6), (
        "GPU-managed chain should NOT be touched by the CPU simulator"
    )


def test_mark_gpu_managed_then_clear_resumes_cpu() -> None:
    sim = SpringSimulator()
    sim.add_force(Gravity(force=vec3(0.0, -9.8, 0.0)))
    chain = _chain()
    sim.add_chain(chain)
    sim.mark_gpu_managed(chain)
    initial = chain.joints[0].node.transform.rotation.copy()
    sim.step(0.1)
    assert np.allclose(
        chain.joints[0].node.transform.rotation, initial, atol=1e-6,
    )
    # Flip back to CPU and step again — joint should move now.
    chain.gpu_managed = False
    sim.step(0.5)
    assert not np.allclose(
        chain.joints[0].node.transform.rotation, initial, atol=1e-3,
    )


def test_gpu_managed_chains_list_filtering() -> None:
    sim = SpringSimulator()
    c1 = _chain()
    c2 = _chain()
    sim.add_chain(c1)
    sim.add_chain(c2)
    sim.mark_gpu_managed(c1)
    managed = sim.gpu_managed_chains()
    assert managed == [c1]
