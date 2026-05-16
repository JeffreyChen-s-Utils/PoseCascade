"""Cloth template — point this at a cape / drape mesh on YOUR model.

The bundled ``examples/assets/herta/herta.glb`` does not ship a separate
cape or skirt mesh: the dress is rigged into the body mesh, and running
cloth simulation on it would deform the torso silhouette into a halo
around the waist + expose back-faces that toon-cull turns into "hollow"
patches. So this script no-ops on the bundled character and is left
here as a template for users who load a model that *does* contain a
cape.

To use it: copy this file, change :data:`CAPE_NODE_NAME` to the name
of your model's cape / cloak / scarf node, and adjust the tuning
constants below for that mesh's scale + topology. The defaults are
calibrated for a free-hanging drape ~20 cm tall.

Sandbox globals used: ``physics_lite`` (engine PhysicsHost + ClothHost
facade), ``scene``, ``vec3``, ``time``, ``sin``, ``tau``.

Run with::

    py -m posecascade --scene <your-scene.glb> --script examples/scripts/cape_cloth.py
"""

# Set this to the actual cape / cloak / scarf node name in YOUR model.
# When the script runs against a scene that doesn't contain this node
# (e.g. the bundled herta.glb, which has no such mesh) ``start()``
# logs a one-line skip message and bails out — no cloth is added.
CAPE_NODE_NAME = "your_cape_mesh_node_name_here"

# Body proxy capsule running through the torso/legs so the cape doesn't
# pass through the body when it sways inward. World-frame coords; tune
# to your model's scale (these defaults assume a ~22 cm tall character).
BODY_CAPSULE_TOP = (0.0, 0.18, 0.0)
BODY_CAPSULE_BOTTOM = (0.0, 0.02, 0.0)
BODY_CAPSULE_RADIUS = 0.02

# Wind: kept very mild because the steady-state displacement of a PBD
# vertex under a constant wind is roughly ``wind / rest_pull``. With
# ``rest_pull = 60`` below, a wind of 0.10 settles at ~1.6 mm of drift.
# Bump these for a more dramatic billow on a long, light cape.
WIND_BASE_SPEED = 0.10
WIND_GUST_AMPLITUDE = 0.10
WIND_GUST_PERIOD_SEC = 4.5

_state = {"wind": None, "configured": False, "cloth": None}


def start():
    # noqa: F821 — physics_lite, scene, vec3 injected by sandbox
    cape_node = scene.find(CAPE_NODE_NAME)  # noqa: F821
    if cape_node is None:
        # No matching mesh in this scene — skip silently. The other
        # systems on the scene (hair springs, body driver, etc.)
        # continue to work; this script just contributes nothing.
        return

    cloth = physics_lite.add_cloth(  # noqa: F821
        cape_node,
        cloth_name="cape",
        anchor_axis=1,
        # Pin roughly the top fifth of the cape so only the lower drape
        # is free to swing. Smaller fractions leave a wide free ring
        # right at the seam, which on a tight costume reads as "the
        # waistband ballooned out".
        anchor_fraction=0.20,
        structural_stiffness=1.0,
        bend_stiffness=0.6,
        # ~15 % velocity loss per tick so oscillations decay fast.
        linear_damping=0.85,
        # 20 PBD iterations resolves the constraint network closer to
        # rest pose every step → less accumulated drift over seconds.
        iterations=20,
        # Strong drape memory — the actual lever that keeps the cape
        # near its artist-authored pose. ``rest_pull = 60`` settles a
        # vertex under a 0.2-magnitude force at ~3 mm of drift.
        rest_pull=60.0,
    )
    _state["cloth"] = cloth

    physics_lite.add_capsule_collider(  # noqa: F821
        a=vec3(*BODY_CAPSULE_TOP),       # noqa: F821
        b=vec3(*BODY_CAPSULE_BOTTOM),    # noqa: F821
        radius=BODY_CAPSULE_RADIUS,
    )

    # Very light gravity: just enough that the cape's free hem reads as
    # weighted, not so much that it overpowers ``rest_pull``.
    physics_lite.set_cloth_gravity(vec3(0.0, -0.1, 0.0))  # noqa: F821

    _state["wind"] = physics_lite.add_cloth_wind(  # noqa: F821
        direction=vec3(1.0, 0.0, 0.4),  # noqa: F821 — slight diagonal so cape both billows + twists
        speed=WIND_BASE_SPEED,
        turbulence_amplitude=0.05,
        turbulence_frequency_hz=1.2,
    )
    _state["configured"] = True


def update(dt):  # noqa: ARG001 — dt unused; gust is driven by total elapsed
    if not _state["configured"]:
        return
    wind = _state["wind"]
    if wind is None:
        return
    elapsed = time()  # noqa: F821
    gust = WIND_GUST_AMPLITUDE * sin(elapsed * tau / WIND_GUST_PERIOD_SEC)  # noqa: F821
    wind.speed = WIND_BASE_SPEED + gust


def on_event(name, payload):  # noqa: ARG001 — payload unused
    """``cape_reset`` returns the cape to its rest pose (useful when debugging)."""
    if name != "cape_reset":
        return
    handle = _state["cloth"]
    if handle is not None:
        handle.reset()
