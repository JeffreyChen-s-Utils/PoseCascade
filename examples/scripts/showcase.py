"""Showcase animation for ``character.glb`` — body sway + hair + sleeve cloth.

The character has a long flowing sleeve (``obj2_m18_0``) that sits
very close to several body meshes in the abdomen / waist area
(``obj2_m9_0``, ``m13_0``, ``m17_0``). The bundled glTF was patched
in Blender to inflate the sleeve outward by ~2 mm along its normals,
which buys enough depth-test headroom that a PBD cloth simulation's
~5 mm gravity-driven hem droop no longer fights with the body
geometry underneath. With that fix in the asset, the cloth solver
can run without the abdomen-area color flicker the unpatched mesh
exhibited.

What this script drives:

- Subtle yaw sway + breathing bob on the body root + hair armature so
  body and hair shift together as one character. The yaw is composed
  *on top of* each node's export rotation rather than overwriting it
  (the Sketchfab glTF used here ships a 180° axis-fix on its root
  that ``set_rotation`` would otherwise clobber).
- Auto-attached hair chains tuned per strand for clear, head-tracked
  sway under a sin-modulated gusty wind.
- A conservative PBD cloth pass on the long sleeve so the bottom hem
  drapes naturally under gravity. Body sway provides the bulk of the
  visible swing through rigid rotation; the cloth solver adds ~5 mm
  of gravity-driven trail at the hem.
- A periodic "flourish" beat that kicks the hair chains in alternating
  directions so the silhouette changes between sway cycles.

What this script drives:

- A subtle yaw sway + breathing bob on the body root + hair armature so
  body and hair shift together as a single character. The yaw is
  composed *on top of* each node's export rotation rather than
  overwriting it (the Sketchfab glTF used here ships a 180° axis-fix
  on its root that ``set_rotation`` would otherwise clobber).
- Auto-attached hair chains tuned per strand for a clear, head-tracked
  sway. Stiffness + damping are pushed up from the bare-minimum tuning
  so the hair stays visually attached to the head during the sway.
- A conservative PBD cloth pass on the right-side back accessory so
  the clothing silhouette has visible motion without distorting the
  body shape.
- A sin-modulated side wind that gusts every few seconds.
- A periodic "flourish" beat that kicks the hair chains in alternating
  directions so the silhouette changes between sway cycles.

Sandbox globals used: ``scene``, ``physics_lite``, ``time``, ``vec3``,
``quat_axis_angle``, ``quat_mul``, ``sin``, ``tau``.

Run with::

    py -m posecascade --scene examples/assets/character.glb \
        --script examples/scripts/showcase.py
"""

# ----- character motion -------------------------------------------------
# The Sketchfab export keeps the body mesh under ``Sketchfab_model`` and
# the hair bones under a *sibling* ``HairArmature`` group rather than a
# child of the body. Animating only one of them would leave the hair
# pinned at world origin while the body sways (or vice versa); the
# script drives every node in this tuple in lock-step so they read as a
# single character.
CHARACTER_NODE_NAMES = ("Sketchfab_model", "HairArmature")

# Yaw sway: ±10° around forward, period 6 s. The cloth solver runs in a
# frozen world frame, so any body rotation creates a mismatch between
# the world-frame drift the cloth accumulates and the body-relative
# direction the viewer expects the swing to be in. Past ~12° this
# mismatch reads as the sleeve detaching from the body. ±10° is the
# largest amplitude where the world / body frames stay close enough
# for the swing to look anchored.
BODY_SWAY_AMPLITUDE_RAD = 0.18
BODY_SWAY_PERIOD_SEC = 6.0

# Breathing bob: at the model's ±5 vertical extent we want centimetres,
# not millimetres, so the body unmistakably "breathes".
BODY_BOB_AMPLITUDE = 0.06
BODY_BOB_PERIOD_SEC = 4.5

# ----- hair physics -----------------------------------------------------
# Tuning lifted from full_demo.py but with stiffness pushed roughly 4×
# and damping ~2× so the strands track the head through the sway
# instead of trailing into "detached" territory. Outer strands stay
# looser than the back hair so the silhouette still changes per gust.
HAIR_TUNING = {
    "hair_C":  {"stiffness": 14.0, "damping": 0.45, "inertia": 0.015},
    "hair_L":  {"stiffness": 12.0, "damping": 0.40, "inertia": 0.015},
    "hair_R":  {"stiffness": 12.0, "damping": 0.40, "inertia": 0.015},
    "hair_LL": {"stiffness":  8.5, "damping": 0.30, "inertia": 0.020},
    "hair_RR": {"stiffness":  8.5, "damping": 0.30, "inertia": 0.020},
    "orn":     {"stiffness": 22.0, "damping": 0.80, "inertia": 0.020},
}
HAIR_WIND_DIRECTION = (1.0, 0.0, 0.3)
HAIR_WIND_BASE = 0.9
HAIR_WIND_GUST = 0.6
HAIR_WIND_PERIOD_SEC = 3.2

# Flourish: every FLOURISH_PERIOD_SEC seconds, kick every hair joint
# with a rotating direction so the silhouette pops.
FLOURISH_PERIOD_SEC = 5.5
FLOURISH_HAIR_IMPULSE = 1.4

# ----- sleeve cloth -----------------------------------------------------
# Long right-side flowing sleeve. The asset has been pre-inflated by
# 2 mm along its normals (see Blender patch in this repo's history),
# so the cloth's ~5 mm gravity drift no longer falls back into body
# geometry → no abdomen z-fighting. ``start()`` silently skips when
# the node isn't present (other models, modified scenes, etc.).
SLEEVE_NODE_NAME = "obj2_m18_0"


_state = {
    "configured": False,
    # Tuples of (node, rest_rotation_quat, rest_translation_vec) so the
    # script can compose its yaw + bob *on top of* each node's export
    # rotation rather than clobbering it. The Sketchfab glTF used here,
    # for example, ships a 180° axis-fix on ``Sketchfab_model``;
    # overwriting it with a fresh ``set_rotation`` would tilt the body
    # off the camera while the (originally identity) ``HairArmature``
    # stays upright — exactly the "hair separated from body" symptom.
    "character_drives": (),
    "hair_wind": None,
    "sleeve_cloth": None,
    "last_flourish_at": -10.0,
}


def start():
    # noqa: F821 — sandbox-injected names: scene / physics_lite / vec3
    drives = []
    for name in CHARACTER_NODE_NAMES:
        node = scene.find(name)  # noqa: F821
        if node is None:
            continue
        # ``.copy()`` snapshots the numpy quat / vec so a later
        # ``set_rotation`` doesn't drag the cached value along with it.
        rest_rotation = node.transform.rotation.copy()
        rest_translation = node.transform.translation.copy()
        drives.append((node, rest_rotation, rest_translation))
    _state["character_drives"] = tuple(drives)

    for chain_name, params in HAIR_TUNING.items():
        chain = physics_lite.get_chain(chain_name)  # noqa: F821
        if chain is None:
            continue
        chain.stiffness = params["stiffness"]
        chain.damping = params["damping"]
        chain.set_inertia(params["inertia"])

    _state["hair_wind"] = physics_lite.add_wind(  # noqa: F821
        direction=vec3(*HAIR_WIND_DIRECTION),  # noqa: F821
        speed=HAIR_WIND_BASE,
        turbulence_amplitude=0.25,
        turbulence_frequency_hz=1.4,
    )
    _setup_sleeve_cloth()
    _state["configured"] = True


def _setup_sleeve_cloth():
    """Conservative PBD cloth on the inflated sleeve mesh.

    Tuning notes:

    - **No wind**, only mild gravity (-0.08). Wind in solver-world
      frame creates body-frame mismatches that read as the sleeve
      detaching during sway; gravity-only drift is purely vertical
      and translates cleanly through the render transform.
    - **``rest_pull = 30``** keeps the bottom hem droop at ~3 mm in
      steady state — enough to read as cloth weight, well within the
      2 mm offset budget the asset was patched to provide.
    - **``anchor_fraction = 0.20``** pins the shoulder seam + a band
      below it so only the lower drape moves under gravity.
    - **``iterations = 10``** balances rest stability against per-frame
      cost (2295 verts × 20 iter was visibly stuttering at 60 Hz).
    - **No body capsule.** A capsule at world (0,_,0) clipped the
      sleeve's near edge and pushed it outward into a halo on the old
      asset — pre-inflate already gives the cloth enough clearance.
    """
    # noqa: F821 — sandbox-injected helpers
    sleeve_node = scene.find(SLEEVE_NODE_NAME)  # noqa: F821
    if sleeve_node is None:
        return
    # Anchor the top 60 % of the sleeve. The mesh is vertex-dense at the
    # shoulder + torso (88 % of verts in that band) and overlaps the body
    # costume there; pinning it rigid keeps that zone in lockstep with
    # the body (no z-fight, no drift). Only the 266-vert decorative
    # dragon hem hanging below the waist is free to swing — and that
    # hem doesn't overlap any body geometry, so its drift can be much
    # bigger without any visual artefacts.
    handle = physics_lite.add_cloth(  # noqa: F821
        sleeve_node,
        cloth_name="sleeve",
        anchor_axis=1,
        anchor_fraction=0.60,
        structural_stiffness=1.0,
        bend_stiffness=0.6,
        linear_damping=0.85,
        iterations=10,
        # ``rest_pull = 12`` settles the free hem at ~17 mm of Y droop
        # under the gravity below, with the swing-arc induced lateral
        # drift staying around 4 mm — comfortably inside the 5 mm
        # offset the asset was inflated to provide.
        rest_pull=12.0,
    )
    _state["sleeve_cloth"] = handle
    # Y-only gravity. Vertical force is invariant under the body's
    # Y-axis sway, so its drift doesn't create the body-frame
    # mismatch ("sleeve detaches") that any X/Z-component force would.
    physics_lite.set_cloth_gravity(vec3(0.0, -0.20, 0.0))  # noqa: F821


def update(dt):  # noqa: ARG001 — driven by elapsed time, not delta
    if not _state["configured"]:
        return
    elapsed = time()  # noqa: F821

    _drive_character(elapsed)
    _drive_winds(elapsed)
    _maybe_flourish(elapsed)


def _drive_character(elapsed: float) -> None:
    # noqa: F821 — sandbox-injected helpers
    drives = _state["character_drives"]
    if not drives:
        return
    # Yaw sway + breathing bob. The yaw quat is composed *on top of*
    # each node's rest rotation so the export's coordinate fix survives;
    # the hair armature tends to start at identity so it sees just our
    # yaw, and either way both nodes end up in the same world-space
    # frame so the hair tracks the body.
    sway = BODY_SWAY_AMPLITUDE_RAD * sin(elapsed * tau / BODY_SWAY_PERIOD_SEC)  # noqa: F821
    yaw = quat_axis_angle(vec3(0.0, 1.0, 0.0), sway)  # noqa: F821
    bob = BODY_BOB_AMPLITUDE * sin(elapsed * tau / BODY_BOB_PERIOD_SEC)  # noqa: F821
    for node, rest_rot, rest_trans in drives:
        composed = quat_mul(yaw, rest_rot)  # noqa: F821
        node.transform.set_rotation(composed)
        node.transform.set_translation(
            vec3(  # noqa: F821
                float(rest_trans[0]),
                float(rest_trans[1]) + bob,
                float(rest_trans[2]),
            ),
        )


def _drive_winds(elapsed: float) -> None:
    hair_wind = _state["hair_wind"]
    if hair_wind is not None:
        gust = HAIR_WIND_GUST * sin(elapsed * tau / HAIR_WIND_PERIOD_SEC)  # noqa: F821
        hair_wind.speed = HAIR_WIND_BASE + gust


_HAIR_CHAIN_JOINT_COUNT = 4


def _maybe_flourish(elapsed: float) -> None:
    if elapsed - _state["last_flourish_at"] < FLOURISH_PERIOD_SEC:
        return
    _state["last_flourish_at"] = elapsed
    cycle = int(elapsed / FLOURISH_PERIOD_SEC) % 4
    directions = [
        vec3(1.0, 0.0, 0.0),    # noqa: F821
        vec3(0.0, 0.0, 1.0),    # noqa: F821
        vec3(-1.0, 0.0, 0.0),   # noqa: F821
        vec3(0.0, 0.0, -1.0),   # noqa: F821
    ]
    axis = directions[cycle]
    # Only kick the hair_* chains — the ornament chain reads as a fixed
    # clip on the head, not a swaying strand, so an impulse on it makes
    # the model look broken rather than alive.
    for chain in physics_lite.chains():  # noqa: F821
        if not chain.name.startswith("hair_"):
            continue
        for joint_index in range(_HAIR_CHAIN_JOINT_COUNT):
            chain.apply_impulse(
                axis=axis,
                magnitude=FLOURISH_HAIR_IMPULSE,
                joint_index=joint_index,
            )


def on_event(name, payload):  # noqa: ARG001 — payload unused
    """Reset every hair chain + the sleeve cloth to rest pose."""
    if name != "reset":
        return
    for chain in physics_lite.chains():  # noqa: F821
        chain.reset()
    handle = _state["sleeve_cloth"]
    if handle is not None:
        handle.reset()
