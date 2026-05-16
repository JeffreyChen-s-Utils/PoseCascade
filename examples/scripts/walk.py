"""Walking demo — character walks forward 2 m then back.

Run with::

    py -m posecascade --scene examples/assets/herta/herta.glb \\
        --script examples/scripts/walk.py

The imperative twin of :file:`walk.json`. The character translates -2 m
along Z (away from the camera), turns 180 deg, then walks back. Arms
hang naturally at the sides; legs swing in a sine cycle.

The script targets canonical bone names (``upper_arm_L`` etc.); the
importer's auto-rig resolves them to the rig's real bone names via
``scene.bone_aliases`` so the same code drives any humanoid rig. If
the loaded mesh contains a separately-named ``Skirt`` node the cloth
host adds a PBD pass for it; on the bundled character (dress rigged
into the body mesh) the cloth registration silently no-ops.
"""

# ---- timing ------------------------------------------------------------------
LOOP_SEC = 8.0
HALF_LOOP_SEC = 4.0
FORWARD_DISTANCE = 2.0
STEP_CYCLE_SEC = 1.0

PI = 3.141592653589793
TAU = 6.283185307179586

# ---- pose constants ----------------------------------------------------------
# March 7th's bind pose is a 45-degree A-pose. Rotating the arm a full
# 45 degrees brings it vertical against the body — but then the elbow
# sits exactly at hip height where the skirt is widest and clips
# through the cloth. PI/6 (30 deg) leaves the arm 15 deg outward of
# vertical so it hangs OUTSIDE the skirt at the hip.
ARM_Z_HANG = 0.5235987755982988  # PI / 6 — 30 degrees

# With yaw=PI applied to the root, conjugating a body-frame X rotation by
# the yaw flips its sign — so a leg "swung forward in body frame" needs
# the OPPOSITE sign than it would without yaw. Concretely: leg_l set via
# a positive X rotation rotates the leg body-BACKWARD here, which made
# the body translate forward while the legs slid back (the "moonwalk"
# illusion the user kept describing). Negate to swing body-forward.
# Amplitudes tuned for The Herta: her dress is rigged into the body
# mesh so the legs swing inside the skirt envelope without any
# separate-mesh clipping artefacts, letting us run a natural 10° hip
# swing and 17° knee bend without the visible kinks the previous
# (separately-rigged) March 7th demo had at large amplitudes.
LEG_SWING_AMP = 0.17
KNEE_BEND = 0.30

YAW_EPS = 1.0e-4

# Body collider radii tuned for the March 7th skirt: her hem sits at
# ~0.15 m lateral, so the capsule extents (centre + radius) MUST stay
# inside that envelope or the PBD projection explodes the cloth on tick 1.
# Thigh centre is at X=±0.05 m, so radius 0.08 → extent 0.13 m, leaving
# 2 cm clearance to the skirt hem. Hip sphere radius 0.10 keeps the same
# margin. Shin is below the hem so its radius is unconstrained by cloth.
HIP_RADIUS = 0.10
THIGH_RADIUS = 0.08
SHIN_RADIUS = 0.05
COLLIDER_SKIN = 0.005

# Skirt cloth parameters (mirror walk.json).
SKIRT_PARAMS = {
    "cloth_name": "skirt",
    "anchor_axis": 1,
    "anchor_fraction": 0.18,
    "anchor_mode": "top_axis",
    "structural_stiffness": 0.85,
    "bend_stiffness": 0.20,
    "linear_damping": 0.96,
    "iterations": 10,
    # rest_pull MUST be high enough that gravity / rest_pull equilibrium
    # sits ABOVE the floor. With the default rest_pull=4 + cloth gravity
    # ~2.5, equilibrium is 0.625 m below rest — for a skirt with verts
    # starting at Y=0.4 the bottom hem collapses to Y=-0.225 and (with
    # a floor clamp) smears onto the floor. rest_pull=30 puts equilibrium
    # at rest - 0.083 m, well above ground, and visually the skirt drapes
    # naturally without falling apart.
    "rest_pull": 30.0,
    # Drive cloth ``rest_positions`` from the mesh's bone skinning each
    # tick — so hem verts with Hip_L/R_02 weight swing forward when the
    # leg swings forward, while waistband verts stay glued to the root.
    # Without this the skirt sits anchored to Root_M_01 and the legs
    # poke through it at unexpected lateral positions during a gait.
    "skin_target": True,
}

# The rig + mesh subtree on March 7th's glTF. After the knee re-weighting
# pass and re-export, the original ``March_7th`` / ``Sketchfab_model``
# wrapper nodes collapsed and ``Object_4`` (the armature) is now the
# top-level node holding every bone and mesh, so we translate that.
CHARACTER_ROOT = "Object_4"
SKIRT_NODE = "Skirt"

_DRIVEN_BONES = (
    "upper_arm_L", "upper_arm_R",
    "upper_leg_L", "upper_leg_R",
    "lower_leg_L", "lower_leg_R",
)

# (start_alias, end_alias_or_None, radius). end=None means sphere.
_BODY_COLLIDERS = (
    ("hip",         None,          HIP_RADIUS),
    ("upper_leg_L", "lower_leg_L", THIGH_RADIUS),
    ("upper_leg_R", "lower_leg_R", THIGH_RADIUS),
    ("lower_leg_L", "foot_L",      SHIN_RADIUS),
    ("lower_leg_R", "foot_R",      SHIN_RADIUS),
)

_state = {
    "root_node": None,
    "root_rest_rot": None,
    "root_rest_trans": None,
    "bone_drives": {},
    "colliders": (),
    "configured": False,
}


def start():
    root = scene.find(CHARACTER_ROOT)  # noqa: F821
    if root is None:
        return
    _state["root_node"] = root
    _state["root_rest_rot"] = root.transform.rotation.copy()
    _state["root_rest_trans"] = root.transform.translation.copy()

    bone_drives = {}
    for name in _DRIVEN_BONES:
        node = scene.find(name)  # noqa: F821
        if node is None:
            continue
        bone_drives[name] = (node, node.transform.rotation.copy())
    _state["bone_drives"] = bone_drives

    skirt = scene.find(SKIRT_NODE)  # noqa: F821
    if skirt is not None:
        hip_bone = scene.find("hip")  # noqa: F821
        physics_lite.add_cloth(  # noqa: F821
            skirt, track_bone=hip_bone, **SKIRT_PARAMS,
        )
        # Body colliders intentionally disabled. We tried two radius sets
        # (0.11 m and 0.08 m thigh capsules) and both pushed the cloth
        # solver into an exploded state on tick 1: the inverse-bind-matrix
        # cloth initialisation lands some hem verts INSIDE the capsule,
        # and the PBD push-out projection sends them flying. The skirt
        # without colliders sits anchored to Root_M_01 — the leg sub-mesh
        # may visibly poke through during peak swing, which is a known
        # asset-quality limitation of this rig (the body mesh has no
        # geometry in Z=0.55-0.77 so the upper thigh is "missing" anyway).
    _state["configured"] = True


def _install_body_colliders():
    """Add hip sphere + leg capsules; return list keyed for per-frame sync."""
    colliders = []
    for start_alias, end_alias, radius in _BODY_COLLIDERS:
        head = scene.find(start_alias)  # noqa: F821
        if head is None:
            continue
        head_pos = world_position(head)  # noqa: F821
        if end_alias is None:
            sphere = physics_lite.add_sphere_collider(head_pos, radius)  # noqa: F821
            sphere.skin_offset = COLLIDER_SKIN
            colliders.append(("sphere", sphere, head, None))
            continue
        tail = scene.find(end_alias)  # noqa: F821
        if tail is None:
            continue
        cap = physics_lite.add_capsule_collider(  # noqa: F821
            head_pos, world_position(tail), radius,  # noqa: F821
        )
        cap.skin_offset = COLLIDER_SKIN
        colliders.append(("capsule", cap, head, tail))
    return tuple(colliders)


def update(dt):  # noqa: ARG001
    if not _state["configured"]:
        return
    elapsed = time() % LOOP_SEC  # noqa: F821
    phase_t = (elapsed % HALF_LOOP_SEC) / HALF_LOOP_SEC
    if elapsed < HALF_LOOP_SEC:
        # March 7th's bind pose faces +Z. Yaw=PI rotates her 180 deg to
        # face -Z so the -Z translation is walking FORWARD relative to
        # her facing — without this she moonwalks (the "倒著走" bug).
        z = -FORWARD_DISTANCE * phase_t
        yaw = PI
    else:
        # Phase 2: back to bind-pose facing (+Z) and translate +Z toward origin.
        z = -FORWARD_DISTANCE * (1.0 - phase_t)
        yaw = 0.0
    _apply_root(z, yaw)
    _apply_gait(elapsed, yaw)
    _sync_colliders()


def _apply_root(z, yaw):
    root = _state["root_node"]
    yaw_q = quat_axis_angle(vec3(0.0, 1.0, 0.0), yaw)  # noqa: F821
    composed = quat_mul(yaw_q, _state["root_rest_rot"])  # noqa: F821
    root.transform.set_rotation(composed)
    rest = _state["root_rest_trans"]
    root.transform.set_translation(vec3(  # noqa: F821
        float(rest[0]), float(rest[1]), float(rest[2]) + z,
    ))


def _apply_gait(elapsed, yaw):
    bones = _state["bone_drives"]
    if not bones:
        return
    phase = elapsed * TAU / STEP_CYCLE_SEC
    cycle = sin(phase)  # noqa: F821
    cos_phase = cos(phase)  # noqa: F821
    # Body-frame leg swing: negative X rotation = leg goes body-forward.
    leg_l = -LEG_SWING_AMP * cycle
    leg_r = +LEG_SWING_AMP * cycle
    # Body-frame knee bend: positive X rotation = calf goes body-backward
    # (natural anatomical knee bend).
    knee_l = KNEE_BEND * 0.5 * (1.0 - cos_phase)
    knee_r = KNEE_BEND * 0.5 * (1.0 + cos_phase)
    # Swing the A-pose arms from 45-degree-out to straight-down.
    _set_bone("upper_arm_L", _z_quat(-ARM_Z_HANG), yaw)
    _set_bone("upper_arm_R", _z_quat(+ARM_Z_HANG), yaw)
    _set_bone("upper_leg_L", _x_quat(leg_l), yaw)
    _set_bone("upper_leg_R", _x_quat(leg_r), yaw)
    _set_bone("lower_leg_L", _x_quat(knee_l), yaw)
    _set_bone("lower_leg_R", _x_quat(knee_r), yaw)


def _sync_colliders():
    """Snap collider centres to current bone world positions each frame."""
    for kind, collider, head, tail in _state["colliders"]:
        head_pos = world_position(head)  # noqa: F821
        if kind == "sphere":
            collider.center = head_pos
        else:
            collider.a = head_pos
            collider.b = world_position(tail)  # noqa: F821


def _x_quat(angle):
    return quat_axis_angle(vec3(1.0, 0.0, 0.0), angle)  # noqa: F821


def _z_quat(angle):
    return quat_axis_angle(vec3(0.0, 0.0, 1.0), angle)  # noqa: F821


def _yaw_to_world(body_delta, yaw):
    """Conjugate a body-frame rotation by ``yaw`` so signs stay consistent.

    Without this, the same body-frame "forward leg lift" swing flips
    direction when the body rotates 180 deg (walk-back phase) and the
    character ends up high-stepping in reverse. The Y-axis yaw commutes
    with Y-axis bone rotations (head turn, hip yaw) but X- and Z-axis
    deltas need ``yaw . delta . yaw^-1`` to stay aligned with the body.
    """
    if abs(yaw) < YAW_EPS:
        return body_delta
    yaw_q = quat_axis_angle(vec3(0.0, 1.0, 0.0), yaw)  # noqa: F821
    yaw_inv = quat_inverse(yaw_q)  # noqa: F821
    return quat_mul(yaw_q, quat_mul(body_delta, yaw_inv))  # noqa: F821


def _set_bone(name, body_delta, yaw):
    entry = _state["bone_drives"].get(name)
    if entry is None:
        return
    node, rest_rot = entry
    delta_world = _yaw_to_world(body_delta, yaw)
    parent_world = quat_axis_angle(vec3(1.0, 0.0, 0.0), 0.0)  # noqa: F821
    chain = []
    cur = node.parent
    while cur is not None:
        chain.append(cur.transform.rotation)
        cur = cur.parent
    for r in reversed(chain):
        parent_world = quat_mul(parent_world, r)  # noqa: F821
    parent_world_inv = quat_inverse(parent_world)  # noqa: F821
    delta_local = quat_mul(  # noqa: F821
        quat_mul(parent_world_inv, delta_world),  # noqa: F821
        parent_world,
    )
    node.transform.set_rotation(quat_mul(delta_local, rest_rot))  # noqa: F821
