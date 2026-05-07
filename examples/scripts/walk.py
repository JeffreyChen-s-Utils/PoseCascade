"""Locomotion demo — walk to stairs and climb, then walk back.

The character starts at the origin facing the staircase (which sits in
world -Z, away from the default camera at +Z=3). They walk toward the
stairs, take 5 deliberate big-stride steps to reach the top, turn
around, descend with the same big strides, and walk back to start.

Asset prep (already baked into ``character.glb``):
  - 5-step staircase, each step 20 cm wide × 4 cm deep × 2 cm rise.
  - Stair tops at world (Y=0.02, Z=-0.22) up to (Y=0.10, Z=-0.38).
  - The full staircase climbs 10 cm over 20 cm of horizontal travel.

Phases (16 second loop):

  Phase 1 (0-4 s):  Walk forward to the stair base.    (Z: 0 → -0.20)
  Phase 2 (4-9 s):  Big-stride climb, 5 steps × 1 sec.  (Y: 0 → 0.10, Z: -0.20 → -0.40)
  Phase 3 (9-10 s): Pause + turn 180° at top.
  Phase 4 (10-14 s):Big-stride descent.                 (Y: 0.10 → 0, Z: -0.40 → -0.20)
  Phase 5 (14-16 s):Walk back to origin + turn back.   (Z: -0.20 → 0)

Big-stride climb details (Phases 2 + 4): each "stride" is one full
step where:
  - First half: lifting leg rotates HIGH (≈55°) to clear the step.
  - Second half: leg plants on the next step, body rises to that
    height, trailing leg drags forward.

Walking gait elsewhere is gentler — small leg swing + arm
counter-coordination, no dance flourish.

Sandbox globals: ``scene``, ``physics_lite``, ``time``, ``vec3``,
``quat_axis_angle``, ``quat_mul``, ``sin``, ``cos``, ``pi``, ``tau``.

Run with::

    py -m posecascade --scene examples/assets/character.glb \
        --script examples/scripts/walk.py
"""

# ----- timing -----------------------------------------------------------
WALK_BPM = 120                    # 2 steps/sec for flat walking
STEP_SEC = 60.0 / WALK_BPM        # 0.5 s per walking step
STEP_CYCLE_SEC = STEP_SEC * 2.0   # full L+R walking cycle

# Phase durations (sum = 16 s)
PHASE_WALK_SEC = 4.0
PHASE_CLIMB_SEC = 5.0             # 5 stride steps × 1 s each
PHASE_TURN_SEC = 1.0
PHASE_DESCEND_SEC = 4.0           # 5 stride steps × 0.8 s each
PHASE_BACK_SEC = 2.0
LOOP_SEC = (
    PHASE_WALK_SEC + PHASE_CLIMB_SEC + PHASE_TURN_SEC
    + PHASE_DESCEND_SEC + PHASE_BACK_SEC
)

P_WALK_FORWARD = 0
P_CLIMB_UP = 1
P_TURN_TOP = 2
P_DESCEND = 3
P_WALK_BACK = 4

# Phase START times (cumulative)
T0 = 0.0
T1 = T0 + PHASE_WALK_SEC          # 4.0
T2 = T1 + PHASE_CLIMB_SEC         # 9.0
T3 = T2 + PHASE_TURN_SEC          # 10.0
T4 = T3 + PHASE_DESCEND_SEC       # 14.0
# T5 = T4 + PHASE_BACK_SEC         # 16.0

# ----- path waypoints (Y up, Z forward; stairs are on -Z side) ----------
WALK_DISTANCE = 0.20              # 20 cm forward (in -Z) — stair front edge
STAIR_RISE = 0.10                 # 10 cm climb (5 × 2 cm)
STAIR_FORWARD = 0.20              # 20 cm horizontal across stairs
STAIR_STEP_COUNT = 5              # number of distinct big strides
STAIR_BASE_Z = -WALK_DISTANCE     # -0.20

# ----- bone amplitudes -------------------------------------------------
# Sign convention (verified by hand vs the rig's rest rotations):
#   upper_leg_L rest = 180° around X. With BodyArmature yaw of π applied
#   for the walk-away phase, a NEGATIVE _x_quat angle on upper_leg
#   swings the leg in -Z = forward (toward the stairs); positive swings
#   it backward. Same sign holds during the walk-back phase (no yaw)
#   because both X and Z flip together.
#
# At peak stride: upper_leg rotates -0.55 forward AND lower_leg folds
# by -1.4, which lifts the foot ~6 cm above the rest plane — well
# clear of the 2 cm stair tread. The foot bone gets the inverse of
# (upper + lower) so the sole stays roughly horizontal instead of
# pointing at the sky.
WALK_LEG_LIFT = 0.50              # ~29° flat-walk leg swing (big strides)
WALK_KNEE_BEND = -0.30            # ~-17° knee flex during mid-swing —
                                  #     each knee bends most when the
                                  #     same-side leg is passing through
                                  #     neutral (mid-swing), straightens
                                  #     toward heel-strike / push-off.
STRIDE_LEG_LIFT = -0.70           # ~-40° upper-leg forward at peak lift
STRIDE_LEG_BACK = 0.25            # ~+14° trailing upper leg back
# Knee bend has the OPPOSITE body-frame sign of STRIDE_LEG_LIFT —
# this is the anatomical knee fold (lower leg rotates back relative
# to the thigh as the thigh swings forward), not "lower leg follows
# the upper". With same-sign values the world rotations of upper
# and lower add together, so the lower leg ends up rotating much
# more than the upper and the animation reads as "only the calf is
# moving while the thigh stays put". Opposite sign keeps the lower
# leg roughly vertical under the lifted knee — a natural step-up.
STRIDE_KNEE_BEND = +0.30          # ~+17° knee fold (opposite of upper lift)

# Arms hang at sides via _x_quat(ARM_DOWN_OVERRIDE) and swing forward /
# backward via _y_quat(swing) layered ON TOP. The down + swing
# rotations are pre-composed into a single quaternion delta in
# _arm_delta(); see the per-axis derivation in the module-level
# comment above _stride_pose. The same swing value goes to both arms
# — the L/R rest rotations are mirrored so identical Y rotations
# produce mirrored arm motion in world space.
ARM_DOWN_OVERRIDE = 0.0           # Reserved for rigs whose upper-arm
                                  #     bone needs an X-tilt at rest.
                                  #     Galaxia's arms only need the Z
                                  #     hang below; leave at 0.
# Galaxia's upper arms rest in T-pose along ±X. ARM_HANG is the world-Z
# rotation that flips them to a downward hang at -Y. Slightly less than
# π/2 in magnitude so the arms angle a few degrees outward from the
# torso instead of clipping into the rib cage.
ARM_HANG = -1.45                  # ~-83° on L arm, +83° mirrored on R
ARM_INWARD = 0.0                  # Extra inward tuck on top of ARM_HANG.
ARM_SWING_AMP = 0.40              # ~23° forward/back arm swing

LEAN_ASCENT = 0.10                # ~6° forward lean climbing
LEAN_DESCENT = -0.05              # ~-3° back lean descending
HIP_SWAY_WALK = 0.0               # no hip wiggle for plain walking
HEAD_TURN = 0.0                   # head stays neutral

# Character faces -Z direction throughout phases 1+2 (so they walk
# *away* from the camera at +Z). The base yaw quat applied to the body
# is π around world Y to flip the model from its authored +Z facing.
BASE_YAW_AWAY = 3.141592653589793         # π — facing -Z
BASE_YAW_TOWARD = 0.0                      # facing +Z (back toward camera)

CHARACTER_NODE_NAMES = ("Sketchfab_model",)
                                  # Only the SCENE ROOT gets the yaw +
                                  # translation applied. BodyArmature
                                  # is inside Sketchfab_model's parent
                                  # chain — driving both nodes with the
                                  # same delta produces a double
                                  # rotation that lays the character
                                  # flat. The Y-up→Z-up rest rotation
                                  # baked into Sketchfab_model is
                                  # preserved via quat_mul(yaw, rest).
DRIVEN_BONE_NAMES = (
    "head", "chest", "hip",
    "upper_arm_L", "upper_arm_R",
    "upper_leg_L", "upper_leg_R",
    "lower_leg_L", "lower_leg_R",
    "foot_L", "foot_R",
)

# ----- hair / wind -----------------------------------------------------
HAIR_TUNING = {
    "hair_C":  {"stiffness": 14.0, "damping": 0.45, "inertia": 0.015},
    "hair_L":  {"stiffness": 12.0, "damping": 0.40, "inertia": 0.015},
    "hair_R":  {"stiffness": 12.0, "damping": 0.40, "inertia": 0.015},
    "hair_LL": {"stiffness":  8.5, "damping": 0.30, "inertia": 0.020},
    "hair_RR": {"stiffness":  8.5, "damping": 0.30, "inertia": 0.020},
    "orn":     {"stiffness": 22.0, "damping": 0.80, "inertia": 0.020},
}
HAIR_WIND_BASE = 0.4

_HALF_CYCLE = 0.5

# Knee hinge limit for ik.solve_two_bone — bend backward only, ~0..-138°.
# The CCD step clamps the lower_leg's local rotation (Euler XYZ) so the
# knee never twists laterally or hyper-extends forward through itself.
_KNEE_LIMIT_MIN = (-2.4, 0.0, 0.0)
_KNEE_LIMIT_MAX = (0.1, 0.0, 0.0)

# Below this, treat the body as un-yawed and skip the conjugation
# matrix multiply in _yaw_to_world (cheap fast path; identity within
# numerical noise).
_YAW_NEGLIGIBLE = 1e-4

_state = {
    "configured": False,
    "character_drives": (),
    "bone_drives": {},
    "hair_wind": None,
    # Per-foot IK pin targets (frozen world position). None = release.
    "lock_target_L": None,
    "lock_target_R": None,
    "last_step_idx": -1,
    "leg_chain_L": None,   # (upper, lower, foot) Nodes
    "leg_chain_R": None,
}


def start():
    # noqa: F821 — sandbox-injected names
    drives = []
    for name in CHARACTER_NODE_NAMES:
        node = scene.find(name)  # noqa: F821
        if node is None:
            continue
        drives.append((
            node, node.transform.rotation.copy(), node.transform.translation.copy(),
        ))
    _state["character_drives"] = tuple(drives)

    bone_drives = {}
    for bone_name in DRIVEN_BONE_NAMES:
        node = scene.find(bone_name)  # noqa: F821
        if node is None:
            continue
        bone_drives[bone_name] = (node, node.transform.rotation.copy())
    _state["bone_drives"] = bone_drives

    # Cache leg chain Nodes for IK foot pinning.
    _state["leg_chain_L"] = (
        scene.find("upper_leg_L"),  # noqa: F821
        scene.find("lower_leg_L"),  # noqa: F821
        scene.find("foot_L"),       # noqa: F821
    )
    _state["leg_chain_R"] = (
        scene.find("upper_leg_R"),  # noqa: F821
        scene.find("lower_leg_R"),  # noqa: F821
        scene.find("foot_R"),       # noqa: F821
    )

    for chain_name, params in HAIR_TUNING.items():
        chain = physics_lite.get_chain(chain_name)  # noqa: F821
        if chain is None:
            continue
        chain.stiffness = params["stiffness"]
        chain.damping = params["damping"]
        chain.set_inertia(params["inertia"])

    _state["hair_wind"] = physics_lite.add_wind(  # noqa: F821
        direction=vec3(1.0, 0.0, 0.0),  # noqa: F821
        speed=HAIR_WIND_BASE,
        turbulence_amplitude=0.10,
        turbulence_frequency_hz=1.0,
    )

    # Engine-level foot planting: register both feet to a stair ground
    # provider that mirrors the asset's geometry. The host runs IK
    # AFTER walk.py's per-frame update, so any clip-through caused
    # by gait timing or IK undershoot gets corrected before render.
    floor.clear()  # noqa: F821 — sandbox-injected
    stair_ground = floor.stairs(  # noqa: F821
        base_z=STAIR_BASE_Z,
        step_depth=STAIR_FORWARD / STAIR_STEP_COUNT,
        step_rise=STAIR_RISE / STAIR_STEP_COUNT,
        count=STAIR_STEP_COUNT,
        base_y=0.0,
        forward_sign=-1,
    )
    # Galaxia foot bone: local +Y = toe direction, local -Z = sole.
    # Sample both heel (slightly back of joint) and toe (forward of
    # joint) so the planter catches whichever end has rotated below
    # ground when the foot is mid-stride dorsi/plantar-flexed. The
    # sole world-distance includes a small skin margin (~6 mm above
    # the bare ankle-to-sole 14 mm) so the foot mesh's lowest
    # vertices don't poke through after the joint is clamped.
    # Multi-sample at heel + toe for collision DETECTION (max
    # penetration = lift amount), but with alignment OFF below —
    # the planter only uses these to find the deepest mesh
    # contact point, not to tilt the foot. This catches mesh
    # parts (toe sticking out forward, heel behind ankle) that a
    # single point at the joint origin would miss when the leg
    # rotation makes the actual mesh extend further down than
    # 14 mm. Axes were derived by decomposing the world heel/toe
    # sole positions onto Galaxia's foot bone local axes.
    # Sole samples and foot offset are auto-derived by the engine
    # from the foot bone's skinned mesh — no hand-tuned constants
    # needed. ``floor.bind_foot`` introspects ``services.imported_*``
    # at bind time and picks heel / sole / toe extremes.
    # No sole_samples / sole_down_local / toe_forward_local — the
    # engine picks samples from the actual mesh and skips alignment.
    # Walk.py just nominates the chain + ground.
    if _state["leg_chain_L"][0] is not None:
        floor.bind_foot(  # noqa: F821
            *_state["leg_chain_L"], stair_ground,
            knee_limit_min=_KNEE_LIMIT_MIN, knee_limit_max=_KNEE_LIMIT_MAX,
        )
    if _state["leg_chain_R"][0] is not None:
        floor.bind_foot(  # noqa: F821
            *_state["leg_chain_R"], stair_ground,
            knee_limit_min=_KNEE_LIMIT_MIN, knee_limit_max=_KNEE_LIMIT_MAX,
        )

    _state["configured"] = True


def update(dt):  # noqa: ARG001
    if not _state["configured"]:
        return
    elapsed = time()  # noqa: F821
    loop_t = elapsed % LOOP_SEC
    phase, phase_t = _phase_for(loop_t)

    # 1. Maintain trailing-foot lock targets BEFORE we update the body
    #    pose this frame: we want to remember where the foot is sitting
    #    at the start of each new stride, freeze that, and let the IK
    #    drag the leg behind the body for the rest of the stride.
    if phase in (P_CLIMB_UP, P_DESCEND):
        _refresh_lock_targets(phase, phase_t)
    else:
        _state["lock_target_L"] = None
        _state["lock_target_R"] = None
        _state["last_step_idx"] = -1

    pos, yaw, lean = _path_state(phase, phase_t)
    _apply_root(pos, yaw, lean)
    _apply_gait(phase, phase_t, elapsed, yaw)
    _apply_foot_ik()
    # Tell the engine which direction the body currently faces, so
    # its post-tick foot planter can align toes the same way. Body
    # forward = R_y(yaw) * (0, 0, 1) = (sin yaw, 0, cos yaw).
    floor.set_body_forward((sin(yaw), 0.0, cos(yaw)))  # noqa: F821


def _phase_for(loop_t):
    """Return (phase index, normalised phase progress 0..1)."""
    if loop_t < T1:
        return P_WALK_FORWARD, (loop_t - T0) / PHASE_WALK_SEC
    if loop_t < T2:
        return P_CLIMB_UP, (loop_t - T1) / PHASE_CLIMB_SEC
    if loop_t < T3:
        return P_TURN_TOP, (loop_t - T2) / PHASE_TURN_SEC
    if loop_t < T4:
        return P_DESCEND, (loop_t - T3) / PHASE_DESCEND_SEC
    return P_WALK_BACK, (loop_t - T4) / PHASE_BACK_SEC


# ----- path -------------------------------------------------------------
def _path_state(phase, phase_t):
    """Return ((x, y, z) world position, yaw_radians, lean_radians)."""
    # noqa: F821
    if phase == P_WALK_FORWARD:
        # Walk in -Z direction (character is facing -Z so this is forward
        # in body frame). Position progresses from (0,0,0) to stair base.
        z = -WALK_DISTANCE * phase_t
        return (0.0, 0.0, z), BASE_YAW_AWAY, 0.0

    if phase == P_CLIMB_UP:
        # Stair climb in 5 discrete strides. Body Y + Z step up at the
        # end of each stride (smoothed with ease-in over the second half).
        y, z_offset = _stair_position(phase_t, ascending=True)
        return (0.0, y, STAIR_BASE_Z - z_offset), BASE_YAW_AWAY, LEAN_ASCENT

    if phase == P_TURN_TOP:
        # Stand at top + rotate from facing -Z to facing +Z.
        y = STAIR_RISE
        z = STAIR_BASE_Z - STAIR_FORWARD
        yaw = BASE_YAW_AWAY + (BASE_YAW_TOWARD - BASE_YAW_AWAY) * phase_t
        return (0.0, y, z), yaw, 0.0

    if phase == P_DESCEND:
        # Now facing +Z, descending means walking +Z (body forward) which
        # maps to world Z increasing back toward origin. Y decreases.
        y, z_offset = _stair_position(phase_t, ascending=False)
        # At phase_t=0: y=STAIR_RISE, z=STAIR_BASE_Z - STAIR_FORWARD (top, far)
        # At phase_t=1: y=0, z=STAIR_BASE_Z (bottom, near origin)
        return (0.0, y, STAIR_BASE_Z - STAIR_FORWARD + z_offset), BASE_YAW_TOWARD, LEAN_DESCENT

    # P_WALK_BACK: walk +Z back to origin while facing +Z (i.e., the
    # back of the character is the camera's view).
    z = STAIR_BASE_Z * (1.0 - phase_t)   # interpolate from -0.20 to 0
    return (0.0, 0.0, z), BASE_YAW_TOWARD, 0.0


def _stair_position(phase_t, ascending):
    """Return (body_y_world, z_offset_from_stair_base) for stair traversal.

    Body Y rises *within* each stride — not as an instant jump at the
    boundary. The rise window is step_t ∈ [0.15, 0.55] of each stride:
    the character starts the stride on the lower stair, the supporting
    leg pushes the body up while the other leg swings forward, by the
    midpoint the body is on the next stair's height, and the second
    half of the stride is a hold before the next stride starts. This
    ease-in window is what gives the "half-leg-lifted, walking up"
    feel rather than the previous teleport.

    Z translates linearly throughout the stride.
    """
    # noqa: F821
    rise_per_step = STAIR_RISE / STAIR_STEP_COUNT

    z_offset = STAIR_FORWARD * phase_t

    step_idx = min(int(phase_t * STAIR_STEP_COUNT), STAIR_STEP_COUNT - 1)
    step_t = phase_t * STAIR_STEP_COUNT - step_idx

    if ascending:
        base_y = step_idx * rise_per_step
        next_y = (step_idx + 1) * rise_per_step
    else:
        base_y = (STAIR_STEP_COUNT - step_idx) * rise_per_step
        next_y = (STAIR_STEP_COUNT - step_idx - 1) * rise_per_step

    # Body rises in parallel with the back half of the leg's forward
    # swing, so by the time the foot lands at step_t≈0.65 the body is
    # already most of the way up to the next stair. This avoids the
    # earlier "foot lands at base_y, then body shoots up" hop that
    # broke foot pinning, and keeps the foot from clipping the next
    # stair's leading edge during the cross-step phase.
    rise_start = 0.40
    rise_end = 0.75
    if step_t < rise_start:
        body_y = base_y
    elif step_t < rise_end:
        progress = (step_t - rise_start) / (rise_end - rise_start)
        eased = 0.5 - 0.5 * cos(progress * 3.141592653589793)  # noqa: F821
        body_y = base_y + (next_y - base_y) * eased
    else:
        body_y = next_y

    # The earlier "body bump" workaround for descent foot clipping was
    # removed once the IK pin (see ``_apply_foot_ik``) took over —
    # trailing foot stays locked to the upper-stair surface while the
    # body lowers, so no body lift is needed.

    return body_y, z_offset


def _apply_root(pos, yaw, lean):
    """Apply character orientation + translation.

    The simple humanoid build has all parent EMPTYs at identity, so a
    plain Y-axis yaw (composed with the captured rest rotation) works
    cleanly. Galaxia's VRoid bone hierarchy needs a different
    composition; revisit if we swap models.
    """
    # noqa: F821
    yaw_q = quat_axis_angle(vec3(0.0, 1.0, 0.0), yaw)  # noqa: F821
    lean_q = quat_axis_angle(vec3(1.0, 0.0, 0.0), lean) if lean != 0.0 else None  # noqa: F821
    for node, rest_rot, rest_trans in _state["character_drives"]:
        composed = quat_mul(yaw_q, rest_rot)  # noqa: F821
        if lean_q is not None:
            composed = quat_mul(lean_q, composed)  # noqa: F821
        node.transform.set_rotation(composed)
        node.transform.set_translation(vec3(  # noqa: F821
            float(rest_trans[0]) + pos[0],
            float(rest_trans[1]) + pos[1],
            float(rest_trans[2]) + pos[2],
        ))


# ----- foot IK ---------------------------------------------------------
def _refresh_lock_targets(phase, phase_t):  # noqa: ARG001
    """Snapshot the trailing foot's current world position at stride start.

    Frozen target = where the foot is sitting BEFORE this frame's body
    translation runs. Once a stride starts, the trailing foot is pinned
    here and the IK keeps it stuck through the body's rise / drop.
    """
    step_idx = min(int(phase_t * STAIR_STEP_COUNT), STAIR_STEP_COUNT - 1)
    if step_idx == _state["last_step_idx"]:
        return
    _state["last_step_idx"] = step_idx
    leading_l = (step_idx % 2 == 0)
    chain_l = _state["leg_chain_L"]
    chain_r = _state["leg_chain_R"]
    if leading_l:
        # R is the trailing leg: pin it. L lifts (release).
        if chain_r is not None and chain_r[2] is not None:
            _state["lock_target_R"] = world_position(chain_r[2])  # noqa: F821
        _state["lock_target_L"] = None
    else:
        if chain_l is not None and chain_l[2] is not None:
            _state["lock_target_L"] = world_position(chain_l[2])  # noqa: F821
        _state["lock_target_R"] = None


def _apply_foot_ik():
    """Run the 2-bone IK against any active lock targets."""
    target_l = _state["lock_target_L"]
    target_r = _state["lock_target_R"]
    if target_l is not None and _state["leg_chain_L"] is not None:
        up, lo, ft = _state["leg_chain_L"]
        if up is not None and lo is not None and ft is not None:
            ik.solve_two_bone(  # noqa: F821
                up, lo, ft, target_l,
                iterations=8,
                step_radian=0.6,
                mid_limit_min=_KNEE_LIMIT_MIN,
                mid_limit_max=_KNEE_LIMIT_MAX,
            )
    if target_r is not None and _state["leg_chain_R"] is not None:
        up, lo, ft = _state["leg_chain_R"]
        if up is not None and lo is not None and ft is not None:
            ik.solve_two_bone(  # noqa: F821
                up, lo, ft, target_r,
                iterations=8,
                step_radian=0.6,
                mid_limit_min=_KNEE_LIMIT_MIN,
                mid_limit_max=_KNEE_LIMIT_MAX,
            )


# ----- gait per phase ---------------------------------------------------
def _arm_delta(swing, side):
    """Compose T-pose→hang Z-tuck with a forward/back X swing.

    Galaxia's upper arms rest in T-pose along world ±X. ``ARM_HANG``
    rotates each arm down to -Y (mirrored via ``side`` so the L arm
    folds with -π/2-ish and the R arm with +π/2-ish). After that the
    arms sit vertical, so a rotation around world X is the real
    pendulum swing — and because both arms now point the same
    direction (-Y), ``swing`` is multiplied by ``side`` to make the
    two arms move opposite ways (cross-body coordination). The
    optional ``ARM_INWARD`` tuck stacks on top of the hang for
    rigs that need a bit more inward angle.
    """
    z_tuck = side * (ARM_HANG + ARM_INWARD)
    return quat_mul(  # noqa: F821
        _x_quat(side * swing),
        quat_mul(_z_quat(z_tuck), _x_quat(ARM_DOWN_OVERRIDE)),  # noqa: F821
    )


def _z_quat(angle):
    return quat_axis_angle(vec3(0.0, 0.0, 1.0), angle)  # noqa: F821


def _apply_gait(phase, phase_t, elapsed, body_yaw):
    bones = _state["bone_drives"]
    if not bones:
        return
    foot_l = 0.0
    foot_r = 0.0
    if phase in (P_CLIMB_UP, P_DESCEND):
        (
            arm_l_q, arm_r_q,
            leg_l, leg_r,
            knee_l, knee_r,
            foot_l, foot_r,
            hip_yaw, chest,
        ) = _stride_pose(phase_t)
    elif phase == P_TURN_TOP:
        arm_l_q = _arm_delta(0.0, +1)
        arm_r_q = _arm_delta(0.0, -1)
        leg_l = leg_r = knee_l = knee_r = 0.0
        hip_yaw = chest = 0.0
    else:
        (
            arm_l_q, arm_r_q,
            leg_l, leg_r,
            knee_l, knee_r,
            hip_yaw, chest,
        ) = _walk_pose(elapsed)

    head_yaw = HEAD_TURN * sin(elapsed * tau / 4.0)  # noqa: F821
    # Y-axis rotations (head turn, hip yaw) commute with body yaw and
    # need no conjugation. Every other delta below is computed in
    # body-frame and run through _yaw_to_world so its sign convention
    # stays consistent across walk-forward (yaw=π) and walk-back
    # (yaw=0) phases — without this the same value would mean
    # "forward" in one phase and "backward" in the other.
    _set_bone(bones, "head", _y_quat(head_yaw))
    _set_bone(bones, "hip", _y_quat(hip_yaw))
    _set_bone(bones, "chest", _yaw_to_world(_x_quat(chest), body_yaw))
    _set_bone(bones, "upper_arm_L", _yaw_to_world(arm_l_q, body_yaw))
    _set_bone(bones, "upper_arm_R", _yaw_to_world(arm_r_q, body_yaw))
    _set_bone(bones, "upper_leg_L", _yaw_to_world(_x_quat(leg_l), body_yaw))
    _set_bone(bones, "upper_leg_R", _yaw_to_world(_x_quat(leg_r), body_yaw))
    _set_bone(bones, "lower_leg_L", _yaw_to_world(_x_quat(knee_l), body_yaw))
    _set_bone(bones, "lower_leg_R", _yaw_to_world(_x_quat(knee_r), body_yaw))
    _set_bone(bones, "foot_L", _yaw_to_world(_x_quat(foot_l), body_yaw))
    _set_bone(bones, "foot_R", _yaw_to_world(_x_quat(foot_r), body_yaw))


def _yaw_to_world(body_delta, yaw):
    """Conjugate a body-frame rotation by ``yaw`` to get a world delta.

    When the character root is rotated by ``yaw`` around world Y, a
    bone delta authored in body-frame coordinates does not directly
    apply as a world delta — Y-axis rotations commute (no change),
    but X- and Z-axis rotations end up with their sign flipped under
    yaw=π. Conjugation here keeps the script's "+x = body-forward"
    sign convention intact across yaws, so a single body-frame value
    means the same thing in walk-forward (yaw=π) and walk-back (0).
    """
    if abs(yaw) < _YAW_NEGLIGIBLE:
        return body_delta
    yaw_q = quat_axis_angle(vec3(0.0, 1.0, 0.0), yaw)  # noqa: F821
    yaw_inv = quat_inverse(yaw_q)  # noqa: F821
    return quat_mul(yaw_q, quat_mul(body_delta, yaw_inv))  # noqa: F821


def _walk_pose(elapsed):
    """Flat-walk gait — big leg strides, knee flex, opposing arm swing.

    Returns (arm_l_q, arm_r_q,
             upper_leg_l, upper_leg_r,
             knee_l, knee_r,
             hip_yaw, chest).

    Knee flex follows the leg swing: ``(1 - cos(phase)) / 2`` peaks at
    mid-swing for the L leg and the corresponding ``(1 + cos(phase))
    / 2`` peaks at mid-swing for the R leg, so the knee unbends as
    the foot reaches its forward / back extreme and bends as it
    passes through neutral. Bend depth is ``WALK_KNEE_BEND``.
    """
    # noqa: F821
    phase = elapsed * tau / STEP_CYCLE_SEC  # noqa: F821
    cycle = sin(phase)  # noqa: F821
    leg_l = WALK_LEG_LIFT * cycle
    leg_r = -WALK_LEG_LIFT * cycle
    cos_phase = cos(phase)  # noqa: F821
    knee_l = WALK_KNEE_BEND * 0.5 * (1.0 - cos_phase)
    knee_r = WALK_KNEE_BEND * 0.5 * (1.0 + cos_phase)
    # Body-frame cross-body: at cycle=+1, leg_l = +0.5 (body-back),
    # so L arm must swing body-forward — that's a NEGATIVE body-frame
    # X rotation on the L shoulder (which sits at body-local -Y after
    # ARM_HANG hangs it). _arm_delta multiplies swing by ``side`` so
    # the R arm gets the opposite swing automatically; passing one
    # value to both calls is enough to get opposing motion.
    arm_swing = -ARM_SWING_AMP * cycle
    arm_l_q = _arm_delta(arm_swing, +1)
    arm_r_q = _arm_delta(arm_swing, -1)
    hip_yaw = HIP_SWAY_WALK * cos_phase
    return arm_l_q, arm_r_q, leg_l, leg_r, knee_l, knee_r, hip_yaw, 0.0


def _bell(t, start, end):
    """Half-sine envelope: 0 at ``start``, 1 at midpoint, 0 at ``end``.

    Outside ``[start, end]`` the envelope is flat 0, so the joint sits
    at its rest pose during the rest of the stride. The two leg
    envelopes overlap only in their tails — the knee is mostly back
    to neutral by the time the upper leg finishes its forward swing.
    """
    if t <= start or t >= end:
        return 0.0
    rel = (t - start) / (end - start)
    return sin(rel * 3.141592653589793)  # noqa: F821


def _stride_pose(phase_t):
    """Two-stage stair stride — lift the foot first, then plant it.

    Stride timeline (step_t):
      0.00–0.10  hold rest, body still on lower stair
      0.10–0.45  KNEE bell — knee folds, foot rises ~3 cm vertically
                  (the upper leg has barely moved → minimal forward
                  drift, so the lift reads as "picking the foot up")
      0.30–0.65  UPPER bell — upper leg swings forward, the (now
                  partly straightened) leg carries the foot across
                  to land on the next stair
      0.65–0.95  body rises onto the next stair (see _stair_position)
      0.95–1.00  brief plant before the next stride begins

    Returns (arm_l, arm_r, upper_l, upper_r, knee_l, knee_r,
             foot_l, foot_r, hip_yaw, chest).
    """
    # noqa: F821
    step_idx = min(int(phase_t * STAIR_STEP_COUNT), STAIR_STEP_COUNT - 1)
    step_t = (phase_t * STAIR_STEP_COUNT) - step_idx

    leading_l = (step_idx % 2 == 0)
    # Knee bell and forward-swing bell share the same window so the
    # upper leg starts rotating forward at the same instant the knee
    # starts folding. With staggered windows the first ~0.15 of the
    # stride had only the knee moving (calf flexing in place while
    # the thigh stood still), which read as "the upper half of the
    # leg isn't animating".
    knee_env = _bell(step_t, 0.10, 0.65)
    forward_env = _bell(step_t, 0.10, 0.65)

    leading_upper = STRIDE_LEG_LIFT * forward_env
    trailing_upper = STRIDE_LEG_BACK * forward_env
    leading_knee = STRIDE_KNEE_BEND * knee_env
    trailing_knee = 0.0
    # Foot dorsiflexion zero-ed: any non-zero value produces a tilt
    # mismatch with the leg's IK-determined orientation that
    # manifests as "climb 只用腳跟 / descend 只用前腳" (foot rolls
    # onto whichever end the residual angle dips toward). Leaving
    # this at 0 lets the foot inherit the leg chain orientation
    # cleanly; the stair-grounding planter then clamps the joint
    # height without introducing extra rotation.
    leading_foot = 0.0
    trailing_foot = 0.0

    if leading_l:
        upper_l, upper_r = leading_upper, trailing_upper
        knee_l, knee_r = leading_knee, trailing_knee
        foot_l, foot_r = leading_foot, trailing_foot
    else:
        upper_l, upper_r = trailing_upper, leading_upper
        knee_l, knee_r = trailing_knee, leading_knee
        foot_l, foot_r = trailing_foot, leading_foot

    # Body-frame cross-body for stride: when leading_l, leading_upper
    # is NEGATIVE (body-forward leg lift), so the same-side L arm
    # must swing body-back — POSITIVE swing on L. ``_arm_delta``
    # mirrors the swing across L/R via ``side`` so passing a single
    # ``arm_swing`` produces opposing arms.
    arm_swing = (+ARM_SWING_AMP if leading_l else -ARM_SWING_AMP) * forward_env
    arm_l_q = _arm_delta(arm_swing, +1)
    arm_r_q = _arm_delta(arm_swing, -1)

    return (
        arm_l_q, arm_r_q,
        upper_l, upper_r,
        knee_l, knee_r,
        foot_l, foot_r,
        0.0, 0.0,
    )


def _x_quat(angle):
    return quat_axis_angle(vec3(1.0, 0.0, 0.0), angle)  # noqa: F821


def _y_quat(angle):
    return quat_axis_angle(vec3(0.0, 1.0, 0.0), angle)  # noqa: F821


def _set_bone(bones, name, delta_world):
    """Apply ``delta_world`` (a WORLD-SPACE rotation) to the named bone.

    Converts the world-space delta into the bone's parent-local frame
    so it composes correctly with the bone's rest pose. This makes
    rotation conventions (``_x_quat``/``_y_quat``/``_z_quat``)
    independent of the rig's specific bone rest rotations — VRoid
    rigs (with hip 180°-Z baked in) and the older Sketchfab body
    behave the same way.
    """
    entry = bones.get(name)
    if entry is None:
        return
    node, rest_rot = entry
    # Compute parent's WORLD rotation by walking up.
    parent_world = quat_axis_angle(vec3(1.0, 0.0, 0.0), 0.0)  # noqa: F821 — identity
    chain = []
    cur = node.parent
    while cur is not None:
        chain.append(cur.transform.rotation)
        cur = cur.parent
    for r in reversed(chain):
        parent_world = quat_mul(parent_world, r)  # noqa: F821
    parent_world_inv = quat_inverse(parent_world)  # noqa: F821
    # Conjugate world-space delta into parent local frame.
    delta_local = quat_mul(  # noqa: F821
        quat_mul(parent_world_inv, delta_world),  # noqa: F821
        parent_world,
    )
    node.transform.set_rotation(quat_mul(delta_local, rest_rot))  # noqa: F821


def on_event(name, payload):  # noqa: ARG001
    """Reset hair to rest pose."""
    if name != "reset":
        return
    for chain in physics_lite.chains():  # noqa: F821
        chain.reset()
