"""Energetic dance routine — full-body coordinated moves.

Drops the "tap dance" framing for bigger, clearer signature moves.
Each 2-second phase commits to ONE iconic shape with the WHOLE body
(arms + legs + hip + lean + weight shift) moving together — that's
what makes the result read as natural choreography rather than
isolated body parts twitching independently.

The routine is built on three principles:

  1. **Whole-body coordination.** Every limb pose is paired with a
     matching root sway / lean / hip rotation that complements it.
     Arms reach in the direction the body weight shifts; opposite-
     side leg lifts to balance.
  2. **Cosine easing.** Sine waves snap through their zero crossings
     — using ``0.5 - 0.5*cos(...)`` instead gives an ease-in-and-out
     so each move feels like the body is *settling into* the pose,
     not whipping through it.
  3. **Amplitudes that push the rig.** K=2 proximity weights handle
     ~70° of bone rotation cleanly on this character. The caps below
     stay around that ceiling so motion is dramatic but not torn.

Phases:

  1. Bounce groove        — body sway + arm swing, the warmup
  2. Right-arm reach      — sweep R arm overhead + L knee up
  3. Left-arm reach       — mirror
  4. Body wave            — chest pitch + flowing arm sweep
  5. Hip pop point        — sharp hip pop with arm point
  6. Spin with V arms     — 360° spin, both arms held high
  7. Step-clap            — alternating side step + chest clap
  8. Finale flourish      — big lean + arm V + leg lift, full pose

Sandbox globals: ``scene``, ``physics_lite``, ``time``, ``vec3``,
``quat_axis_angle``, ``quat_mul``, ``sin``, ``cos``, ``pi``, ``tau``.

Run with::

    py -m posecascade --scene examples/assets/character.glb \
        --script examples/scripts/dance.py
"""

# ----- timing -----------------------------------------------------------
BPM = 110                    # slightly under 120 so motion has room to breathe
BEATS_PER_PHASE = 4
PHASE_COUNT = 8
BEAT_SEC = 60.0 / BPM
HALF_BEAT = BEAT_SEC / 2.0
PHASE_SEC = BEAT_SEC * BEATS_PER_PHASE
LOOP_SEC = PHASE_SEC * PHASE_COUNT

P_BOUNCE = 0
P_REACH_R = 1
P_REACH_L = 2
P_WAVE = 3
P_HIP_POP = 4
P_SPIN_V = 5
P_STEP_CLAP = 6
P_FINALE = 7

CHARACTER_NODE_NAMES = ("Sketchfab_model", "BodyArmature")

# ----- amplitudes ------------------------------------------------------
# Bigger than the previous tap version — readable from across the
# viewport. Limits stay below the K=2 proximity weights' ~70° pinch
# threshold for arms / legs.

# Root motion
WEIGHT_SHIFT = 0.05            # 5 cm lateral weight shift
WEIGHT_SHIFT_BIG = 0.07        # 7 cm during big phases
ROOT_YAW = 0.18                # ±10° root yaw
ROOT_YAW_BIG = 0.28            # ±16° big phases
LEAN_FORWARD = 0.10            # ~6° forward
LEAN_BACK = -0.10              # ~6° back
LEAN_BIG = 0.16                # ~9° finale

# Bone amplitudes
ARM_REACH_HIGH = 1.20           # ~69° arm reach overhead
ARM_OPEN_MID = 0.80             # ~46° arm out forward
ARM_CHEST_LOW = 0.30            # ~17° arm at chest level
ARM_FINALE_PEAK = 1.10          # ~63° finale arm raise
LEG_LIFT_HIGH = 0.55            # ~31° knee up
LEG_LIFT_MID = 0.35             # ~20° smaller lift
LEG_TAP = 0.18                  # ~10° toe tap
HIP_SWAY = 0.20                 # ~11° base hip rotation
HIP_POP = 0.32                  # ~18° sharp hip pop
HIP_FINALE = 0.30               # ~17° finale hip
CHEST_PITCH = 0.18              # ~10° chest forward/back
HEAD_TURN = 0.30                # ~17° head turn

DRIVEN_BONE_NAMES = (
    "head", "chest", "hip",
    "upper_arm_L", "upper_arm_R",
    "upper_leg_L", "upper_leg_R",
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
HAIR_WIND_BASE = 0.6
HAIR_WIND_GUST = 0.4
HAIR_CHAIN_JOINT_COUNT = 4

_HALF_CYCLE = 0.5
_FOOT_TAP_THRESHOLD = 0.7

_state = {
    "configured": False,
    "character_drives": (),
    "bone_drives": {},
    "hair_wind": None,
    "last_beat": -1,
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

    for chain_name, params in HAIR_TUNING.items():
        chain = physics_lite.get_chain(chain_name)  # noqa: F821
        if chain is None:
            continue
        chain.stiffness = params["stiffness"]
        chain.damping = params["damping"]
        chain.set_inertia(params["inertia"])

    _state["hair_wind"] = physics_lite.add_wind(  # noqa: F821
        direction=vec3(1.0, 0.0, 0.3),  # noqa: F821
        speed=HAIR_WIND_BASE,
        turbulence_amplitude=0.25,
        turbulence_frequency_hz=1.4,
    )
    _state["configured"] = True


def update(dt):  # noqa: ARG001
    if not _state["configured"]:
        return
    elapsed = time()  # noqa: F821
    loop_t = elapsed % LOOP_SEC
    beat_index = int(loop_t / BEAT_SEC) + 1
    phase = int(loop_t / PHASE_SEC)
    phase_t = (loop_t - phase * PHASE_SEC) / PHASE_SEC

    yaw, side_x, lean = _root_for(phase, phase_t, elapsed)
    _apply_root(yaw, side_x, lean)
    _apply_bones(phase, phase_t, elapsed)

    if beat_index != _state["last_beat"]:
        _state["last_beat"] = beat_index
        _kick_hair(beat_index, phase)
    _drive_hair_wind(loop_t, phase)


# ----- easing helpers --------------------------------------------------
def _ease_in_out(t):
    """0..1 cosine ease — slow start + slow end."""
    # noqa: F821
    return 0.5 - 0.5 * cos(t * pi)  # noqa: F821


def _bipulse(elapsed, period):
    """Smooth +1..-1..+1 wave via cosine. One full cycle per ``period``."""
    # noqa: F821
    return cos(elapsed * tau / period)  # noqa: F821


def _holdpose(elapsed, period):
    """0..1 ease-in to peak, hold, ease-out — one cycle per ``period``."""
    # noqa: F821
    phase = (elapsed / period) % 1.0
    # smooth bell curve: peaks at 0.5
    return 0.5 - 0.5 * cos(phase * tau)  # noqa: F821


# ----- root pose -------------------------------------------------------
def _root_for(phase, phase_t, elapsed):
    handler = _ROOT_HANDLERS.get(phase, _root_finale)
    return handler(phase_t, elapsed)


def _root_bounce(phase_t, elapsed):  # noqa: ARG001
    # noqa: F821
    yaw = ROOT_YAW * sin(elapsed * tau / (BEAT_SEC * 2.0))  # noqa: F821 — half-period sway
    side_x = WEIGHT_SHIFT * sin(elapsed * tau / (BEAT_SEC * 2.0))  # noqa: F821
    lean = LEAN_FORWARD * 0.5
    return yaw, side_x, lean


def _root_reach_r(phase_t, elapsed):  # noqa: ARG001
    """Body shifts left + leans back as right arm reaches up."""
    # noqa: F821
    progress = _ease_in_out(phase_t * 2.0 if phase_t < _HALF_CYCLE else (1.0 - phase_t) * 2.0)
    yaw = -ROOT_YAW * 0.6 * progress  # body turns away from raised arm
    side_x = -WEIGHT_SHIFT_BIG * progress
    lean = LEAN_BACK * progress
    return yaw, side_x, lean


def _root_reach_l(phase_t, elapsed):  # noqa: ARG001
    yaw, side_x, lean = _root_reach_r(phase_t, elapsed)
    return -yaw, -side_x, lean


def _root_wave(phase_t, elapsed):  # noqa: ARG001
    # noqa: F821 — body wave: pitch forward then back
    yaw = ROOT_YAW * 0.4 * sin(elapsed * tau / 4.0)  # noqa: F821
    side_x = WEIGHT_SHIFT * sin(elapsed * tau / 2.0)  # noqa: F821
    lean = LEAN_FORWARD * cos(phase_t * tau)  # noqa: F821 — ease forward then back
    return yaw, side_x, lean


def _root_hip_pop(phase_t, elapsed):  # noqa: ARG001
    # noqa: F821 — sharp body pops on the beat
    yaw = ROOT_YAW * 0.6 * sin(elapsed * tau / BEAT_SEC)  # noqa: F821
    side_x = WEIGHT_SHIFT_BIG * sin(elapsed * tau / BEAT_SEC)  # noqa: F821
    lean = LEAN_BACK * 0.3
    return yaw, side_x, lean


def _root_spin_v(phase_t, elapsed):  # noqa: ARG001
    # noqa: F821 — full 360° eased spin
    yaw = _ease_in_out(phase_t) * tau  # noqa: F821 — 0 → 2π
    return yaw, 0.0, LEAN_FORWARD * 0.5


def _root_step_clap(phase_t, elapsed):  # noqa: ARG001
    # noqa: F821 — alternating step left/right per beat
    yaw = ROOT_YAW * 0.3 * sin(elapsed * tau / BEAT_SEC)  # noqa: F821
    side_x = WEIGHT_SHIFT_BIG * sin(elapsed * tau / BEAT_SEC)  # noqa: F821
    return yaw, side_x, LEAN_FORWARD * 0.4


def _root_finale(phase_t, elapsed):  # noqa: ARG001
    # noqa: F821 — biggest motion of the loop
    yaw = ROOT_YAW_BIG * sin(elapsed * tau / 2.0)  # noqa: F821
    side_x = WEIGHT_SHIFT_BIG * sin(elapsed * tau / 2.0)  # noqa: F821
    lean = LEAN_BIG * sin(elapsed * tau / 2.0)  # noqa: F821
    return yaw, side_x, lean


_ROOT_HANDLERS = {
    P_BOUNCE: _root_bounce,
    P_REACH_R: _root_reach_r,
    P_REACH_L: _root_reach_l,
    P_WAVE: _root_wave,
    P_HIP_POP: _root_hip_pop,
    P_SPIN_V: _root_spin_v,
    P_STEP_CLAP: _root_step_clap,
    P_FINALE: _root_finale,
}


def _apply_root(yaw, side_x, lean):
    # noqa: F821
    yaw_q = quat_axis_angle(vec3(0.0, 1.0, 0.0), yaw)  # noqa: F821
    lean_q = quat_axis_angle(vec3(1.0, 0.0, 0.0), lean) if lean != 0.0 else None  # noqa: F821
    for node, rest_rot, rest_trans in _state["character_drives"]:
        composed = quat_mul(yaw_q, rest_rot)  # noqa: F821
        if lean_q is not None:
            composed = quat_mul(lean_q, composed)  # noqa: F821
        node.transform.set_rotation(composed)
        node.transform.set_translation(vec3(  # noqa: F821
            float(rest_trans[0]) + side_x,
            float(rest_trans[1]),
            float(rest_trans[2]),
        ))


# ----- bone pose -------------------------------------------------------
def _apply_bones(phase, phase_t, elapsed):
    bones = _state["bone_drives"]
    if not bones:
        return
    arm_l, arm_r, leg_l, leg_r, hip_yaw, chest = _bones_for(phase, phase_t, elapsed)
    head_yaw = HEAD_TURN * 0.6 * sin(elapsed * tau / 3.0)  # noqa: F821
    _set_bone(bones, "chest", _x_quat(chest))
    _set_bone(bones, "head", _y_quat(head_yaw))
    _set_bone(bones, "hip", _y_quat(hip_yaw))
    _set_bone(bones, "upper_arm_L", _x_quat(arm_l))
    _set_bone(bones, "upper_arm_R", _x_quat(arm_r))
    _set_bone(bones, "upper_leg_L", _x_quat(leg_l))
    _set_bone(bones, "upper_leg_R", _x_quat(leg_r))


def _bones_for(phase, phase_t, elapsed):
    handler = _BONE_HANDLERS.get(phase, _bones_finale)
    return handler(phase_t, elapsed)


def _bones_bounce(phase_t, elapsed):  # noqa: ARG001
    """Warmup: body sways, arms swing in counter-rhythm."""
    # noqa: F821
    swing = sin(elapsed * tau / (BEAT_SEC * 2.0))  # noqa: F821
    arm_l = ARM_OPEN_MID * 0.5 * swing
    arm_r = -arm_l
    hip_yaw = HIP_SWAY * swing
    leg_l = LEG_TAP * (1.0 if swing > _FOOT_TAP_THRESHOLD else 0.0)
    leg_r = LEG_TAP * (1.0 if swing < -_FOOT_TAP_THRESHOLD else 0.0)
    chest = CHEST_PITCH * 0.3 * swing
    return arm_l, arm_r, leg_l, leg_r, hip_yaw, chest


def _bones_reach_r(phase_t, elapsed):  # noqa: ARG001
    """Right arm sweeps up + L knee lifts. Hip rotates left."""
    # noqa: F821
    progress = _holdpose(elapsed, PHASE_SEC)  # peaks mid-phase, eases in/out
    arm_r = ARM_REACH_HIGH * progress
    arm_l = ARM_CHEST_LOW * (1.0 - progress) - ARM_OPEN_MID * 0.3 * progress
    leg_l = LEG_LIFT_MID * progress
    leg_r = 0.0
    hip_yaw = -HIP_SWAY * progress  # hip turns to balance the reach
    chest = CHEST_PITCH * -0.3 * progress  # slight back-arch
    return arm_l, arm_r, leg_l, leg_r, hip_yaw, chest


def _bones_reach_l(phase_t, elapsed):
    arm_l, arm_r, leg_l, leg_r, hip_yaw, chest = _bones_reach_r(phase_t, elapsed)
    return arm_r, arm_l, leg_r, leg_l, -hip_yaw, chest


def _bones_wave(phase_t, elapsed):  # noqa: ARG001
    """Body wave: chest pitches forward + arms flow in sequence."""
    # noqa: F821
    chest = CHEST_PITCH * cos(phase_t * tau)  # noqa: F821 — forward then back
    # Arms phase-shifted so they flow through the wave
    arm_l = ARM_OPEN_MID * sin(phase_t * tau)  # noqa: F821
    arm_r = ARM_OPEN_MID * sin(phase_t * tau + pi * 0.3)  # noqa: F821 — slight offset
    leg_l = LEG_TAP * 0.5
    leg_r = LEG_TAP * 0.5
    hip_yaw = HIP_SWAY * 0.4 * sin(elapsed * tau / 2.0)  # noqa: F821
    return arm_l, arm_r, leg_l, leg_r, hip_yaw, chest


def _bones_hip_pop(phase_t, elapsed):  # noqa: ARG001
    """Sharp hip pop with arm pointing in the direction."""
    # noqa: F821
    side = sin(elapsed * tau / BEAT_SEC)  # noqa: F821 — peaks every beat
    hip_yaw = HIP_POP * side
    # Both arms point toward the popped hip side
    if side > 0:
        arm_l = ARM_CHEST_LOW
        arm_r = ARM_OPEN_MID
    else:
        arm_l = ARM_OPEN_MID
        arm_r = ARM_CHEST_LOW
    # Opposite foot taps as counterbalance
    leg_l = LEG_TAP * (1.0 if side < -_HALF_CYCLE else 0.0)
    leg_r = LEG_TAP * (1.0 if side > _HALF_CYCLE else 0.0)
    chest = CHEST_PITCH * 0.2 * side
    return arm_l, arm_r, leg_l, leg_r, hip_yaw, chest


def _bones_spin_v(phase_t, elapsed):  # noqa: ARG001
    """Both arms held high in V during the spin."""
    # noqa: F821
    arm = ARM_REACH_HIGH * 0.85
    leg_l = LEG_LIFT_MID * 0.4 * sin(elapsed * tau / BEAT_SEC)  # noqa: F821
    leg_r = -leg_l * 0.5
    chest = CHEST_PITCH * -0.2  # slight back-arch under high arms
    return arm, arm, leg_l, leg_r, 0.0, chest


def _bones_step_clap(phase_t, elapsed):  # noqa: ARG001
    """Step left/right with chest-level arm clap each beat."""
    # noqa: F821
    side = sin(elapsed * tau / BEAT_SEC)  # noqa: F821
    # Both arms come together at center on each beat (clap)
    clap_pulse = abs(sin(elapsed * tau / BEAT_SEC))  # noqa: F821
    arm_l = ARM_OPEN_MID * 0.7 - ARM_CHEST_LOW * clap_pulse
    arm_r = ARM_OPEN_MID * 0.7 - ARM_CHEST_LOW * clap_pulse
    # Step pattern — small leg lifts toward the side of weight shift
    leg_l = LEG_LIFT_MID * (1.0 - clap_pulse) if side > 0 else 0.0
    leg_r = LEG_LIFT_MID * (1.0 - clap_pulse) if side < 0 else 0.0
    hip_yaw = HIP_SWAY * 0.6 * side
    chest = CHEST_PITCH * 0.4 * clap_pulse
    return arm_l, arm_r, leg_l, leg_r, hip_yaw, chest


def _bones_finale(phase_t, elapsed):  # noqa: ARG001
    """Big finale: both arms up + alternating leg lifts + dramatic hip."""
    # noqa: F821
    # Both arms swing in unison up + down on every beat
    arm_pulse = ARM_FINALE_PEAK * abs(sin(elapsed * tau / BEAT_SEC))  # noqa: F821
    arm_l = arm_pulse
    arm_r = arm_pulse
    # Legs alternate — high knee every other beat
    leg_l_phase = (elapsed / BEAT_SEC) % 2.0
    leg_l = LEG_LIFT_HIGH * sin(leg_l_phase * pi) if leg_l_phase < 1.0 else 0.0  # noqa: F821
    leg_r_phase = ((elapsed + BEAT_SEC) / BEAT_SEC) % 2.0
    leg_r = LEG_LIFT_HIGH * sin(leg_r_phase * pi) if leg_r_phase < 1.0 else 0.0  # noqa: F821
    hip_yaw = HIP_FINALE * sin(elapsed * tau / 2.0)  # noqa: F821
    chest = CHEST_PITCH * 0.5 * sin(elapsed * tau / 2.0)  # noqa: F821
    return arm_l, arm_r, leg_l, leg_r, hip_yaw, chest


_BONE_HANDLERS = {
    P_BOUNCE: _bones_bounce,
    P_REACH_R: _bones_reach_r,
    P_REACH_L: _bones_reach_l,
    P_WAVE: _bones_wave,
    P_HIP_POP: _bones_hip_pop,
    P_SPIN_V: _bones_spin_v,
    P_STEP_CLAP: _bones_step_clap,
    P_FINALE: _bones_finale,
}


def _x_quat(angle):
    return quat_axis_angle(vec3(1.0, 0.0, 0.0), angle)  # noqa: F821


def _y_quat(angle):
    return quat_axis_angle(vec3(0.0, 1.0, 0.0), angle)  # noqa: F821


def _set_bone(bones, name, delta_quat):
    entry = bones.get(name)
    if entry is None:
        return
    node, rest_rot = entry
    node.transform.set_rotation(quat_mul(delta_quat, rest_rot))  # noqa: F821


# ----- hair / wind -----------------------------------------------------
def _kick_hair(beat_index, phase):
    # noqa: F821
    magnitude_table = {
        P_BOUNCE: 1.0, P_REACH_R: 1.6, P_REACH_L: 1.6, P_WAVE: 1.4,
        P_HIP_POP: 2.0, P_SPIN_V: 2.2, P_STEP_CLAP: 1.4, P_FINALE: 2.4,
    }
    magnitude = magnitude_table.get(phase, 1.5)
    if phase == P_SPIN_V:
        axis = vec3(0.0, 0.0, 1.0)  # noqa: F821
    elif phase in (P_REACH_R, P_REACH_L):
        axis = vec3(0.0, 0.0, -1.0)  # noqa: F821 — pull back during reach
    else:
        axis = vec3(1.0, 0.0, 0.3) if beat_index % 2 else vec3(-1.0, 0.0, 0.3)  # noqa: F821
    for chain in physics_lite.chains():  # noqa: F821
        if not chain.name.startswith("hair_"):
            continue
        for joint_index in range(HAIR_CHAIN_JOINT_COUNT):
            chain.apply_impulse(
                axis=axis, magnitude=magnitude, joint_index=joint_index,
            )


def _drive_hair_wind(loop_t, phase):
    # noqa: F821
    hair_wind = _state["hair_wind"]
    if hair_wind is None:
        return
    pulse = HAIR_WIND_GUST * sin(loop_t * tau / (BEAT_SEC * 2.0))  # noqa: F821
    boost = 0.4 if phase == P_SPIN_V else 0.0
    hair_wind.speed = HAIR_WIND_BASE + pulse + boost


def on_event(name, payload):  # noqa: ARG001
    """Reset all dynamic systems."""
    if name != "reset":
        return
    for chain in physics_lite.chains():  # noqa: F821
        chain.reset()
