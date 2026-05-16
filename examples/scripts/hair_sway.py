"""Hair sway demo: tunes the auto-detected chains on herta.glb and adds wind.

Sandbox globals used: ``physics_lite``, ``scene``, ``time``, ``vec3``,
``quat_axis_angle``, ``quat_mul``, ``sin``, ``pi``, ``tau``. The chains
themselves are auto-attached by the glTF importer when loading
herta.glb — this script only adjusts their parameters, adds wind, and
nods the head to make the inertia-driven sway visible (most of the
hair mesh is head-weighted; without the anchor moving, the spring tips
sway but the bulk of the hair silhouette barely shifts).

Run with::

    py -m posecascade --scene examples/assets/herta/herta.glb \
        --script examples/scripts/hair_sway.py
"""

# Tuning per chain — heavily under-critically damped (ζ ≈ 0.15-0.25) so
# the strands trail visibly after the head moves and settle into a loose
# drape rather than snapping back. Heavy inertia means the bone tips
# lag behind their anchor through any motion — that lag is what reads
# as "hair" instead of a rigid stick. Earlier tunings (ζ ≈ 0.4 or
# higher) over-damped the trail so the strands just leaned and never
# actually swung. Names match the chains the auto-detector found.
_TUNING = {
    # The Herta — back hair (L/R strands hanging from BackHairUpper2).
    # Very loose: stiffness 1.0, damping 0.08 (ζ ≈ 0.10) and heavy
    # inertia so the strand trails dramatically. The previous "natural"
    # tuning (k≈2.5) still read as rigid because BackHair has 5
    # uniformly-stiff joints — each segment needs to be loose for the
    # tip to actually trail.
    "BackHair_L":    {"stiffness": 1.0, "damping": 0.08, "inertia": 0.15},
    "BackHair_R":    {"stiffness": 1.0, "damping": 0.08, "inertia": 0.15},
    # Upper back hair sits on the head — stiffer (it should pivot at the
    # crown rather than collapse) but still loose enough that the
    # BackHair chains below it inherit a swinging anchor.
    "BackHairUpper": {"stiffness": 2.5, "damping": 0.18, "inertia": 0.10},
    # March-7th-style glTF rigs (back-compat). Same loose-tip philosophy.
    "hair_C":  {"stiffness": 2.5, "damping": 0.18, "inertia": 0.10},
    "hair_L":  {"stiffness": 1.5, "damping": 0.11, "inertia": 0.13},
    "hair_R":  {"stiffness": 1.5, "damping": 0.11, "inertia": 0.13},
    "hair_LL": {"stiffness": 0.8, "damping": 0.06, "inertia": 0.15},
    "hair_RR": {"stiffness": 0.8, "damping": 0.06, "inertia": 0.15},
    "orn":     {"stiffness": 5.0, "damping": 0.35, "inertia": 0.08},
}

# Wind: gentle, mostly horizontal. Strong wind on loose under-damped
# springs produces over-shooting "flag whipping" — not natural hair.
# A modest base + slow gust gives steady drift the springs can trail.
WIND_BASE_SPEED = 1.0
WIND_GUST_AMPLITUDE = 0.6
WIND_GUST_PERIOD_SEC = 3.0

# Head sway — small enough to read as a breathing / looking-around
# motion rather than a dance head-bang. With chains this loose, even
# a 5° yaw produces ~15° of hair-tip trail, which looks alive.
HEAD_YAW_AMP_RAD = 0.09      # ≈ 5° peak-to-peak
HEAD_NOD_AMP_RAD = 0.04      # ≈ 2.3° forward / back
HEAD_SWAY_PERIOD_SEC = 5.5

_state = {"wind": None, "configured": False, "head": None, "head_rest_rot": None}


def start():
    # noqa: F821 — physics_lite, scene injected by sandbox  # pylint: disable=undefined-variable
    for name, params in _TUNING.items():
        chain = physics_lite.get_chain(name)  # noqa: F821
        if chain is None:
            continue
        chain.stiffness = params["stiffness"]
        chain.damping = params["damping"]
        chain.set_inertia(params["inertia"])

    _state["wind"] = physics_lite.add_wind(  # noqa: F821
        direction=vec3(1.0, 0.0, 0.2),  # noqa: F821
        speed=WIND_BASE_SPEED,
        turbulence_amplitude=0.15,
        turbulence_frequency_hz=1.5,
    )
    # Snapshot the head's rest rotation so the per-frame sway composes on
    # top of the rig's bind pose rather than overwriting any axis-fix.
    head = scene.find("head")  # noqa: F821 — canonical alias
    if head is not None:
        _state["head"] = head
        _state["head_rest_rot"] = head.transform.rotation.copy()
    _state["configured"] = True


def update(dt):  # noqa: ARG001 — dt unused; we drive the gust via total time
    if not _state["configured"]:
        return
    wind = _state["wind"]
    elapsed = time()  # noqa: F821
    if wind is not None:
        # Sinusoidal gust on top of the base wind so chains see varying force.
        gust = WIND_GUST_AMPLITUDE * sin(elapsed * tau / WIND_GUST_PERIOD_SEC)  # noqa: F821
        wind.speed = WIND_BASE_SPEED + gust

    head = _state["head"]
    if head is None or _state["head_rest_rot"] is None:
        return
    phase = elapsed * tau / HEAD_SWAY_PERIOD_SEC  # noqa: F821
    # Yaw on Y, nod on X — slightly out of phase so the head traces a
    # gentle figure-eight rather than swinging on a single plane.
    yaw_angle = HEAD_YAW_AMP_RAD * sin(phase)  # noqa: F821
    nod_angle = HEAD_NOD_AMP_RAD * sin(phase * 0.5 + pi * 0.5)  # noqa: F821
    yaw_q = quat_axis_angle(vec3(0.0, 1.0, 0.0), yaw_angle)  # noqa: F821
    nod_q = quat_axis_angle(vec3(1.0, 0.0, 0.0), nod_angle)  # noqa: F821
    delta = quat_mul(yaw_q, nod_q)  # noqa: F821
    head.transform.set_rotation(quat_mul(delta, _state["head_rest_rot"]))  # noqa: F821


def on_event(name, payload):  # noqa: ARG001 — payload unused
    """``head_kick`` event injects an impulse into every chain — useful when the
    user wants to demo the reactive sway from the script console or a UI button."""
    if name != "head_kick":
        return
    for chain in physics_lite.chains():  # noqa: F821
        chain.apply_impulse(axis=vec3(0.0, 0.0, 1.0), magnitude=4.0)  # noqa: F821
