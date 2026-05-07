"""Hair sway demo: tunes the auto-detected chains on character.glb and adds wind.

Sandbox globals used: ``physics_lite`` (engine PhysicsHost facade), ``time``,
``vec3``, ``sin``, ``pi``, ``tau``. The chains themselves are auto-attached by
the glTF importer when loading character.glb — this script only adjusts
parameters and forces.

Run with::

    py -m posecascade --scene examples/assets/character.glb --script examples/scripts/hair_sway.py
"""

# Tuning per chain — inner back hair stiffer (less sway), outer strands looser
# so they trail more dramatically.
_TUNING = {
    "hair_C":  {"stiffness": 18.0, "damping": 3.0,  "inertia": 0.04},
    "hair_L":  {"stiffness": 14.0, "damping": 2.5,  "inertia": 0.04},
    "hair_R":  {"stiffness": 14.0, "damping": 2.5,  "inertia": 0.04},
    "hair_LL": {"stiffness": 9.0,  "damping": 2.0,  "inertia": 0.05},
    "hair_RR": {"stiffness": 9.0,  "damping": 2.0,  "inertia": 0.05},
    "orn":     {"stiffness": 26.0, "damping": 4.5,  "inertia": 0.05},
}

WIND_BASE_SPEED = 0.6
WIND_GUST_AMPLITUDE = 0.4
WIND_GUST_PERIOD_SEC = 4.0

_state = {"wind": None, "configured": False}


def start():
    # noqa: F821 — physics_lite injected by sandbox  # pylint: disable=undefined-variable
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
    _state["configured"] = True


def update(dt):  # noqa: ARG001 — dt unused; we drive the gust via total time
    if not _state["configured"]:
        return
    wind = _state["wind"]
    if wind is None:
        return
    elapsed = time()  # noqa: F821
    # Sinusoidal gust on top of the base wind so the chains see varying force.
    gust = WIND_GUST_AMPLITUDE * sin(elapsed * tau / WIND_GUST_PERIOD_SEC)  # noqa: F821
    wind.speed = WIND_BASE_SPEED + gust


def on_event(name, payload):  # noqa: ARG001 — payload unused
    """``head_kick`` event injects an impulse into every chain — useful when the
    user wants to demo the reactive sway from the script console or a UI button."""
    if name != "head_kick":
        return
    for chain in physics_lite.chains():  # noqa: F821
        chain.apply_impulse(axis=vec3(0.0, 0.0, 1.0), magnitude=4.0)  # noqa: F821
