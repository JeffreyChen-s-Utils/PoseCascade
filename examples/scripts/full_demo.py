"""PoseCascade hair-sway demo (full herta.glb scene).

Only hair is animated. Auto-detected spring chains are tuned with per-chain
stiffness/damping, exposed to a sin-driven gusty wind, and given a periodic
4-direction impulse so the sway is unmistakable.

Cloth physics is intentionally NOT used on this character — her dress is
rigged into the body mesh, so cloth simulation would tear the visible
silhouette. A character with separate-mesh cape / cloak topology would be
needed for clean cloth sim.

Run::

    py -m posecascade --scene examples/assets/herta/herta.glb \
        --script examples/scripts/full_demo.py
"""

# Per-chain hair tuning. Damping kept under-critical (ζ < 1) so a kick produces
# visible oscillation rather than a smooth lazy return. Outer strands looser
# than back hair; ornament chain stiffer so the head clip doesn't flop loosely.
# Critical damping for k=3, I=0.015 is c≈0.42 — we sit at ~0.15 (ζ≈0.35).
HAIR_TUNING = {
    "hair_C":  {"stiffness": 3.5, "damping": 0.18, "inertia": 0.015},
    "hair_L":  {"stiffness": 3.0, "damping": 0.15, "inertia": 0.015},
    "hair_R":  {"stiffness": 3.0, "damping": 0.15, "inertia": 0.015},
    "hair_LL": {"stiffness": 2.0, "damping": 0.12, "inertia": 0.020},
    "hair_RR": {"stiffness": 2.0, "damping": 0.12, "inertia": 0.020},
    "orn":     {"stiffness": 7.0, "damping": 0.40, "inertia": 0.020},
}

# Wind: hair stiffness × bone scale (~5mm) needs strong wind to be visibly
# displaced. Pure sideways direction so motion is left-right rather than
# front-back lift.
WIND_DIRECTION = (1.0, 0.0, 0.0)
HAIR_WIND_BASE = 1.5
HAIR_WIND_GUST = 1.0
WIND_GUST_PERIOD_SEC = 3.0

# Periodic hair impulse — guarantees visible motion no matter the wind tuning.
HAIR_IMPULSE_PERIOD_SEC = 2.5
HAIR_IMPULSE_MAGNITUDE = 1.8

_state = {
    "wind_hair": None,
    "configured": False,
    "last_impulse_at": -10.0,
}


def start():
    # noqa: F821 — physics_lite, vec3 injected by sandbox  # pylint: disable=undefined-variable
    for name, params in HAIR_TUNING.items():
        chain = physics_lite.get_chain(name)  # noqa: F821
        if chain is None:
            continue
        chain.stiffness = params["stiffness"]
        chain.damping = params["damping"]
        chain.set_inertia(params["inertia"])

    _state["wind_hair"] = physics_lite.add_wind(  # noqa: F821
        direction=vec3(*WIND_DIRECTION),  # noqa: F821
        speed=HAIR_WIND_BASE,
        turbulence_amplitude=0.3,
        turbulence_frequency_hz=1.4,
    )
    _state["configured"] = True


def update(dt):  # noqa: ARG001 — dt unused; gust + impulse drive off elapsed time
    if not _state["configured"]:
        return
    elapsed = time()  # noqa: F821

    # Sin-driven gust on top of base speed.
    hair_gust = HAIR_WIND_GUST * sin(elapsed * tau / WIND_GUST_PERIOD_SEC)  # noqa: F821
    hair_wind = _state["wind_hair"]
    if hair_wind is not None:
        hair_wind.speed = HAIR_WIND_BASE + hair_gust

    # Periodic impulse to every joint of every hair chain — guarantees visible
    # sway. Cycles direction so hair "bounces" rather than drifting one way.
    if elapsed - _state["last_impulse_at"] >= HAIR_IMPULSE_PERIOD_SEC:
        _state["last_impulse_at"] = elapsed
        cycle = int(elapsed / HAIR_IMPULSE_PERIOD_SEC) % 4
        directions = [
            vec3(1.0, 0.0, 0.0),    # noqa: F821
            vec3(0.0, 0.0, 1.0),    # noqa: F821
            vec3(-1.0, 0.0, 0.0),   # noqa: F821
            vec3(0.0, 0.0, -1.0),   # noqa: F821
        ]
        axis = directions[cycle]
        for chain in physics_lite.chains():  # noqa: F821
            if not chain.name.startswith("hair_"):
                continue
            for j in range(4):
                chain.apply_impulse(
                    axis=axis, magnitude=HAIR_IMPULSE_MAGNITUDE, joint_index=j,
                )


def on_event(name, payload):  # noqa: ARG001 — payload unused
    """``reset`` snaps every chain back to its rest pose."""
    if name != "reset":
        return
    for chain in physics_lite.chains():  # noqa: F821
        chain.reset()
