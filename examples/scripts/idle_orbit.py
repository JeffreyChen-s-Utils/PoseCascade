"""Idle script: rotate the character about Y, orbit the dog around it.

Loaded by the script host through the sandboxed loader. Globals available:
``scene``, ``time`` (callable), ``vec3``, ``quat_axis_angle``, ``sin``,
``cos``, ``pi``, ``tau``, ``lerp``, ``clamp``. ``import`` is forbidden —
extend ``posecascade/scripting/api.py`` if a script needs more helpers.
"""

ORBIT_RADIUS = 1.5
ORBIT_PERIOD_SEC = 6.0
SPIN_PERIOD_SEC = 8.0
DOG_HEIGHT = 0.0


def update(dt):  # noqa: ARG001 — dt unused; we read total elapsed via time()
    elapsed = time()  # noqa: F821 — injected by sandbox

    character = scene.find("character")  # noqa: F821 — injected by sandbox
    if character is not None:
        spin_angle = (elapsed / SPIN_PERIOD_SEC) * tau  # noqa: F821
        character.transform.set_rotation(quat_axis_angle(vec3(0.0, 1.0, 0.0), spin_angle))  # noqa: F821

    dog = scene.find("dog")  # noqa: F821
    if dog is not None:
        orbit_angle = (elapsed / ORBIT_PERIOD_SEC) * tau  # noqa: F821
        x = ORBIT_RADIUS * cos(orbit_angle)  # noqa: F821
        z = ORBIT_RADIUS * sin(orbit_angle)  # noqa: F821
        dog.transform.set_translation(vec3(x, DOG_HEIGHT, z))  # noqa: F821
        # Face the direction of motion: tangent to the orbit circle.
        facing = orbit_angle + pi * 0.5  # noqa: F821
        dog.transform.set_rotation(quat_axis_angle(vec3(0.0, 1.0, 0.0), facing))  # noqa: F821
