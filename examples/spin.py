"""Example user script: spin the first node about the Y axis.

Loaded by the script host through :func:`posecascade.scripting.sandbox.load_script`.
The globals available here are limited to the curated engine API
(``scene``, ``time``) plus a whitelisted subset of builtins. ``import`` is not
available — extend the API surface in ``posecascade/scripting/sandbox.py``
instead.
"""

ROTATIONS_PER_SECOND = 0.25


def update(dt):
    if not scene.root.children:  # noqa: F821 — scene injected by sandbox  # pylint: disable=undefined-variable
        return
    target = scene.root.children[0]  # noqa: F821
    angle = ROTATIONS_PER_SECOND * dt * 6.2831853
    rotation = list(target.transform.rotation)
    rotation[1] += angle
    target.transform.rotation[1] = rotation[1]
