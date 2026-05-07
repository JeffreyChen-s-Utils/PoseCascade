"""Post-effect framework — MME-equivalent without HLSL.

PMX users come from MikuMikuEffect, where each effect is an HLSL ``.fx``
file that defines passes, render targets, and parameters. The effect
ecosystem there is huge but tied to D3D9 and HLSL — porting the runtime
verbatim isn't feasible. PoseCascade ships an analogous framework based
on GLSL fragments + TOML descriptors:

- Each effect is a fullscreen post-pass: input textures from previous
  passes / the main scene → fragment shader → output texture.
- A chain orders multiple effects and lets the user enable / disable
  individual passes + tune uniforms.

The plumbing for actually executing the chain (FBO pool, blit-to-screen)
is renderer work that lands in a follow-up; this module covers the
data layer (descriptor + chain), TOML I/O, and the four built-in
effects (autoluminous / hgshadow / o_greener / ikeshita_ray).
"""

from posecascade.render.effects.chain import (
    ChainEntry,
    EffectChain,
    EffectLibrary,
)
from posecascade.render.effects.descriptor import (
    EffectBlendMode,
    EffectDescriptor,
    EffectInput,
    EffectUniform,
    EffectUniformKind,
)
from posecascade.render.effects.executor import (
    CompiledEffect,
    EffectChainExecutor,
)
from posecascade.render.effects.loader import (
    load_descriptor_from_toml,
    serialize_chain_to_toml,
)

__all__ = [
    "ChainEntry",
    "CompiledEffect",
    "EffectBlendMode",
    "EffectChain",
    "EffectChainExecutor",
    "EffectDescriptor",
    "EffectInput",
    "EffectLibrary",
    "EffectUniform",
    "EffectUniformKind",
    "load_descriptor_from_toml",
    "serialize_chain_to_toml",
]
