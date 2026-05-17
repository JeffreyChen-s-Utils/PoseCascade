# Re-baking a custom rest pose in Blender

This guide is for the case where a rig's bind orientation makes a desired
pose impractical to author via JSON `bones` rotations alone — typically a
HoYoverse-style rig where the leg / arm bones won't accept large explicit
`x_rad` rotations cleanly, and the gait system has no kind for the target
pose (e.g. a quadruped crawl, a sitting pose, a sumo squat).

The fix is to bake the pose into the rig as a saved bone-pose action,
re-export the GLB, then reference the pose from JSON via the existing
`pose:` mechanism. All the rig-specific quirks get absorbed by the
saved action; the JSON stays clean.

## Prerequisites

- **Blender 4.0+** with the bundled glTF 2.0 add-on enabled (it is by
  default).
- The source GLB (in this repo, `examples/assets/herta/herta.glb`).
- A target pose in your head — e.g. "hands and knees with torso
  horizontal", "praying / seiza", "T-pose with arms 30° forward".

## Step 1 — import the GLB

In Blender, `File → Import → glTF 2.0` and select the source `.glb`.
After import the scene contains:

- One **Armature** object holding the bone hierarchy.
- One or more **Mesh** objects parented to the armature via vertex
  groups (the LBS skinning).

Select the armature, switch to **Pose Mode** (`Ctrl+Tab` → Pose). Bone
manipulation is now live: clicking + `R` rotates around an axis,
`R X X` rotates around the bone's local X, etc.

## Step 2 — author the pose

Manipulate bones until the rig sits in the target pose. Common tools:

- **Auto IK** (toolbar → Pose Tools → Auto IK on) lets you drag a wrist
  or ankle and the whole chain follows. Turn off when you're happy so
  further adjustments don't disturb existing joints.
- **Numeric input** (`N` panel → Item → Rotation) for precise per-axis
  values matching what the JSON would have written.
- **Mirror** (`Pose → Apply → Symmetrize`) if the pose is left/right
  symmetric.

For a dog-crawl pose specifically on the bundled Herta rig:

1. Click the **Hips** bone, type `R Y 90 Enter` to pitch the torso
   forward 90°.
2. Click each upper leg bone, type `R X 90 Enter` to fold the thigh
   back under the body.
3. Click each lower leg bone, type `R X -90 Enter` to fold the calf
   forward (knees + shins now flat on the floor).
4. Click each shoulder bone, type `R X -75 Enter` to swing the arm
   forward.
5. Click each elbow bone, type `R X -30 Enter` to soften the elbow so
   the wrist plants flat instead of stiff-arming.
6. Eye-balance the head bone so the character looks forward, not at the
   floor.

The exact axes vary per rig — Blender's number panel always shows the
**current rotation**, which is the ground truth. Don't transfer angles
from this guide blindly; tune live until the silhouette is right.

## Step 3 — save the pose as an Action

The Action editor is how Blender bundles a bone-pose set for export:

1. Open the **Dope Sheet** editor and switch its mode (top-left) to
   **Action Editor**.
2. With the armature selected, click **New** to create a fresh action.
   Name it something the JSON will reference, e.g. `dog_crawl`.
3. Select every bone in Pose Mode (`A` to select all).
4. Insert a keyframe at frame 1 with `I → LocRot` (location + rotation
   both, because the root needs to move too).
5. Stamp the action's **Fake User** (the shield icon) so Blender keeps
   it even when nothing's "using" it — the glTF exporter looks for
   shielded actions to bundle.

If you want **multiple poses** (e.g. dog_crawl + sit + prone), repeat
steps 2-5 for each one. Each becomes its own action and the JSON can
reference them by name.

## Step 4 — export back to GLB

`File → Export → glTF 2.0`. In the right-side panel:

- **Format**: GLB (single file). Keep it next to the original.
- **Include → Selected Objects**: armature + meshes.
- **Animation → Animations**: ✓
- **Animation → Use Current Frame**: leave off (we want the action
  data, not just frame 1's snapshot).
- **Animation → Group by NLA Track**: leave off (actions go in as
  named animations — exactly what PoseCascade's pose-library
  loader looks for).

Export. The new GLB now contains every shielded action as a named
animation channel.

## Step 5 — reference the pose from JSON

There are two ways to plug the Blender-authored bone rotations into
PoseCascade:

### 5a — `bones_local` (Blender-native)

Use this when you've finished posing in Blender and want to drop the
raw bone rotations straight into a phase. The runtime treats
`bones_local` values as **bone-local basis rotations** (matching
Blender's `pose_bone.rotation_euler` with `rotation_mode='XYZ'`) and
writes `node.transform.rotation = basis_quat × rest_rotation`
without the world-frame conjugation that the regular `bones:` path
applies.

Extract values from your Blender pose:

```python
import math
arm = bpy.data.objects["Armature"]   # your rig's armature name
for bone_name in ("Spine", "Chest", "Left arm", "Right arm", ...):
    pb = arm.pose.bones[bone_name]
    eul = pb.matrix_basis.to_euler('XYZ')   # critical: 'XYZ' = intrinsic XYZ
    print(f'  "{bone_name}": {{"x_rad": {eul.x:.3f}, '
          f'"y_rad": {eul.y:.3f}, "z_rad": {eul.z:.3f}}},')
```

Paste into the JSON under `bones_local`:

```json
{
  "schema_version": 1,
  "extends": "_herta_profile.json",
  "phases": [{
    "name": "hold",
    "duration_sec": 4.0,
    "body": {"translation": [0, -0.45, 0]},
    "bones_local": {
      "chest": {"x_rad": -0.916},
      "upper_arm_L": {"x_rad": 0.653, "y_rad": -0.463, "z_rad": 0.878}
    }
  }]
}
```

Notes:

- `bones_local` uses **intrinsic XYZ** Euler order (`Rx · Ry · Rz`),
  matching Blender's default. The regular `bones:` field uses
  extrinsic XYZ — they produce different rotations for the same
  three angle values. Don't mix the two paths on the same bone.
- The runtime resolves bone keys through the rig's `body_bones`
  alias map, so you can use canonical names like `"chest"` even
  when the GLB names the bone `"Chest_0179"`.
- Don't combine `bones_local` with the gait system — gait writes
  world-frame deltas through `_set_bone` and they don't compose
  cleanly with local-frame writes.

### 5b — `pose_library` preset (per-bone shorthand)

Use this when the same pose should be reused across multiple phases
(e.g. an "idle stance" referenced from several animations). Define
the pose once in a `pose_library` block, then phases reference it
with `pose: "<name>"`. The runtime composes it through the
gait-aware slerp path (different from `bones_local`'s hard write).

```json
{
  "schema_version": 1,
  "extends": "_herta_profile.json",
  "pose_library": {
    "my_pose": {
      "chest": {"x_rad": -0.10},
      "upper_arm_L": {"x_rad": -1.30, "z_rad": 0.55}
    }
  },
  "phases": [{
    "name": "hold", "duration_sec": 2.0, "pose": "my_pose"
  }]
}
```

> Loading actions from the GLB into the pose library is tracked for
> a future PR; until then, both paths above require copying values
> out of Blender's rotation panel by hand or by the script snippet
> above.

## Driving Blender from PoseCascade's MCP server

The bundled MCP server can run arbitrary Blender code if you have
Blender open with its own MCP add-on. The relevant tool is
`mcp__blender__execute_blender_code`:

```python
import bpy

armature = bpy.data.objects["Armature"]
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode='POSE')

# Apply some rotations
hips = armature.pose.bones["Hips_04"]
hips.rotation_mode = 'XYZ'
hips.rotation_euler = (1.5707, 0.0, 0.0)  # pitch torso 90° forward

# … further bone edits …

# Save as a new action with fake user
action = bpy.data.actions.new("dog_crawl")
armature.animation_data_create()
armature.animation_data.action = action
action.use_fake_user = True
bpy.ops.anim.keyframe_insert_menu(type='LocRotScale')
```

This is faster than clicking through Blender's UI when the pose is
well-defined, and produces the same exported GLB.

## Sanity checks

After the re-bake, run:

```bash
py -m posecascade --scene examples/assets/herta/herta.glb \
                   --script examples/scripts/dog_crawl.json
```

Confirm:

- The character holds the saved pose without "deformed" limbs.
- The cloth (skirt, cape) stays at or above Y=0 — the engine
  ground clamp covers every skinned mesh automatically when
  `ground.kind == "flat"`. If anything still clips, the mesh might
  be excluded via `auto_clamp_skinned_to_ground: false` somewhere in
  the document chain.

If a particular bone still looks wrong (e.g. one foot rotates
backward), the bind orientation likely shifted between the Blender
session and the GLB export. Re-open in Blender, verify Pose Mode shows
the intended rest, and re-export — Blender's "Apply Pose as Rest"
operator (`Ctrl+A → Apply Pose as Rest`) is sometimes needed to flush
a pose into the bind state.
