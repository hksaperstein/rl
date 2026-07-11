"""
Prototype render-and-annotate stage for object-detection training data.

Composes randomized scenes from the generated dice asset library
(data/raw/dice_sets_v1): 3-8 dice appended from their exported .blend
files, dropped with Blender's built-in rigid-body physics so they rest in
natural poses, lit by a random Poly Haven HDRI (data/hdris), viewed from
a random camera orbit, rendered with EEVEE. Per-die 2D bounding boxes
are computed by projecting each settled die's vertices through the
camera; each image gets a JSON annotation record carrying die_type
class labels and asset_id lineage back to the asset manifests.

Prototype limitations (deliberate, noted for the real stage):
- Box visibility isn't occlusion-tested: a die fully hidden behind
  another still gets a box. Fine at these dice counts; the real stage
  should use Cryptomatte/object-index masks instead.
- No distractor clutter objects yet.

Run:
  blender --background --python scripts/render_detection_samples.py -- \
      --count 12 --seed 7 --outdir data/detection_samples
"""
import argparse
import glob
import json
import math
import os
import random
import sys

import bpy
from mathutils import Vector

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
ASSET_DIR = os.path.join(REPO, "data", "raw", "dice_sets_v1")
HDRI_DIR = os.path.join(REPO, "data", "hdris")

# Assets are authored in mm; scale to meters for physically-plausible
# rigid-body behavior and camera optics.
MM_TO_M = 0.001


def clear_scene():
    bpy.ops.wm.read_homefile(use_empty=True)


def append_die(blend_path, rng):
    with bpy.data.libraries.load(blend_path, link=False) as (src, dst):
        mesh_objs = [n for n in src.objects]
        dst.objects = mesh_objs
    appended = [o for o in dst.objects if o is not None and o.type == 'MESH']
    obj = appended[0]
    bpy.context.scene.collection.objects.link(obj)
    obj.scale = (MM_TO_M, MM_TO_M, MM_TO_M)
    return obj


def settle_physics(dice, ground, frames=140):
    scene = bpy.context.scene
    bpy.ops.rigidbody.world_add()
    scene.rigidbody_world.point_cache.frame_end = frames

    bpy.context.view_layer.objects.active = ground
    ground.select_set(True)
    bpy.ops.rigidbody.object_add()
    ground.rigid_body.type = 'PASSIVE'
    ground.rigid_body.friction = 0.7
    # Blender's default collision margin is 0.04m -- 2x the size of a
    # whole die at real scale, which left every die "resting" 4cm above
    # the ground (seen in the first smoke render as floating dice with
    # detached shadows). Explicit sub-millimeter margins fix it.
    ground.rigid_body.use_margin = True
    ground.rigid_body.collision_margin = 0.0002
    ground.select_set(False)

    for die in dice:
        bpy.context.view_layer.objects.active = die
        die.select_set(True)
        bpy.ops.rigidbody.object_add()
        die.rigid_body.type = 'ACTIVE'
        die.rigid_body.collision_shape = 'CONVEX_HULL'
        die.rigid_body.friction = 0.6
        die.rigid_body.restitution = 0.1
        die.rigid_body.mass = 0.005
        die.rigid_body.use_margin = True
        die.rigid_body.collision_margin = 0.0002
        die.select_set(False)

    scene.frame_start = 1
    scene.frame_end = frames
    for f in range(1, frames + 1):
        scene.frame_set(f)

    # Freeze the settled poses. Order is load-bearing: read EVERY die's
    # evaluated (simulated) matrix first, then remove rigid bodies, then
    # assign -- removing any object from the rigid-body world invalidates
    # the sim cache, so an interleaved copy/remove loop snaps the
    # not-yet-copied dice back to their pre-sim transforms (seen as every
    # die rendering at its initial mid-air drop position).
    deps = bpy.context.evaluated_depsgraph_get()
    settled = {die.name: die.evaluated_get(deps).matrix_world.copy() for die in dice}
    for die in dice:
        bpy.context.view_layer.objects.active = die
        die.select_set(True)
        bpy.ops.rigidbody.object_remove()
        die.select_set(False)
    for die in dice:
        die.matrix_world = settled[die.name]


def project_bbox(obj, cam, scene):
    """Pixel-space AABB of obj's evaluated vertices through cam."""
    from bpy_extras.object_utils import world_to_camera_view
    deps = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(deps)
    mesh = ev.to_mesh()
    xs, ys = [], []
    w = scene.render.resolution_x
    h = scene.render.resolution_y
    for v in mesh.vertices:
        co = world_to_camera_view(scene, cam, ev.matrix_world @ v.co)
        if co.z <= 0:
            continue
        xs.append(co.x * w)
        ys.append((1.0 - co.y) * h)
    ev.to_mesh_clear()
    if not xs:
        return None
    x0, x1 = max(0, min(xs)), min(w, max(xs))
    y0, y1 = max(0, min(ys)), min(h, max(ys))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]


def compose_and_render(idx, manifest_by_id, blend_files, hdris, outdir, rng):
    clear_scene()
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = 1024
    scene.render.resolution_y = 768

    # ground plane with a simple randomized material
    bpy.ops.mesh.primitive_plane_add(size=1.5)
    ground = bpy.context.active_object
    gmat = bpy.data.materials.new("ground")
    gmat.use_nodes = True
    bsdf = gmat.node_tree.nodes["Principled BSDF"]
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.0, 0.4), rng.uniform(0.15, 0.85))
    bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.3, 0.95)
    ground.data.materials.append(gmat)

    # HDRI world
    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    nt = world.node_tree
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(rng.choice(hdris))
    bg = nt.nodes["Background"]
    bg.inputs["Strength"].default_value = rng.uniform(0.4, 1.6)
    mapping = nt.nodes.new("ShaderNodeMapping")
    texco = nt.nodes.new("ShaderNodeTexCoord")
    mapping.inputs["Rotation"].default_value[2] = rng.uniform(0, 2 * math.pi)
    nt.links.new(texco.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    nt.links.new(env.outputs["Color"], bg.inputs["Color"])
    scene.world = world

    # dice
    n_dice = rng.randint(3, 8)
    picks = rng.sample(blend_files, n_dice)
    dice = []
    for i, bp in enumerate(picks):
        die = append_die(bp, rng)
        angle = rng.uniform(0, 2 * math.pi)
        radius = rng.uniform(0.0, 0.075)
        die.location = (radius * math.cos(angle), radius * math.sin(angle),
                        0.03 + 0.018 * i)
        die.rotation_euler = (rng.uniform(0, 2 * math.pi),
                              rng.uniform(0, 2 * math.pi),
                              rng.uniform(0, 2 * math.pi))
        die["asset_id"] = os.path.splitext(os.path.basename(bp))[0]
        dice.append(die)

    settle_physics(dice, ground)

    # camera orbit
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = rng.uniform(35, 80)
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    az = rng.uniform(0, 2 * math.pi)
    elev = rng.uniform(math.radians(15), math.radians(60))
    dist = rng.uniform(0.16, 0.34)
    cam.location = (dist * math.cos(az) * math.cos(elev),
                    dist * math.sin(az) * math.cos(elev),
                    dist * math.sin(elev))
    look = Vector((0, 0, 0.008)) - cam.location
    cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()

    # render
    img_name = f"sample_{idx:03d}.png"
    scene.render.filepath = os.path.join(outdir, img_name)
    bpy.ops.render.render(write_still=True)
    if os.environ.get("SAVE_SCENE_DEBUG"):
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(outdir, f"scene_{idx:03d}.blend"), copy=True)

    # annotations
    annos = []
    for die in dice:
        bbox = project_bbox(die, cam, scene)
        if bbox is None:
            continue
        asset_id = die["asset_id"]
        rec = manifest_by_id.get(asset_id, {})
        annos.append({
            "bbox_xywh": bbox,
            "class": rec.get("die_type", "unknown"),
            "asset_id": asset_id,
            "glyph_style": rec.get("glyph_style"),
            "material_category": rec.get("material_category"),
        })
    return {"image": img_name, "width": 1024, "height": 768, "annotations": annos}


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default="data/detection_samples")
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    manifest = json.load(open(os.path.join(ASSET_DIR, "manifest.json")))
    manifest_by_id = {r["asset_id"]: r for r in manifest}
    blend_files = sorted(glob.glob(os.path.join(ASSET_DIR, "*.blend")))
    hdris = sorted(glob.glob(os.path.join(HDRI_DIR, "*.hdr")))
    assert blend_files and hdris, "need assets and HDRIs"

    records = []
    for i in range(args.count):
        records.append(compose_and_render(
            i, manifest_by_id, blend_files, hdris, args.outdir, rng))
        print(f"RENDERED {i + 1}/{args.count}")

    with open(os.path.join(args.outdir, "annotations.json"), "w") as f:
        json.dump(records, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
