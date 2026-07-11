"""
Production render-and-annotate stage: object-detection training images
from the dice asset library, with COCO annotations.

Builds on scripts/render_detection_samples.py (the observed prototype)
and closes its deliberately-deferred gaps:

- Occlusion-aware boxes via a flat-emission ID pass: after the beauty
  render, every die is re-materialed to a unique flat color, the world
  is blacked out, and a 1-sample re-render yields an exact per-die
  visibility mask. Boxes are the VISIBLE region (standard for
  detection); dice with fewer than MIN_VISIBLE_PIXELS are skipped
  entirely instead of shipping a box around something the model cannot
  see.
- Real COCO output (one shard file per worker; merge with --merge).
- JPEG images with camera-realism effects, all render-native or numpy
  (Blender's bundled Python has no PIL): exposure jitter via
  view_settings.exposure, optional real camera depth-of-field, sensor
  noise added to the pixel buffer, random JPEG quality. Closes part of
  the sim-to-real gap and cuts storage ~10x vs PNG.
- Distractor clutter: 0-3 random primitives dropped alongside the dice
  so the model learns what NOT to detect. Excluded from annotations and
  painted background-black in the ID pass.

Sharding: run N workers in parallel, each with --shard k --shards N
(disjoint scene indices, deterministic from --seed), then a final
  --merge pass combines coco_shard*.json into coco.json.

Run (single worker):
  blender --background --python scripts/render_detection_dataset.py -- \
      --count 1000 --seed 99 --shard 0 --shards 6 --outdir data/detection_v1
Merge (plain python, no blender needed):
  python3 scripts/render_detection_dataset.py --merge --outdir data/detection_v1
"""
import argparse
import glob
import json
import math
import os
import random
import sys

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
ASSET_DIR = os.path.join(REPO, "data", "raw", "dice_sets_v1")
HDRI_DIR = os.path.join(REPO, "data", "hdris")

MM_TO_M = 0.001
RES_X, RES_Y = 1024, 768
MIN_VISIBLE_PIXELS = 60
CLASSES = ["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"]
CATEGORY_ID = {c: i + 1 for i, c in enumerate(CLASSES)}
# ID-pass color spacing: die i gets red value (i+1)*ID_STEP/255. The step
# leaves room for anti-aliasing edge blends to round back to a valid id.
ID_STEP = 20


def merge(outdir):
    shards = sorted(glob.glob(os.path.join(outdir, "coco_shard*.json")))
    images, annotations = [], []
    img_id, ann_id = 1, 1
    for sp in shards:
        data = json.load(open(sp))
        remap = {}
        for im in data["images"]:
            remap[im["id"]] = img_id
            im["id"] = img_id
            images.append(im)
            img_id += 1
        for an in data["annotations"]:
            an["id"] = ann_id
            an["image_id"] = remap[an["image_id"]]
            annotations.append(an)
            ann_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": CATEGORY_ID[c], "name": c, "supercategory": "die"}
            for c in CLASSES
        ],
    }
    out = os.path.join(outdir, "coco.json")
    with open(out, "w") as f:
        json.dump(coco, f)
    print(f"MERGED {len(shards)} shards -> {out}: "
          f"{len(images)} images, {len(annotations)} annotations")


def main_blender():
    import bpy
    import numpy as np
    from mathutils import Vector

    argv = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, required=True,
                        help="total scenes across ALL shards")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--closeup", action="store_true",
                        help="datagen-v2 close-up mode (spec "
                        "2026-07-11-datagen-v2-closeup-design.md): 1-2 "
                        "dice per scene, camera distance computed from a "
                        "target frame-height fraction (decoupled from "
                        "die class) instead of sampled directly.")
    args = parser.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    manifest = json.load(open(os.path.join(ASSET_DIR, "manifest.json")))
    manifest_by_id = {r["asset_id"]: r for r in manifest}
    blend_files = sorted(glob.glob(os.path.join(ASSET_DIR, "*.blend")))
    hdris = sorted(glob.glob(os.path.join(HDRI_DIR, "*.hdr")))
    assert blend_files and hdris

    coco_images, coco_annotations = [], []
    ann_id = 1

    def append_die(blend_path):
        with bpy.data.libraries.load(blend_path, link=False) as (src, dst):
            dst.objects = [n for n in src.objects]
        obj = [o for o in dst.objects if o is not None and o.type == 'MESH'][0]
        bpy.context.scene.collection.objects.link(obj)
        obj.scale = (MM_TO_M, MM_TO_M, MM_TO_M)
        return obj

    def add_rigid(obj, typ, shape='CONVEX_HULL'):
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.rigidbody.object_add()
        obj.rigid_body.type = typ
        obj.rigid_body.collision_shape = shape
        obj.rigid_body.friction = 0.65
        obj.rigid_body.restitution = 0.1
        obj.rigid_body.use_margin = True
        obj.rigid_body.collision_margin = 0.0002
        if typ == 'ACTIVE':
            obj.rigid_body.mass = 0.005
        obj.select_set(False)

    def flat_emission(name, rgba):
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        for n in list(nt.nodes):
            nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = rgba
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
        return mat

    def compose(idx, rng):
        nonlocal ann_id
        bpy.ops.wm.read_homefile(use_empty=True)
        scene = bpy.context.scene
        scene.render.engine = 'BLENDER_EEVEE'
        scene.render.resolution_x = RES_X
        scene.render.resolution_y = RES_Y

        import colorsys
        bpy.ops.mesh.primitive_plane_add(size=1.5)
        ground = bpy.context.active_object
        gmat = bpy.data.materials.new("ground")
        gmat.use_nodes = True
        gnt = gmat.node_tree
        bsdf = gnt.nodes["Principled BSDF"]
        r, g, b = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.0, 0.9),
                                      rng.uniform(0.05, 0.95))
        bsdf.inputs["Base Color"].default_value = (r, g, b, 1.0)
        bsdf.inputs["Roughness"].default_value = rng.uniform(0.2, 0.95)
        if rng.random() < 0.15:
            bsdf.inputs["Metallic"].default_value = 1.0
        # ~half the grounds get a procedural two-tone pattern (noise /
        # voronoi / checker / waves at random scale) instead of a flat
        # color -- tabletop surfaces are rarely uniform.
        if rng.random() < 0.5:
            r2, g2, b2 = colorsys.hsv_to_rgb(rng.random(),
                                             rng.uniform(0.0, 0.9),
                                             rng.uniform(0.05, 0.95))
            tex_kind = rng.choice(["noise", "voronoi", "checker", "wave"])
            if tex_kind == "noise":
                tex = gnt.nodes.new("ShaderNodeTexNoise")
                tex.inputs["Scale"].default_value = rng.uniform(2, 40)
                fac_out = tex.outputs["Fac"]
            elif tex_kind == "voronoi":
                tex = gnt.nodes.new("ShaderNodeTexVoronoi")
                tex.inputs["Scale"].default_value = rng.uniform(3, 60)
                fac_out = tex.outputs["Distance"]
            elif tex_kind == "checker":
                tex = gnt.nodes.new("ShaderNodeTexChecker")
                tex.inputs["Scale"].default_value = rng.uniform(4, 80)
                fac_out = tex.outputs["Fac"]
            else:
                tex = gnt.nodes.new("ShaderNodeTexWave")
                tex.inputs["Scale"].default_value = rng.uniform(1, 20)
                fac_out = tex.outputs["Fac"]
            ramp = gnt.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color = (r, g, b, 1.0)
            ramp.color_ramp.elements[1].color = (r2, g2, b2, 1.0)
            gnt.links.new(fac_out, ramp.inputs["Fac"])
            gnt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
        ground.data.materials.append(gmat)

        world = bpy.data.worlds.new("w")
        world.use_nodes = True
        nt = world.node_tree
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(rng.choice(hdris))
        bg = nt.nodes["Background"]
        bg.inputs["Strength"].default_value = rng.uniform(0.35, 1.7)
        mapping = nt.nodes.new("ShaderNodeMapping")
        texco = nt.nodes.new("ShaderNodeTexCoord")
        mapping.inputs["Rotation"].default_value[2] = rng.uniform(0, 2 * math.pi)
        nt.links.new(texco.outputs["Generated"], mapping.inputs["Vector"])
        nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        scene.world = world

        n_dice = rng.randint(1, 2) if args.closeup else rng.randint(3, 8)
        picks = rng.sample(blend_files, n_dice)
        dice = []
        # Close-up mode packs dice near the frame center (real close-up
        # photos are single-die-dominant); the wider 0.075 radius used for
        # multi-die tabletop scenes would push a second die out of a tight
        # close-up frame.
        placement_radius_max = 0.02 if args.closeup else 0.075
        for i, bp in enumerate(picks):
            die = append_die(bp)
            angle = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(0.0, placement_radius_max)
            die.location = (radius * math.cos(angle),
                            radius * math.sin(angle),
                            0.022 + 0.012 * i)
            die.rotation_euler = (rng.uniform(0, 2 * math.pi),
                                  rng.uniform(0, 2 * math.pi),
                                  rng.uniform(0, 2 * math.pi))
            die["asset_id"] = os.path.splitext(os.path.basename(bp))[0]
            dice.append(die)

        distractors = []
        for _ in range(rng.randint(0, 5)):
            kind = rng.choice(["cube", "sphere", "cylinder", "cone",
                               "torus", "ico", "slab", "stick"])
            size = rng.uniform(0.006, 0.055)
            if kind == "cube":
                bpy.ops.mesh.primitive_cube_add(size=size)
            elif kind == "sphere":
                bpy.ops.mesh.primitive_uv_sphere_add(radius=size / 2)
            elif kind == "cylinder":
                bpy.ops.mesh.primitive_cylinder_add(radius=size / 2,
                                                    depth=size)
            elif kind == "cone":
                bpy.ops.mesh.primitive_cone_add(radius1=size / 2, depth=size)
            elif kind == "torus":
                bpy.ops.mesh.primitive_torus_add(
                    major_radius=size / 2, minor_radius=size * rng.uniform(0.08, 0.2))
            elif kind == "ico":
                bpy.ops.mesh.primitive_ico_sphere_add(
                    radius=size / 2, subdivisions=1)
            elif kind == "slab":
                bpy.ops.mesh.primitive_cube_add(size=size)
                bpy.context.active_object.scale = (
                    rng.uniform(1.0, 2.5), rng.uniform(1.0, 2.5),
                    rng.uniform(0.1, 0.35))
            else:  # stick
                bpy.ops.mesh.primitive_cylinder_add(radius=size * 0.12,
                                                    depth=size * rng.uniform(2, 5))
            d = bpy.context.active_object
            angle = rng.uniform(0, 2 * math.pi)
            radius = rng.uniform(0.03, 0.12)
            d.location = (radius * math.cos(angle), radius * math.sin(angle),
                          0.05 + rng.uniform(0, 0.03))
            d.rotation_euler = (rng.uniform(0, 6.3), rng.uniform(0, 6.3),
                                rng.uniform(0, 6.3))
            dmat = bpy.data.materials.new("distractor")
            dmat.use_nodes = True
            db = dmat.node_tree.nodes["Principled BSDF"]
            r, g, b = colorsys.hsv_to_rgb(rng.random(), rng.uniform(0.0, 1.0),
                                          rng.uniform(0.03, 0.97))
            db.inputs["Base Color"].default_value = (r, g, b, 1.0)
            db.inputs["Roughness"].default_value = rng.uniform(0.05, 0.95)
            db.inputs["Metallic"].default_value = rng.choice([0.0, 0.0, 0.0, 1.0])
            d.data.materials.append(dmat)
            distractors.append(d)

        # physics settle
        frames = 60
        bpy.ops.rigidbody.world_add()
        scene.rigidbody_world.point_cache.frame_end = frames
        add_rigid(ground, 'PASSIVE')
        for obj in dice + distractors:
            add_rigid(obj, 'ACTIVE')
        scene.frame_start = 1
        scene.frame_end = frames
        for f in range(1, frames + 1):
            scene.frame_set(f)
        # Read ALL settled matrices before removing ANY rigid body -- see
        # the prototype's comment: removal invalidates the sim cache.
        deps = bpy.context.evaluated_depsgraph_get()
        settled = {o.name: o.evaluated_get(deps).matrix_world.copy()
                   for o in dice + distractors}
        for obj in dice + distractors:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.rigidbody.object_remove()
            obj.select_set(False)
        for obj in dice + distractors:
            obj.matrix_world = settled[obj.name]

        # camera
        cam_data = bpy.data.cameras.new("cam")
        # Blender's default clip_start (0.1m) silently culls anything
        # closer than 10cm -- never an issue for detection_v1's 0.15-0.35m
        # camera distances, but closeup mode's computed distances can be
        # a few cm (found via smoke-test visual inspection: several
        # renders showed only a distractor/background with the die fully
        # clipped). 0.005m is well below closeup's clamped 0.05m floor and
        # harmless for the non-closeup range.
        cam_data.clip_start = 0.005
        cam_data.lens = rng.uniform(35, 80)
        cam = bpy.data.objects.new("cam", cam_data)
        scene.collection.objects.link(cam)
        scene.camera = cam
        az = rng.uniform(0, 2 * math.pi)
        elev = rng.uniform(math.radians(15), math.radians(65))
        if args.closeup:
            # Decouple apparent size from die class (datagen-v2 close-up
            # hypothesis): pick a target frame-height fraction independent
            # of which class is being rendered, then solve for the camera
            # distance that achieves it, instead of sampling distance
            # directly from a fixed range shared by every class. Target
            # range Uniform(0.28, 0.82) is measured from the real frozen
            # test set's own label heights, not assumed (see spec).
            target_frac = rng.uniform(0.28, 0.82)
            # die.dimensions is the world-space AABB extent post-scale and
            # is rotation-invariant (local-axis bbox * scale) -- a
            # deliberate approximation, since real apparent height still
            # varies with settled orientation, but dice are roughly
            # isotropic solids (manifest size_mm spans only 14-24mm across
            # all 7 classes), so the approximation error is small relative
            # to the size-class confound being removed, and does not
            # itself depend on class.
            size = max(max(d.dimensions) for d in dice)
            # angle_y is Blender's own computed vertical FOV (accounts for
            # sensor_fit/aspect correctly) -- read directly rather than
            # hand-deriving the sensor/aspect formula, which empirically
            # does not match a naive resolution-aspect-scaled calculation
            # for this project's default (non-VERTICAL) sensor_fit.
            dist = size / (2 * target_frac * math.tan(cam_data.angle_y / 2))
            dist = max(dist, 0.05)
            centroid = Vector((0.0, 0.0, 0.0))
            for d in dice:
                centroid += d.matrix_world.translation
            centroid /= len(dice)
            # Offset is applied relative to the actual dice centroid (not
            # a fixed world origin) so the achieved camera-to-subject
            # distance matches `dist` exactly even when the centroid is
            # off-origin (always possible at closeup's placement radius,
            # and would otherwise silently bias the target frame-fraction
            # since dist can be small, e.g. ~0.05m, comparable to the
            # placement radius). The non-closeup path below is untouched
            # (byte-identical to the pre-existing detection_v1 behavior:
            # camera offset from world origin, look-at a fixed point).
            cam.location = (centroid.x + dist * math.cos(az) * math.cos(elev),
                            centroid.y + dist * math.sin(az) * math.cos(elev),
                            centroid.z + dist * math.sin(elev))
            look = centroid - cam.location
        else:
            dist = rng.uniform(0.15, 0.35)
            cam.location = (dist * math.cos(az) * math.cos(elev),
                            dist * math.sin(az) * math.cos(elev),
                            dist * math.sin(elev))
            look = Vector((0, 0, 0.008)) - cam.location
        cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
        if rng.random() < 0.45:
            cam_data.dof.use_dof = True
            cam_data.dof.focus_distance = dist * rng.uniform(0.85, 1.1)
            cam_data.dof.aperture_fstop = rng.uniform(2.0, 8.0)
        scene.view_settings.exposure = rng.uniform(-0.4, 0.35)

        # beauty render (PNG buffer file; noise + JPEG conversion below)
        stem = f"img_{idx:06d}"
        png_path = os.path.join(args.outdir, stem + ".tmp.png")
        scene.render.filepath = png_path
        bpy.ops.render.render(write_still=True)

        # ID pass: unique flat emission per die, everything else black
        black = flat_emission("idpass_black", (0, 0, 0, 1))
        for obj in [ground] + distractors:
            obj.data.materials.clear()
            obj.data.materials.append(black)
        for i, die in enumerate(dice):
            idmat = flat_emission(f"idpass_{i}",
                                  ((i + 1) * ID_STEP / 255.0, 0, 0, 1))
            die.data.materials.clear()
            die.data.materials.append(idmat)
        wblack = bpy.data.worlds.new("wblack")
        wblack.use_nodes = True
        wblack.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.0
        scene.world = wblack
        scene.render.filter_size = 0.01
        scene.eevee.taa_render_samples = 1
        scene.view_settings.view_transform = 'Standard'
        # The beauty pass's camera-realism settings MUST NOT leak into
        # the ID pass: DOF blur smears id colors across the frame
        # (seen: a 706px-wide box from 672 scattered pixels) and the
        # exposure jitter rescales every id value.
        cam_data.dof.use_dof = False
        scene.view_settings.exposure = 0.0
        id_path = os.path.join(args.outdir, stem + ".id.png")
        scene.render.filepath = id_path
        bpy.ops.render.render(write_still=True)

        id_img = bpy.data.images.load(id_path)
        # Read RAW file values (the PNG stores sRGB-encoded numbers via
        # the Standard view transform); linearize explicitly. Decoding
        # the file values as if linear collided neighboring ids (seen:
        # two dice sharing one box).
        id_img.colorspace_settings.name = 'Non-Color'
        px = np.empty(RES_X * RES_Y * 4, dtype=np.float32)
        id_img.pixels.foreach_get(px)
        raw = px.reshape(RES_Y, RES_X, 4)[::-1, :, 0]  # flip to image coords
        linear = np.where(raw <= 0.04045, raw / 12.92,
                          ((raw + 0.055) / 1.055) ** 2.4)
        levels = linear * 255.0 / ID_STEP
        ids = np.rint(levels).astype(np.int32)
        # Anti-aliased edge pixels blend an id color toward black and can
        # land exactly on a lower id's level; discard anything not close
        # to a clean level.
        ids[np.abs(levels - ids) > 0.25] = 0
        bpy.data.images.remove(id_img)
        os.remove(id_path)

        # sensor noise + JPEG conversion (numpy + Blender's own writer)
        beauty = bpy.data.images.load(png_path)
        bpx = np.empty(RES_X * RES_Y * 4, dtype=np.float32)
        beauty.pixels.foreach_get(bpx)
        noise = np.random.default_rng(rng.getrandbits(32)).normal(
            0, rng.uniform(0.002, 0.012), bpx.shape).astype(np.float32)
        bpx = np.clip(bpx + noise, 0.0, 1.0)
        bpx[3::4] = 1.0
        beauty.pixels.foreach_set(bpx)
        scene.render.image_settings.file_format = 'JPEG'
        scene.render.image_settings.quality = rng.randint(65, 95)
        jpg_name = stem + ".jpg"
        beauty.save_render(os.path.join(args.outdir, jpg_name))
        bpy.data.images.remove(beauty)
        os.remove(png_path)
        scene.render.image_settings.file_format = 'PNG'

        image_id = idx + 1
        coco_images.append({"id": image_id, "file_name": jpg_name,
                            "width": RES_X, "height": RES_Y})
        for i, die in enumerate(dice):
            ys, xs = np.nonzero(ids == i + 1)
            if len(xs) < MIN_VISIBLE_PIXELS:
                continue
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            w, h = x1 - x0 + 1, y1 - y0 + 1
            asset_id = die["asset_id"]
            rec = manifest_by_id.get(asset_id, {})
            coco_annotations.append({
                "id": ann_id, "image_id": image_id,
                "category_id": CATEGORY_ID.get(rec.get("die_type"), 0),
                "bbox": [x0, y0, w, h], "area": w * h, "iscrowd": 0,
                "visible_pixels": int(len(xs)),
                "asset_id": asset_id,
                "glyph_style": rec.get("glyph_style"),
                "material_category": rec.get("material_category"),
            })
            ann_id += 1

    shard_path = os.path.join(args.outdir, f"coco_shard{args.shard}.json")

    def write_shard():
        tmp = shard_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"images": coco_images, "annotations": coco_annotations}, f)
        os.replace(tmp, shard_path)

    # Resume support: a killed worker loses nothing but its last few
    # scenes. Annotations are checkpointed every CHECKPOINT_EVERY scenes
    # (the first run of this script wrote the shard file only at the
    # very end -- a mid-run kill orphaned 6,193 rendered images with no
    # annotations). A scene counts as done only if it's in the
    # checkpoint AND its JPEG exists on disk.
    done_indices = set()
    if os.path.exists(shard_path):
        prev = json.load(open(shard_path))
        for im in prev["images"]:
            idx = int(im["file_name"].split("_")[1].split(".")[0])
            if os.path.exists(os.path.join(args.outdir, im["file_name"])):
                done_indices.add(idx)
        coco_images.extend(
            im for im in prev["images"]
            if int(im["file_name"].split("_")[1].split(".")[0]) in done_indices)
        kept_ids = {im["id"] for im in coco_images}
        coco_annotations.extend(
            a for a in prev["annotations"] if a["image_id"] in kept_ids)
        ann_id = max((a["id"] for a in coco_annotations), default=0) + 1
        print(f"SHARD{args.shard} RESUMING: {len(done_indices)} scenes already done")

    CHECKPOINT_EVERY = 20
    rng_master = random.Random(args.seed)
    scene_seeds = [rng_master.getrandbits(48) for _ in range(args.count)]
    my_indices = [i for i in range(args.count) if i % args.shards == args.shard]
    since_checkpoint = 0
    for n, i in enumerate(my_indices):
        if i in done_indices:
            continue
        compose(i, random.Random(scene_seeds[i]))
        since_checkpoint += 1
        if since_checkpoint >= CHECKPOINT_EVERY:
            write_shard()
            since_checkpoint = 0
        print(f"SHARD{args.shard} RENDERED {n + 1}/{len(my_indices)}", flush=True)

    write_shard()
    print(f"SHARD{args.shard} DONE -> {shard_path}")


if __name__ == "__main__":
    if "--merge" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--merge", action="store_true")
        parser.add_argument("--outdir", type=str, required=True)
        a = parser.parse_args()
        merge(a.outdir)
    else:
        main_blender()
