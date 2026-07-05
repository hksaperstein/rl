"""
Bakes the non-destructive edge bevel, exports the die as USD, renders a
thumbnail for visual spot-checking, and writes the per-asset JSON manifest.

Bevel uses limit_method='ANGLE' (not 'NONE') so it only rounds the die's
structural edges (e.g. a cube's ~90 degree edges) while leaving shallow
engraved-numeral recesses (much shallower angle deltas) crisp.
"""
import json
import math
import os

import bpy


def export_asset(die_obj, manifest_record, outdir, bevel_fraction, size_mm):
    os.makedirs(outdir, exist_ok=True)
    asset_id = manifest_record["asset_id"]

    mod = die_obj.modifiers.new(name="Bevel", type='BEVEL')
    mod.width = size_mm * bevel_fraction
    mod.limit_method = 'ANGLE'
    mod.angle_limit = math.radians(35)
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

    usd_path = os.path.join(outdir, f"{asset_id}.usd")
    bpy.ops.object.select_all(action='DESELECT')
    die_obj.select_set(True)
    bpy.context.view_layer.objects.active = die_obj
    bpy.ops.wm.usd_export(filepath=usd_path, selected_objects_only=True)

    thumb_path = os.path.join(outdir, f"{asset_id}_thumb.png")
    _render_thumbnail(die_obj, thumb_path, size_mm)

    manifest_record["usd_path"] = f"{asset_id}.usd"
    manifest_record["thumbnail_path"] = f"{asset_id}_thumb.png"
    manifest_path = os.path.join(outdir, f"{asset_id}.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_record, f, indent=2)

    return manifest_path


def _render_thumbnail(die_obj, thumb_path, size_mm, resolution=512):
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = True

    cam_data = bpy.data.cameras.new(f"{die_obj.name}_cam")
    cam_obj = bpy.data.objects.new(f"{die_obj.name}_cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    dist = size_mm * 0.12
    cam_obj.location = (dist, -dist, dist)
    direction = die_obj.location - cam_obj.location
    cam_obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    scene.camera = cam_obj

    light_data = bpy.data.lights.new(f"{die_obj.name}_light", type='SUN')
    light_data.energy = 3.0
    light_obj = bpy.data.objects.new(f"{die_obj.name}_light", light_data)
    light_obj.location = (dist, dist, dist * 1.5)
    bpy.context.collection.objects.link(light_obj)

    scene.render.filepath = thumb_path
    bpy.ops.render.render(write_still=True)

    bpy.data.objects.remove(cam_obj, do_unlink=True)
    bpy.data.objects.remove(light_obj, do_unlink=True)
