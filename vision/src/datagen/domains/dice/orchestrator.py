import json
import os
import traceback

from . import geometry, glyphs, numbering, sampler
from ... import exporter, materials


def generate_batch(count, seed, outdir):
    os.makedirs(outdir, exist_ok=True)
    master_manifest = []
    failures = []

    for i in range(count):
        variant_seed = seed + i
        asset_id = f"asset_{i:05d}"
        try:
            record = _generate_one(asset_id, variant_seed, outdir)
            master_manifest.append(record)
        except Exception as e:
            failures.append({
                "asset_id": asset_id,
                "seed": variant_seed,
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    _write_manifest_and_failures(outdir, master_manifest, failures)

    return len(master_manifest), len(failures)


def generate_set_batch(num_sets, seed, outdir):
    os.makedirs(outdir, exist_ok=True)
    master_manifest = []
    failures = []

    for s in range(num_sets):
        set_seed = seed + s
        set_id = f"set_{s:05d}"
        variants = sampler.sample_set(set_seed)
        for die_type in sampler.DIE_TYPES:
            asset_id = f"{set_id}_{die_type}"
            try:
                record = _generate_from_params(asset_id, variants[die_type], outdir)
                record["set_id"] = set_id
                master_manifest.append(record)
            except Exception as e:
                failures.append({
                    "asset_id": asset_id,
                    "seed": set_seed,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })

    _write_manifest_and_failures(outdir, master_manifest, failures)

    return len(master_manifest), len(failures)


def _write_manifest_and_failures(outdir, master_manifest, failures):
    with open(os.path.join(outdir, "manifest.json"), "w") as f:
        json.dump(master_manifest, f, indent=2)
    with open(os.path.join(outdir, "failures.json"), "w") as f:
        json.dump(failures, f, indent=2)


def _generate_one(asset_id, seed, outdir):
    params = sampler.sample_variant(seed)
    return _generate_from_params(asset_id, params, outdir)


def _generate_from_params(asset_id, params, outdir):
    import bpy

    die_obj = geometry.build_die_base_mesh(params.die_type, params.size_mm)

    face_pairs = geometry.compute_opposite_face_pairs(die_obj)
    poles = geometry.compute_face_poles(die_obj, params.die_type)
    hemisphere_of_face = None
    if poles is not None:
        top_pole_z = max(p.z for p in poles.values())
        hemisphere_of_face = {
            face_idx: ("top" if pole.z == top_pole_z else "bottom")
            for face_idx, pole in poles.items()
        }
    assignment = numbering.assign_values_to_opposite_pairs(
        params.die_type, face_pairs, hemisphere_of_face=hemisphere_of_face,
    )
    if not numbering.verify_opposite_sum(params.die_type, face_pairs, assignment):
        raise ValueError(f"{asset_id}: numbering invariant failed for {params.die_type}")

    if params.glyph_method == "engraved":
        engraving_warnings = glyphs.apply_engraved_glyphs(
            die_obj, params.die_type, assignment, params.glyph_style,
            params.glyph_fill, params.font_or_style_id, params.size_mm,
        )
        mat = materials.build_material(die_obj.name, params.material_category, params.material_params)
        materials.apply_material(die_obj, mat, slot_index=0)
        if params.glyph_fill == "painted":
            # Fill lightness is chosen from the base material's REAL
            # rendered luminance, not its HSV params -- see
            # materials.build_fill_material.
            base_luminance = glyphs.material_rendered_luminance(mat, outdir, asset_id)
            fill_mat = materials.build_fill_material(
                die_obj.name, params.material_params, base_luminance=base_luminance,
            )
            materials.apply_material(die_obj, fill_mat, slot_index=1)
    else:
        engraving_warnings = []
        mat = materials.build_material(die_obj.name, params.material_category, params.material_params)
        materials.apply_material(die_obj, mat, slot_index=0)
        glyphs.apply_decal_glyphs(
            die_obj, params.die_type, assignment, params.glyph_style,
            params.font_or_style_id, params.size_mm, asset_id, outdir,
        )

    manifest_record = {
        "asset_id": asset_id,
        "die_type": params.die_type,
        "num_sides": len(numbering.get_values(params.die_type)),
        "size_mm": params.size_mm,
        "bevel_fraction": params.bevel_fraction,
        "numbering_scheme": params.numbering_scheme,
        "glyph_style": params.glyph_style,
        "glyph_method": params.glyph_method,
        "glyph_fill": params.glyph_fill,
        "font_or_style_id": params.font_or_style_id,
        "material_category": params.material_category,
        "material_params": params.material_params,
        "seed": params.seed,
        "engraving_warnings": engraving_warnings,
    }

    exporter.export_asset(die_obj, manifest_record, outdir, params.bevel_fraction, params.size_mm)

    bpy.data.objects.remove(die_obj, do_unlink=True)
    return manifest_record
