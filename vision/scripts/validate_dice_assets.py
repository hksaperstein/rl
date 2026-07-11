import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datagen.domains.dice import numbering

EXPECTED_SET_DIE_TYPES = {"d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"}


def validate(outdir):
    manifest_path = os.path.join(outdir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    errors = []
    for record in manifest:
        asset_id = record["asset_id"]
        usd_path = os.path.join(outdir, record["usd_path"])
        thumb_path = os.path.join(outdir, record["thumbnail_path"])

        if not os.path.exists(usd_path):
            errors.append(f"{asset_id}: missing USD file {usd_path}")
        elif os.path.getsize(usd_path) == 0:
            errors.append(f"{asset_id}: empty USD file {usd_path}")

        if not os.path.exists(thumb_path):
            errors.append(f"{asset_id}: missing thumbnail {thumb_path}")

        die_type = record["die_type"]
        expected_sides = len(numbering.get_values(die_type))
        if record["num_sides"] != expected_sides:
            errors.append(
                f"{asset_id}: num_sides {record['num_sides']} != expected "
                f"{expected_sides} for {die_type}"
            )

        for warning in record.get("engraving_warnings") or []:
            errors.append(f"{asset_id}: {warning}")

        for warning in record.get("mesh_quality_warnings") or []:
            errors.append(f"{asset_id}: {warning}")

        blend_rel_path = record.get("blend_path")
        if blend_rel_path:
            blend_path = os.path.join(outdir, blend_rel_path)
            if not os.path.exists(blend_path):
                errors.append(f"{asset_id}: missing .blend file {blend_path}")
            elif os.path.getsize(blend_path) == 0:
                errors.append(f"{asset_id}: empty .blend file {blend_path}")

        stl_rel_path = record.get("stl_path")
        if stl_rel_path:
            stl_path = os.path.join(outdir, stl_rel_path)
            if not os.path.exists(stl_path):
                errors.append(f"{asset_id}: missing STL file {stl_path}")
            elif os.path.getsize(stl_path) == 0:
                errors.append(f"{asset_id}: empty STL file {stl_path}")

    sets = defaultdict(list)
    for record in manifest:
        set_id = record.get("set_id")
        if set_id is not None:
            sets[set_id].append(record)

    for set_id, records in sets.items():
        die_types_seen = [r["die_type"] for r in records]
        seen_counts = {}
        for dt in die_types_seen:
            seen_counts[dt] = seen_counts.get(dt, 0) + 1

        missing = EXPECTED_SET_DIE_TYPES - set(die_types_seen)
        if missing:
            errors.append(f"{set_id}: missing die types {missing}")

        for dt, count in seen_counts.items():
            if dt in EXPECTED_SET_DIE_TYPES and count > 1:
                errors.append(f"{set_id}: duplicate die type {dt}")

    return errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("outdir")
    args = parser.parse_args()

    found_errors = validate(args.outdir)
    print(f"Checked manifest at {args.outdir}: {len(found_errors)} error(s).")
    for e in found_errors:
        print(" -", e)
    sys.exit(1 if found_errors else 0)
