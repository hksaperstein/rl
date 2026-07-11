#!/usr/bin/env python3
"""Draw COCO bounding boxes onto rendered detection images for spot-checking.

Writes box-overlaid copies (never touches the originals) so a human can
verify that annotations actually line up with the dice at the level the
data will be consumed: the merged coco.json, reloaded fresh from disk.

Usage:
  # 16 random previews into data/detection_v1/previews/
  python3 scripts/visualize_annotations.py --outdir data/detection_v1 --sample 16

  # specific images
  python3 scripts/visualize_annotations.py --outdir data/detection_v1 --files img_000000.jpg img_004242.jpg

  # every image (slow; ~10k files)
  python3 scripts/visualize_annotations.py --outdir data/detection_v1 --all

  # include per-die metadata (glyph style, material) in the labels
  python3 scripts/visualize_annotations.py --outdir data/detection_v1 --sample 16 --verbose-labels
"""
import argparse
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

# One fixed color per die class so previews are comparable across images.
CATEGORY_COLORS = {
    "d4": (230, 57, 70),
    "d6": (244, 162, 97),
    "d8": (233, 196, 106),
    "d10": (42, 157, 143),
    "d10_pct": (38, 70, 83),
    "d12": (108, 91, 123),
    "d20": (69, 123, 157),
}
FALLBACK_COLOR = (200, 200, 200)


def load_font(size=16):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_annotations(img, anns, cat_names, font, verbose):
    draw = ImageDraw.Draw(img)
    for a in anns:
        x, y, w, h = a["bbox"]
        name = cat_names.get(a["category_id"], f"cat{a['category_id']}")
        color = CATEGORY_COLORS.get(name, FALLBACK_COLOR)
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        label = name
        if verbose:
            label += f" | {a.get('material_category', '?')} | {a.get('glyph_style', '?')}"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        # Put the label chip above the box; flip inside if it would clip the top.
        ly = y - th - 8 if y - th - 8 >= 0 else y + 2
        draw.rectangle([x, ly, x + tw + 8, ly + th + 8], fill=color)
        draw.text((x + 4, ly + 2), label, fill=(255, 255, 255), font=font)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True,
                        help="dataset dir containing coco.json and the JPEGs")
    parser.add_argument("--previews", default=None,
                        help="where to write overlaid images (default: <outdir>/previews)")
    parser.add_argument("--sample", type=int, default=None, help="N random images")
    parser.add_argument("--files", nargs="*", default=None, help="specific file names")
    parser.add_argument("--all", action="store_true", help="every image in coco.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose-labels", action="store_true",
                        help="append material/glyph metadata to each label")
    args = parser.parse_args()

    coco = json.load(open(os.path.join(args.outdir, "coco.json")))
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    anns_by_image = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)

    if args.files:
        wanted = set(args.files)
        images = [im for im in coco["images"] if im["file_name"] in wanted]
        missing = wanted - {im["file_name"] for im in images}
        if missing:
            raise SystemExit(f"not in coco.json: {sorted(missing)}")
    elif args.all:
        images = coco["images"]
    else:
        n = args.sample or 16
        images = random.Random(args.seed).sample(coco["images"], n)

    previews_dir = args.previews or os.path.join(args.outdir, "previews")
    os.makedirs(previews_dir, exist_ok=True)
    font = load_font()

    for im in images:
        src = os.path.join(args.outdir, im["file_name"])
        img = Image.open(src).convert("RGB")
        draw_annotations(img, anns_by_image.get(im["id"], []),
                         cat_names, font, args.verbose_labels)
        img.save(os.path.join(previews_dir, im["file_name"]), quality=90)
    print(f"wrote {len(images)} previews -> {previews_dir}")


if __name__ == "__main__":
    main()
