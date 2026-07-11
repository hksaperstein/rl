# Dice Detector v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train YOLO11s on the 10k synthetic `detection_v1` dice dataset, in two variants (synthetic-only vs synthetic+real), and measure sim-to-real transfer on a frozen real-photo test set.

**Architecture:** Ultralytics YOLO11s fine-tuned from COCO-pretrained weights. Synthetic COCO json converted to YOLO layout with an image-level 90/10 split; two public Roboflow real-photo datasets ingested, class-remapped, and split into a fine-tune slice + frozen test set. Evaluation computes per-class mAP via pycocotools with a `d10_pct→d10` merge on real images.

**Tech Stack:** Python 3.12 venv at `.venv/`, PyTorch ≥2.7 cu128 (RTX 5070 Ti is Blackwell/sm_120), ultralytics, roboflow, pycocotools, pytest.

## Global Constraints

- Repo root for all paths: `/home/saps/projects/Dice-Detection` (all commands run from there).
- Python: `.venv/bin/python` (created in Task 1). Never system python for training.
- Class order everywhere: `["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"]` → indices 0–6.
- Roboflow API key comes from env var `ROBOFLOW_API_KEY` — never hardcode, never commit it.
- `data/yolo/`, `data/real/`, `models/`, `.venv/` are gitignored derived artifacts; scripts and tests are committed.
- Training runs use `seed=42`, `imgsz=640`, `epochs=60`, `batch=32` (halve batch on OOM).
- Commits go straight to `main` of the Dice-Detection repo.
- This project never touches Isaac Sim — no `flock`/GPU lock needed, but do not run both trainings concurrently (16 GB VRAM).

---

### Task 1: Environment + spec amendment

**Files:**
- Create: `.venv/` (gitignored)
- Modify: `requirements.txt`, `.gitignore`, `docs/superpowers/specs/2026-07-10-dice-detector-v1-design.md`

**Interfaces:**
- Produces: working `.venv/bin/python` with `torch` (CUDA), `ultralytics`, `roboflow`, `pycocotools`, `pytest`, `pillow` importable. All later tasks use this interpreter.

- [ ] **Step 1: Create venv and install**

```bash
cd /home/saps/projects/Dice-Detection
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install ultralytics roboflow pycocotools pytest
```

- [ ] **Step 2: GPU smoke test (Blackwell sm_120 gate)**

```bash
.venv/bin/python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not available"
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
x = torch.randn(64, 3, 64, 64, device="cuda")
y = torch.nn.Conv2d(3, 8, 3).cuda()(x).sum()
y.backward() if y.requires_grad else None
print("conv fwd OK:", float(y))
EOF
```

Expected: prints a 2.7+ torch version, CUDA 12.8, "NVIDIA GeForce RTX 5070 Ti", and `conv fwd OK` with no sm_120 warning. If a "no kernel image" error appears, install the nightly cu128 wheel instead and re-run.

- [ ] **Step 3: YOLO end-to-end smoke test**

```bash
cd /home/saps/projects/Dice-Detection
.venv/bin/yolo predict model=yolo11n.pt source=data/detection_v1/img_000000.jpg device=0 save=False
```

Expected: completes without error, prints inference speed (weights auto-download to repo root; add `yolo11*.pt` to `.gitignore` in Step 4).

- [ ] **Step 4: Update requirements.txt and .gitignore**

Replace `requirements.txt` contents with:

```
# Detector training (install torch/torchvision from the cu128 index — see docs/superpowers/plans/2026-07-10-dice-detector-v1.md Task 1)
torch>=2.7
torchvision
ultralytics
roboflow
pycocotools
pytest
numpy
opencv-python
matplotlib
```

Append to `.gitignore` (create any missing lines only; keep existing content):

```
.venv/
data/yolo/
data/real/
models/
yolo11*.pt
```

- [ ] **Step 5: Amend spec — asset-level split is infeasible**

In `docs/superpowers/specs/2026-07-10-dice-detector-v1-design.md`, replace the bullet beginning "**Split by asset set, not by image.**" with:

```markdown
- **Split: image-level random 90/10 (seed 42).** The originally-specified
  asset-set-level split is infeasible: 9,948 of 10,000 images contain dice
  from more than one of the 600 asset sets, so no image partition keeps
  sets disjoint. Consequence: synthetic val mAP is optimistic (same die
  instances appear in train and val) and is used only for training
  monitoring — the frozen real test set is the sole generalization
  measure. Feedback to the generator for the next dataset iteration:
  sample each image's dice from a split-designated pool of asset sets so a
  leak-free synthetic val exists.
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt .gitignore docs/superpowers/specs/2026-07-10-dice-detector-v1-design.md
git commit -m "chore: detector training env + spec amendment (image-level split)"
```

---

### Task 2: COCO→YOLO conversion script

**Files:**
- Create: `scripts/convert_coco_to_yolo.py`
- Test: `tests/test_convert_coco_to_yolo.py`

**Interfaces:**
- Consumes: `data/detection_v1/coco.json`, `data/detection_v1/*.jpg`, `.venv` from Task 1.
- Produces:
  - `data/yolo/images/{train,val}/` (symlinks to detection_v1 jpgs), `data/yolo/labels/{train,val}/*.txt`
  - `data/yolo/dice.yaml` (ultralytics dataset config; `names:` in the Global Constraints class order)
  - Importable functions: `coco_bbox_to_yolo(bbox, img_w, img_h) -> tuple[float, float, float, float]` and `build_split(image_ids: list[int], val_frac: float = 0.1, seed: int = 42) -> tuple[set[int], set[int]]` (returns `(train_ids, val_ids)`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_convert_coco_to_yolo.py`:

```python
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from convert_coco_to_yolo import coco_bbox_to_yolo, build_split, CLASS_NAMES


def test_class_order_locked():
    assert CLASS_NAMES == ["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"]


def test_bbox_conversion():
    # 100x50 box at top-left corner (10,20) in a 1000x500 image
    cx, cy, w, h = coco_bbox_to_yolo([10, 20, 100, 50], 1000, 500)
    assert cx == pytest.approx(0.06)   # (10+50)/1000
    assert cy == pytest.approx(0.09)   # (20+25)/500
    assert w == pytest.approx(0.10)
    assert h == pytest.approx(0.10)


def test_split_deterministic_and_disjoint():
    ids = list(range(100))
    train1, val1 = build_split(ids, val_frac=0.1, seed=42)
    train2, val2 = build_split(ids, val_frac=0.1, seed=42)
    assert train1 == train2 and val1 == val2
    assert train1.isdisjoint(val1)
    assert len(val1) == 10 and len(train1) == 90


def test_end_to_end_tiny_dataset(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(2):
        Image.new("RGB", (100, 80)).save(src / f"img_{i:06d}.jpg")
    coco = {
        "images": [
            {"id": 1, "file_name": "img_000000.jpg", "width": 100, "height": 80},
            {"id": 2, "file_name": "img_000001.jpg", "width": 100, "height": 80},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 7, "bbox": [10, 10, 20, 20]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 50, 40]},
        ],
        "categories": [
            {"id": 1, "name": "d4"}, {"id": 2, "name": "d6"}, {"id": 3, "name": "d8"},
            {"id": 4, "name": "d10"}, {"id": 5, "name": "d10_pct"},
            {"id": 6, "name": "d12"}, {"id": 7, "name": "d20"},
        ],
    }
    (src / "coco.json").write_text(json.dumps(coco))
    out = tmp_path / "yolo"
    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_coco_to_yolo.py"
    subprocess.run(
        [sys.executable, str(script), "--src", str(src), "--out", str(out), "--val-frac", "0.5"],
        check=True,
    )
    label_files = sorted(out.glob("labels/*/*.txt"))
    assert len(label_files) == 2
    all_lines = [l for f in label_files for l in f.read_text().splitlines()]
    # category_id 7 = d20 -> class 6; category_id 1 = d4 -> class 0
    classes = sorted(int(l.split()[0]) for l in all_lines)
    assert classes == [0, 6]
    assert (out / "dice.yaml").exists()
    # every label has a matching image symlink
    for f in label_files:
        split = f.parent.name
        assert (out / "images" / split / (f.stem + ".jpg")).exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && .venv/bin/pytest tests/test_convert_coco_to_yolo.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'convert_coco_to_yolo'`

- [ ] **Step 3: Write the script**

Create `scripts/convert_coco_to_yolo.py`:

```python
#!/usr/bin/env python3
"""Convert a COCO detection json + images dir to ultralytics YOLO layout.

Images are symlinked (not copied); labels are written as YOLO txt.
Split is image-level random (see spec amendment 2026-07-10: an
asset-set-level split is infeasible because 99.5% of images mix sets).
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

CLASS_NAMES = ["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"]


def coco_bbox_to_yolo(bbox, img_w, img_h):
    """COCO [x, y, w, h] (top-left, pixels) -> YOLO (cx, cy, w, h) normalized."""
    x, y, w, h = bbox
    return ((x + w / 2) / img_w, (y + h / 2) / img_h, w / img_w, h / img_h)


def build_split(image_ids, val_frac=0.1, seed=42):
    """Deterministic random split. Returns (train_ids, val_ids) as sets."""
    ids = sorted(image_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = round(len(ids) * val_frac)
    return set(ids[n_val:]), set(ids[:n_val])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("data/detection_v1"))
    ap.add_argument("--out", type=Path, default=Path("data/yolo"))
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    coco = json.loads((args.src / "coco.json").read_text())
    cat_to_idx = {}
    for cat in coco["categories"]:
        cat_to_idx[cat["id"]] = CLASS_NAMES.index(cat["name"])

    anns_by_image = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_image[a["image_id"]].append(a)

    train_ids, val_ids = build_split(
        [im["id"] for im in coco["images"]], args.val_frac, args.seed
    )

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "val": 0}
    for im in coco["images"]:
        split = "train" if im["id"] in train_ids else "val"
        stem = Path(im["file_name"]).stem
        lines = []
        for a in anns_by_image[im["id"]]:
            cx, cy, w, h = coco_bbox_to_yolo(a["bbox"], im["width"], im["height"])
            cls = cat_to_idx[a["category_id"]]
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        link = args.out / "images" / split / im["file_name"]
        if not link.exists():
            link.symlink_to((args.src / im["file_name"]).resolve())
        counts[split] += 1

    yaml_text = (
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (args.out / "dice.yaml").write_text(yaml_text)
    print(f"[DONE] train={counts['train']} val={counts['val']} -> {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && .venv/bin/pytest tests/test_convert_coco_to_yolo.py -v`
Expected: 4 passed

- [ ] **Step 5: Run the real conversion and sanity-check**

```bash
cd /home/saps/projects/Dice-Detection
.venv/bin/python scripts/convert_coco_to_yolo.py
ls data/yolo/labels/train | wc -l   # expected 9000
ls data/yolo/labels/val | wc -l     # expected 1000
head -3 data/yolo/labels/val/$(ls data/yolo/labels/val | head -1)
```

Expected: 9000/1000 label files; label lines look like `6 0.795410 0.758464 0.283203 0.373698` (class 0–6, four floats in [0,1]).

- [ ] **Step 6: Commit**

```bash
git add scripts/convert_coco_to_yolo.py tests/test_convert_coco_to_yolo.py
git commit -m "feat: COCO->YOLO conversion with deterministic image-level split"
```

---

### Task 3: Real-data ingest (download, remap, split, spot-check)

**Files:**
- Create: `scripts/prepare_real_data.py`
- Test: `tests/test_prepare_real_data.py`

**Interfaces:**
- Consumes: env var `ROBOFLOW_API_KEY`; `.venv` from Task 1; `CLASS_NAMES` convention.
- Produces:
  - `data/real/finetune/{images,labels}/` and `data/real/test/{images,labels}/` in YOLO format, classes already remapped to the 7-class Global-Constraints order (real images will simply never use index 4 `d10_pct`)
  - `data/real/real.yaml` (train=finetune, val=test) and `data/real/spotcheck/*.jpg` (label overlays for human review)
  - Importable: `remap_label_line(line: str, src_names: list[str]) -> str | None` (returns remapped YOLO line, or None to drop the box), `SOURCE_NAME_MAP: dict[str, str | None]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prepare_real_data.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_real_data import remap_label_line, SOURCE_NAME_MAP


def test_known_names_map_to_our_taxonomy():
    for name in ["d4", "d6", "d8", "d10", "d12", "d20"]:
        assert SOURCE_NAME_MAP[name] == name


def test_percentile_variants_merge_to_d10():
    for name in ["d10%", "d100", "d10_percentile"]:
        assert SOURCE_NAME_MAP.get(name, "d10") == "d10"


def test_remap_label_line_translates_class_index():
    # source dataset ordered [d20, d4]: class 0 = d20 -> our index 6
    out = remap_label_line("0 0.5 0.5 0.2 0.2", ["d20", "d4"])
    assert out.split()[0] == "6"
    assert out.split()[1:] == ["0.5", "0.5", "0.2", "0.2"]


def test_remap_label_line_drops_unmappable():
    assert remap_label_line("0 0.5 0.5 0.2 0.2", ["coin"]) is None


def test_remap_label_line_unknown_name_raises():
    with pytest.raises(KeyError):
        remap_label_line("0 0.5 0.5 0.2 0.2", ["mystery_die"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && .venv/bin/pytest tests/test_prepare_real_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'prepare_real_data'`

- [ ] **Step 3: Write the script**

Create `scripts/prepare_real_data.py`:

```python
#!/usr/bin/env python3
"""Download public real-photo dice datasets from Roboflow, remap labels to
this repo's 7-class taxonomy, split into finetune + frozen test, and render
spot-check overlays for human label-quality review.

Requires ROBOFLOW_API_KEY in the environment. Never commit the key.
"""
import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import yaml

CLASS_NAMES = ["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"]

# source class name (lowercased) -> our class name; None = drop the box.
# Real sets can't distinguish percentile d10s reliably -> merge into d10.
SOURCE_NAME_MAP = {
    "d4": "d4", "d6": "d6", "d8": "d8", "d10": "d10", "d12": "d12",
    "d20": "d20",
    "d10%": "d10", "d100": "d10", "d10_percentile": "d10", "d10-percentile": "d10",
    "dice": None, "die": None, "coin": None,
}

SOURCES = [
    {"workspace": "sunib-p9bq2", "project": "dnd-dices-ycaiz", "slug": "dnd_dices"},
    {"workspace": "thomas-phillips-t0qi6", "project": "d-d-dice-detection", "slug": "dd_dice"},
]


def remap_label_line(line, src_names):
    """Remap one YOLO label line from a source dataset's class order to ours.

    Returns the remapped line, None to drop the box, or raises KeyError on a
    source class name we have no explicit decision for (forces a human call).
    """
    parts = line.split()
    src_name = src_names[int(parts[0])].lower()
    our_name = SOURCE_NAME_MAP[src_name]  # KeyError on purpose if unknown
    if our_name is None:
        return None
    return " ".join([str(CLASS_NAMES.index(our_name))] + parts[1:])


def download_source(src, dest_root):
    from roboflow import Roboflow
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace(src["workspace"]).project(src["project"])
    versions = project.versions()
    version = max(versions, key=lambda v: int(str(v.version).split("/")[-1]))
    loc = str(dest_root / "raw" / src["slug"])
    version.download("yolov8", location=loc, overwrite=False)
    return Path(loc)


def collect_pairs(raw_dir):
    """Yield (img_path, label_path) across the download's train/valid/test dirs."""
    for split_dir in raw_dir.iterdir():
        img_dir = split_dir / "images"
        if not img_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            lbl = split_dir / "labels" / (img.stem + ".txt")
            if lbl.exists():
                yield img, lbl


def spotcheck(img_path, label_path, out_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    for line in label_path.read_text().splitlines():
        c, cx, cy, bw, bh = line.split()
        cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
        p1 = (int(cx - bw / 2), int(cy - bh / 2))
        p2 = (int(cx + bw / 2), int(cy + bh / 2))
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.putText(img, CLASS_NAMES[int(c)], (p1[0], max(p1[1] - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(str(out_path), img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/real"))
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--spotcheck-n", type=int, default=20)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for sub in ("finetune/images", "finetune/labels", "test/images",
                "test/labels", "spotcheck"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    total = {"finetune": 0, "test": 0}
    for src in SOURCES:
        raw = download_source(src, args.out)
        data_yaml = yaml.safe_load((raw / "data.yaml").read_text())
        src_names = data_yaml["names"]
        print(f"[{src['slug']}] source classes: {src_names}")

        pairs = list(collect_pairs(raw))
        rng.shuffle(pairs)
        n_test = round(len(pairs) * args.test_frac)
        for i, (img, lbl) in enumerate(pairs):
            split = "test" if i < n_test else "finetune"
            remapped = [
                r for line in lbl.read_text().splitlines() if line.strip()
                if (r := remap_label_line(line, src_names)) is not None
            ]
            if not remapped:
                continue  # image with no mappable boxes: skip entirely
            stem = f"{src['slug']}_{img.stem}"
            shutil.copy(img, args.out / split / "images" / f"{stem}{img.suffix}")
            (args.out / split / "labels" / f"{stem}.txt").write_text(
                "\n".join(remapped) + "\n")
            total[split] += 1

        # spot-check overlays from this source's finetune slice
        done = 0
        for img in sorted((args.out / "finetune" / "images").glob(f"{src['slug']}_*")):
            if done >= args.spotcheck_n:
                break
            lbl = args.out / "finetune" / "labels" / (img.stem + ".txt")
            spotcheck(img, lbl, args.out / "spotcheck" / img.name)
            done += 1

    yaml_text = (
        f"path: {args.out.resolve()}\n"
        "train: finetune/images\n"
        "val: test/images\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (args.out / "real.yaml").write_text(yaml_text)
    print(f"[DONE] finetune={total['finetune']} test={total['test']}")


if __name__ == "__main__":
    main()
```

Note the walrus-in-comprehension in `main()` — if the reviewer prefers, an explicit loop is equally fine; behavior is what the tests pin down.

- [ ] **Step 4: Run unit tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && .venv/bin/pytest tests/test_prepare_real_data.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the real download**

```bash
cd /home/saps/projects/Dice-Detection
ROBOFLOW_API_KEY=<provided at execution time> .venv/bin/python scripts/prepare_real_data.py
```

Expected: prints each source's class list (verify every printed name appears in `SOURCE_NAME_MAP` — a KeyError here means a new name needs an explicit mapping decision; add it to `SOURCE_NAME_MAP` with a comment and re-run), then `[DONE] finetune=~5500 test=~1380` (exact counts depend on current dataset versions; test must be ≥300 per the spec).

- [ ] **Step 6: Human spot-check gate**

Send `data/real/spotcheck/` overlays (up to 40 images) to the user / review them visually: boxes tight around dice, class names plausible. Record the verdict in the commit message. If a source's labels are garbage, drop that source from `SOURCES` and re-run — the spec allows substitution.

- [ ] **Step 7: Commit**

```bash
git add scripts/prepare_real_data.py tests/test_prepare_real_data.py
git commit -m "feat: real dice dataset ingest with class remap + spot-check overlays"
```

---

### Task 4: Training script (fills the empty `scripts/train.py` stub)

**Files:**
- Modify: `scripts/train.py` (currently 0 bytes)
- Create: `data/yolo/dice_s_plus_r.yaml` (generated by the script, gitignored)

**Interfaces:**
- Consumes: `data/yolo/dice.yaml` (Task 2), `data/real/` (Task 3).
- Produces: `models/runs/<variant>/weights/best.pt` for `variant ∈ {s, s_plus_r}`; CLI `python scripts/train.py --variant {s,s_plus_r} [--epochs 60] [--batch 32] [--imgsz 640]`.

- [ ] **Step 1: Write `scripts/train.py`**

```python
#!/usr/bin/env python3
"""Train YOLO11s dice detector.

Variants (spec 2026-07-10-dice-detector-v1):
  s        - synthetic detection_v1 only
  s_plus_r - synthetic + real finetune slice mixed into training
Both validate on the synthetic val split during training (monitoring only);
the frozen real test set is never seen here.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = ["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"]
REPO = Path(__file__).resolve().parents[1]


def build_s_plus_r_yaml():
    out = REPO / "data/yolo/dice_s_plus_r.yaml"
    out.write_text(
        "path: .\n"
        "train:\n"
        f"  - {(REPO / 'data/yolo/images/train').resolve()}\n"
        f"  - {(REPO / 'data/real/finetune/images').resolve()}\n"
        f"val: {(REPO / 'data/yolo/images/val').resolve()}\n"
        "names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["s", "s_plus_r"], required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    data = (REPO / "data/yolo/dice.yaml" if args.variant == "s"
            else build_s_plus_r_yaml())
    model = YOLO("yolo11s.pt")
    model.train(
        data=str(data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        seed=42,
        deterministic=True,
        project=str(REPO / "models/runs"),
        name=args.variant,
        exist_ok=True,
        device=0,
    )
    print(f"[DONE] best weights: models/runs/{args.variant}/weights/best.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run 1 epoch on the real dataset config**

```bash
cd /home/saps/projects/Dice-Detection
.venv/bin/python scripts/train.py --variant s --epochs 1 --batch 16
```

Expected: completes one epoch (~3-6 min), writes `models/runs/s/weights/best.pt`, per-class val table prints all 7 class names. This validates the data pipeline end-to-end before the long runs. (An OOM here → retry with `--batch 8` and note it for Task 5/6.)

- [ ] **Step 3: Commit**

```bash
git add scripts/train.py
git commit -m "feat: YOLO11s training script with s / s_plus_r variants"
```

---

### Task 5: Full training run — variant S (synthetic only)

**Files:**
- Create: `models/runs/s/` (gitignored artifacts), `docs/results/2026-07-dice-detector-v1/train_s.md` (committed summary)

**Interfaces:**
- Consumes: Task 4 CLI.
- Produces: `models/runs/s/weights/best.pt` used by Task 7.

- [ ] **Step 1: Launch full S training (background, long-running)**

```bash
cd /home/saps/projects/Dice-Detection
nohup .venv/bin/python scripts/train.py --variant s > models/runs/train_s.log 2>&1 &
```

Expected duration: roughly 1.5–3 h (60 epochs × ~2-3 min). Monitor via `tail models/runs/train_s.log`; watch for NaN losses or collapsing mAP (abort + halve batch/LR if so).

- [ ] **Step 2: Verify completion**

```bash
tail -5 models/runs/train_s.log
ls -la models/runs/s/weights/best.pt
```

Expected: log shows `[DONE] best weights: ...` and final per-class metrics; `best.pt` exists and is >15 MB.

- [ ] **Step 3: Write and commit the run summary**

Create `docs/results/2026-07-dice-detector-v1/train_s.md` containing: final epoch, overall and per-class mAP50/mAP50-95 on synthetic val (copy from log), training wall time, batch size actually used.

```bash
git add docs/results/2026-07-dice-detector-v1/train_s.md
git commit -m "results: variant S (synthetic-only) training run"
```

---

### Task 6: Full training run — variant S+R

**Files:**
- Create: `models/runs/s_plus_r/` (gitignored), `docs/results/2026-07-dice-detector-v1/train_s_plus_r.md` (committed)

**Interfaces:**
- Consumes: Task 4 CLI, Task 3 real finetune slice.
- Produces: `models/runs/s_plus_r/weights/best.pt` used by Task 7.

- [ ] **Step 1: Launch full S+R training (after S finishes — never concurrent)**

```bash
cd /home/saps/projects/Dice-Detection
nohup .venv/bin/python scripts/train.py --variant s_plus_r > models/runs/train_s_plus_r.log 2>&1 &
```

- [ ] **Step 2: Verify completion**

```bash
tail -5 models/runs/train_s_plus_r.log
ls -la models/runs/s_plus_r/weights/best.pt
```

Expected: `[DONE]` line, `best.pt` >15 MB.

- [ ] **Step 3: Write and commit the run summary**

Create `docs/results/2026-07-dice-detector-v1/train_s_plus_r.md` (same fields as Task 5's summary).

```bash
git add docs/results/2026-07-dice-detector-v1/train_s_plus_r.md
git commit -m "results: variant S+R (synthetic + real finetune) training run"
```

---

### Task 7: Evaluation script (fills the empty `scripts/evaluate.py` stub)

**Files:**
- Modify: `scripts/evaluate.py` (currently 0 bytes)
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `models/runs/<variant>/weights/best.pt`, `data/yolo/dice.yaml`, `data/real/test/`.
- Produces:
  - CLI `python scripts/evaluate.py --weights <path> --name <variant>` writing `docs/results/2026-07-dice-detector-v1/eval_<variant>.md` and `models/eval/<variant>/overlays/*.jpg`
  - Importable: `merge_d10(class_idx: int) -> int` (maps `d10_pct` index 4 → `d10` index 3, identity otherwise) and `eval_real(pred_records, gt_records, class_names) -> dict[str, tuple[float, float]]` returning per-class `(mAP50, mAP50_95)` in the merged 6-class space.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_evaluate.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate import merge_d10, eval_real, MERGED_CLASS_NAMES


def test_merge_maps_d10_pct_onto_d10():
    # CLASS_NAMES: d4=0 d6=1 d8=2 d10=3 d10_pct=4 d12=5 d20=6
    assert merge_d10(4) == 3
    assert [merge_d10(i) for i in [0, 1, 2, 3, 5, 6]] == [0, 1, 2, 3, 4, 5]


def test_merged_class_names():
    assert MERGED_CLASS_NAMES == ["d4", "d6", "d8", "d10", "d12", "d20"]


def test_eval_real_perfect_predictions_score_1():
    # one image, one d20 GT box; prediction identical with conf 0.9
    gt = [{"image_id": 1, "class_idx": 6, "bbox": [10, 10, 50, 50],
           "width": 640, "height": 480}]
    preds = [{"image_id": 1, "class_idx": 6, "bbox": [10, 10, 50, 50],
              "score": 0.9}]
    per_class = eval_real(preds, gt, MERGED_CLASS_NAMES)
    map50, map5095 = per_class["d20"]
    assert map50 == pytest.approx(1.0)
    assert map5095 == pytest.approx(1.0)


def test_eval_real_merges_pct_prediction_onto_d10_gt():
    # GT says d10 (merged idx 3); model predicted d10_pct (raw idx 4).
    # After merge they must count as a match.
    gt = [{"image_id": 1, "class_idx": 3, "bbox": [10, 10, 50, 50],
           "width": 640, "height": 480}]
    preds = [{"image_id": 1, "class_idx": merge_d10(4),
              "bbox": [10, 10, 50, 50], "score": 0.9}]
    per_class = eval_real(preds, gt, MERGED_CLASS_NAMES)
    assert per_class["d10"][0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/saps/projects/Dice-Detection && .venv/bin/pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evaluate'`

- [ ] **Step 3: Write the script**

Create `scripts/evaluate.py`:

```python
#!/usr/bin/env python3
"""Evaluate a trained dice detector on (a) the synthetic val split and
(b) the frozen real test set, with d10_pct merged into d10 for (b).

Real-set metrics are computed via pycocotools in the merged 6-class space;
synthetic metrics come from ultralytics' own val() in the full 7-class space.
"""
import argparse
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASS_NAMES = ["d4", "d6", "d8", "d10", "d10_pct", "d12", "d20"]
MERGED_CLASS_NAMES = ["d4", "d6", "d8", "d10", "d12", "d20"]
REPO = Path(__file__).resolve().parents[1]


def merge_d10(class_idx):
    """Map 7-class index to merged 6-class index (d10_pct -> d10)."""
    if class_idx == 4:
        return 3
    return class_idx if class_idx < 4 else class_idx - 1


def eval_real(pred_records, gt_records, class_names):
    """Per-class (mAP50, mAP50-95) via COCOeval.

    gt_records: [{image_id, class_idx (merged), bbox [x,y,w,h] px, width, height}]
    pred_records: [{image_id, class_idx (merged), bbox [x,y,w,h] px, score}]
    """
    images = {}
    for r in gt_records:
        images[r["image_id"]] = {"id": r["image_id"],
                                 "width": r["width"], "height": r["height"]}
    gt = {
        "images": list(images.values()),
        "annotations": [
            {"id": i + 1, "image_id": r["image_id"],
             "category_id": r["class_idx"] + 1, "bbox": r["bbox"],
             "area": r["bbox"][2] * r["bbox"][3], "iscrowd": 0}
            for i, r in enumerate(gt_records)
        ],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
    }
    coco_gt = COCO()
    coco_gt.dataset = gt
    coco_gt.createIndex()
    dets = [
        {"image_id": r["image_id"], "category_id": r["class_idx"] + 1,
         "bbox": r["bbox"], "score": r["score"]}
        for r in pred_records
    ]
    results = {}
    for i, name in enumerate(class_names):
        ev = COCOeval(coco_gt, coco_gt.loadRes(dets) if dets else None, "bbox")
        ev.params.catIds = [i + 1]
        ev.evaluate(); ev.accumulate(); ev.summarize()
        # stats[1] = mAP50, stats[0] = mAP50-95 (COCO summarize convention)
        results[name] = (max(ev.stats[1], 0.0), max(ev.stats[0], 0.0))
    return results


def yolo_label_to_gt(label_path, image_id, w, h):
    recs = []
    for line in label_path.read_text().splitlines():
        c, cx, cy, bw, bh = (float(v) for v in line.split())
        recs.append({
            "image_id": image_id,
            "class_idx": merge_d10(int(c)),
            "bbox": [(cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h],
            "width": w, "height": h,
        })
    return recs


def main():
    from ultralytics import YOLO

    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--name", required=True, help="variant label, e.g. s")
    ap.add_argument("--overlay-n", type=int, default=30)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    model = YOLO(str(args.weights))
    out_md = REPO / f"docs/results/2026-07-dice-detector-v1/eval_{args.name}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    overlay_dir = REPO / f"models/eval/{args.name}/overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    # (a) synthetic val, full 7-class space, ultralytics metrics
    synth = model.val(data=str(REPO / "data/yolo/dice.yaml"), split="val")
    synth_rows = [
        f"| {CLASS_NAMES[c]} | {synth.box.ap50[i]:.3f} | {synth.box.ap[i]:.3f} |"
        for i, c in enumerate(synth.box.ap_class_index)
    ]

    # (b) frozen real test, merged 6-class space
    test_images = sorted((REPO / "data/real/test/images").iterdir())
    gt_records, pred_records = [], []
    for image_id, img_path in enumerate(test_images, start=1):
        res = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        h, w = res.orig_shape
        lbl = REPO / "data/real/test/labels" / (img_path.stem + ".txt")
        gt_records += yolo_label_to_gt(lbl, image_id, w, h)
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            pred_records.append({
                "image_id": image_id,
                "class_idx": merge_d10(int(box.cls)),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(box.conf),
            })
        if image_id <= args.overlay_n:
            res.save(filename=str(overlay_dir / img_path.name))

    real = eval_real(pred_records, gt_records, MERGED_CLASS_NAMES)
    real_rows = [f"| {n} | {m50:.3f} | {m5095:.3f} |"
                 for n, (m50, m5095) in real.items()]

    out_md.write_text(
        f"# Eval: {args.name}\n\nWeights: `{args.weights}`\n\n"
        "## Synthetic val (7 classes — optimistic, see spec amendment)\n\n"
        "| class | mAP50 | mAP50-95 |\n|---|---|---|\n"
        + "\n".join(synth_rows) +
        "\n\n## Real test (frozen, d10_pct merged into d10)\n\n"
        "| class | mAP50 | mAP50-95 |\n|---|---|---|\n"
        + "\n".join(real_rows) +
        f"\n\nOverlays: `models/eval/{args.name}/overlays/` "
        f"({min(len(test_images), args.overlay_n)} images)\n"
    )
    print(f"[DONE] wrote {out_md}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/saps/projects/Dice-Detection && .venv/bin/pytest tests/test_evaluate.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate.py tests/test_evaluate.py
git commit -m "feat: evaluation with pycocotools real-set metrics and d10_pct merge"
```

---

### Task 8: Run evaluations + results write-up

**Files:**
- Create: `docs/results/2026-07-dice-detector-v1/eval_s.md`, `eval_s_plus_r.md`, `summary.md` (all committed); overlay jpgs under `models/eval/` (gitignored, shared with user directly)

**Interfaces:**
- Consumes: Task 7 CLI, both `best.pt` weights.

- [ ] **Step 1: Evaluate both variants**

```bash
cd /home/saps/projects/Dice-Detection
.venv/bin/python scripts/evaluate.py --weights models/runs/s/weights/best.pt --name s
.venv/bin/python scripts/evaluate.py --weights models/runs/s_plus_r/weights/best.pt --name s_plus_r
```

Expected: two `[DONE]` lines; both eval md files contain filled 7-row synthetic and 6-row real tables (no `nan`).

Then produce confusion matrices on the real test set (7-class, un-merged — ultralytics saves `confusion_matrix.png` in the val run dir; the synthetic-val ones were already saved during training):

```bash
.venv/bin/yolo val model=models/runs/s/weights/best.pt data=data/real/real.yaml project=models/eval/s name=real_cm
.venv/bin/yolo val model=models/runs/s_plus_r/weights/best.pt data=data/real/real.yaml project=models/eval/s_plus_r name=real_cm
```

- [ ] **Step 2: Visual verification**

Review `models/eval/s/overlays/` and `models/eval/s_plus_r/overlays/` (30 real-photo overlays each) and send a representative sample to the user. Look for: missed dice, wrong classes (esp. d10 vs d12 confusion), boxes on non-dice objects.

- [ ] **Step 3: Write `summary.md`**

Create `docs/results/2026-07-dice-detector-v1/summary.md` with:
- The S vs S+R per-class real-test mAP50 table side by side, plus the synthetic-val vs real-test gap for variant S (the headline sim-to-real number).
- 3-5 concrete recommendations back to the data generator, grounded in the per-class gaps and overlay failure modes (e.g. "d4 real mAP lags 30 pts → synthetic d4s may be too large/too clean; check scale and motion blur", "render split-designated asset pools to enable leak-free synthetic val").
- Wall-clock and env facts (torch/ultralytics versions, batch used).

- [ ] **Step 4: Commit and finish**

```bash
git add docs/results/2026-07-dice-detector-v1/
git commit -m "results: dice detector v1 - S vs S+R sim-to-real evaluation"
```

Then run the full test suite once as the final gate:

```bash
.venv/bin/pytest tests/test_convert_coco_to_yolo.py tests/test_prepare_real_data.py tests/test_evaluate.py -v
```

Expected: 13 passed.
