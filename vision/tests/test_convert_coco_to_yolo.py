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
