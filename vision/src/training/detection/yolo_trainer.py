#!/usr/bin/env python3
"""Train YOLO11s dice detector.

Variants (spec 2026-07-10-dice-detector-v1):
  s        - synthetic detection_v1 only
  s_plus_r - synthetic + real finetune slice mixed into training
Both validate on the synthetic val split during training (monitoring only);
the frozen real test set is never seen here.

Moved from scripts/train.py (2026-07-10 platform refactor, see
docs/superpowers/specs/2026-07-10-training-platform-architecture-design.md)
-- scripts/train.py is now a thin CLI over this module. Behavior is
unchanged; CLASS_NAMES now comes from configs/domains/dice.yaml. This is
the first (and, for now, only) Trainer-protocol implementation the platform
spec describes -- a future non-detection model family would be a sibling
package under src/training/, not a change to this one.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

from datasets.domain_config import load_domain_config

CLASS_NAMES = list(load_domain_config("dice")["classes"])
REPO = Path(__file__).resolve().parents[3]


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
    # Optional run-name override (defaults to --variant, i.e. every
    # pre-refactor invocation is unaffected): lets a smoke/regression-check
    # run write to e.g. models/runs/s_refactor_check/ instead of clobbering
    # models/runs/<variant>/, which holds the committed-results-backing
    # 60-epoch weights (see docs/results/2026-07-dice-detector-v1/).
    ap.add_argument("--name", default=None,
                     help="run name under models/runs/ (default: --variant)")
    args = ap.parse_args()
    run_name = args.name or args.variant

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
        name=run_name,
        exist_ok=True,
        device=0,
    )
    print(f"[DONE] best weights: models/runs/{run_name}/weights/best.pt")


if __name__ == "__main__":
    main()
