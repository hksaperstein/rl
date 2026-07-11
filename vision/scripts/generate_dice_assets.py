import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datagen.domains.dice import orchestrator


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    parser = argparse.ArgumentParser(description="Generate a library of dice USD assets.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--count", type=int, default=None)
    group.add_argument("--sets", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="data/raw/dice_assets")
    args = parser.parse_args(argv)

    if args.sets is not None:
        generated, failed = orchestrator.generate_set_batch(args.sets, args.seed, args.outdir)
    elif args.count is not None:
        generated, failed = orchestrator.generate_batch(args.count, args.seed, args.outdir)
    else:
        generated, failed = orchestrator.generate_batch(100, args.seed, args.outdir)
    print(f"Generated: {generated}, Failed: {failed}")


if __name__ == "__main__":
    main()
