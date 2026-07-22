"""Extract final/max values of named TensorBoard scalars from a run dir.

Run under Isaac Lab's python (it has tensorboard available):

    /home/saps/IsaacLab/isaaclab.sh -p sweeps/_extract_scalars.py <run_dir> <tag> [<tag> ...]

Prints one tab-separated line per tag: ``<tag>\tLAST\t<last>\tMAX\t<max>``,
or ``<tag>\tNOT_FOUND``. A dedicated file (not an inline ``-p -c`` snippet)
per senior-agent.md's note that inline snippets have hung reproducibly.
"""

import glob
import sys

from tensorboard.backend.event_processing import event_accumulator


def main() -> None:
    run_dir = sys.argv[1]
    tags = sys.argv[2:]
    paths = sorted(glob.glob(run_dir.rstrip("/") + "/events.out.tfevents.*"))
    if not paths:
        print("NO_EVENT_FILE")
        return
    ea = event_accumulator.EventAccumulator(paths[-1])
    ea.Reload()
    available = ea.Tags()["scalars"]
    for tag in tags:
        if tag in available:
            vals = ea.Scalars(tag)
            print(f"{tag}\tLAST\t{vals[-1].value}\tMAX\t{max(v.value for v in vals)}")
        else:
            print(f"{tag}\tNOT_FOUND")


if __name__ == "__main__":
    main()
