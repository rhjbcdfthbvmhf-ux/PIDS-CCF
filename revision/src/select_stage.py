#!/usr/bin/env python3
"""Select stage-1 or stage-2 checkpoint using validation evidence only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, required=True)
    parser.add_argument("--stage2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = []
    for stage, run_dir in ((1, args.stage1.resolve()), (2, args.stage2.resolve())):
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        candidates.append(
            {
                "stage": stage,
                "run_dir": str(run_dir),
                "best_validation_score": float(metrics["best_validation_score"]),
                "best_epoch": int(metrics["best_epoch"]),
                "checkpoint_sha256": digest(run_dir / "best.pt"),
                "split_sha256": config["split_sha256"],
            }
        )
    if candidates[0]["split_sha256"] != candidates[1]["split_sha256"]:
        raise ValueError("stage candidates use different splits")
    selected = max(
        candidates,
        key=lambda row: (row["best_validation_score"], -row["stage"]),
    )
    output = {
        "selection_scope": "validation only; test metrics are not consulted",
        "tie_break": "prefer stage 1",
        "candidates": candidates,
        "selected_stage": selected["stage"],
        "selected_run_dir": selected["run_dir"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
