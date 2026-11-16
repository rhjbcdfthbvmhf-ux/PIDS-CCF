#!/usr/bin/env python3
"""Aggregate the matched clean/PIDS baseline tracks over final seeds."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

from aggregate_final import paired_statistics, holm_adjust


DATASETS = ("IndianPines", "PaviaU", "Salinas")
SEEDS = (23, 37, 51, 71, 101)
CONDITIONS = ("Clean", "L1", "L2", "L3", "L4", "L5")
BACKBONES = ("mlp", "spectral_cnn", "hybrid3d", "spatial_resnet")
METRICS = ("oa", "aa", "kappa", "macro_f1", "worst_class_f1", "ece", "nll")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    count = len(array)
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    half = float(t.ppf(0.975, count - 1) * std / math.sqrt(count)) if count > 1 else 0.0
    mean = float(array.mean())
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "values": array.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.server_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    records: dict[str, dict] = {}
    for dataset in DATASETS:
        records[dataset] = {}
        for backbone in BACKBONES:
            records[dataset][backbone] = {}
            for seed in SEEDS:
                clean_path = (
                    root / "runs_final_baselines" / backbone / "clean_stage"
                    / dataset / "clean" / f"seed_{seed}" / "metrics.json"
                )
                selection_path = (
                    root / "runs_final_baselines" / backbone / "selection"
                    / dataset / f"seed_{seed}" / "selection.json"
                )
                clean = read_json(clean_path)
                selection = read_json(selection_path)
                selected = read_json(Path(selection["selected_run_dir"]) / "metrics.json")
                stage1 = read_json(root / "runs_final_baselines" / backbone / "pids_stage1" / dataset / "pids" / f"seed_{seed}" / "metrics.json")
                stage2 = read_json(root / "runs_final_baselines" / backbone / "pids_stage2" / dataset / "pids" / f"seed_{seed}" / "metrics.json")
                if any(m["evaluation_parameters"] != clean["evaluation_parameters"] for m in (stage1, stage2)):
                    raise AssertionError(f"unmatched degradation draws: {dataset}/{backbone}/{seed}")
                winner = max(selection["candidates"], key=lambda row: (row["best_validation_score"], -row["stage"]))
                if int(selection["selected_stage"]) != int(winner["stage"]):
                    raise AssertionError("stage selection does not match validation scores")
                records[dataset][backbone][str(seed)] = {
                    "selected_stage": int(selection["selected_stage"]),
                    "clean_end_to_end_seconds": float(clean["training_seconds"]),
                    "pids_two_stage_selection_seconds": float(stage1["training_seconds"]) + float(stage2["training_seconds"]),
                    "clean": clean["results"],
                    "pids": selected["results"],
                }

    aggregate: dict[str, dict] = {}
    for dataset in DATASETS:
        aggregate[dataset] = {}
        for backbone in BACKBONES:
            aggregate[dataset][backbone] = {}
            for regime in ("clean", "pids"):
                aggregate[dataset][backbone][regime] = {}
                for condition in CONDITIONS:
                    aggregate[dataset][backbone][regime][condition] = {
                        metric: summarize(
                            [
                                records[dataset][backbone][str(seed)][regime][condition][metric]
                                for seed in SEEDS
                            ]
                        )
                        for metric in METRICS
                    }

    runtime = {
        dataset: {
            backbone: {regime: summarize([records[dataset][backbone][str(seed)][key] for seed in SEEDS])
                       for regime, key in (("clean", "clean_end_to_end_seconds"), ("pids", "pids_two_stage_selection_seconds"))}
            for backbone in BACKBONES}
        for dataset in DATASETS}
    core = read_json(root / "final_aggregates" / "core" / "final_aggregate.json")
    paired = {}
    for dataset in DATASETS:
        paired[dataset] = {}
        for backbone in BACKBONES:
            paired[dataset][backbone] = {
                condition: paired_statistics(core["aggregate"][dataset]["selected_core"][condition]["oa"]["values"],
                                             aggregate[dataset][backbone]["pids"][condition]["oa"]["values"])
                for condition in CONDITIONS}
        corrected = holm_adjust([paired[dataset][backbone]["L5"]["wilcoxon_p"] for backbone in BACKBONES])
        for backbone, p in zip(BACKBONES, corrected):
            paired[dataset][backbone]["L5"]["holm_p_across_backbones"] = p
    payload = {
        "datasets": DATASETS,
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "backbones": BACKBONES,
        "statistical_unit": "independent training seed",
        "baseline_status": "in-repository reimplementations under a common protocol",
        "records": records,
        "aggregate": aggregate,
        "runtime": runtime,
        "runtime_scope": core["runtime_scope"],
        "secondary_paired_ccf_vs_pids_backbones": paired,
    }
    (output / "baseline_aggregate.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    with (output / "baseline_long.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["dataset", "backbone", "regime", "condition", "metric", "mean", "std", "ci95_low", "ci95_high"]
        )
        for dataset in DATASETS:
            for backbone in BACKBONES:
                for regime in ("clean", "pids"):
                    for condition in CONDITIONS:
                        for metric in METRICS:
                            row = aggregate[dataset][backbone][regime][condition][metric]
                            writer.writerow(
                                [dataset, backbone, regime, condition, metric,
                                 row["mean"], row["std"], row["ci95_low"], row["ci95_high"]]
                            )

    print(output / "baseline_aggregate.json")


if __name__ == "__main__":
    main()
