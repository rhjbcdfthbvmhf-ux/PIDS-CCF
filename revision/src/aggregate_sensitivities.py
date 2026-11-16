#!/usr/bin/env python3
"""Aggregate ablation and full patch-no-overlap sensitivity experiments."""

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
BUFFERED_DATASETS = ("PaviaU", "Salinas")
SEEDS = (23, 37, 51, 71, 101)
CONDITIONS = ("Clean", "L1", "L2", "L3", "L4", "L5")
METRICS = ("oa", "aa", "kappa", "macro_f1", "worst_class_f1", "ece", "nll")
ABLATIONS = ("uniform_risk", "random_subsets", "spectral_only", "patch7", "patch15")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_metrics(selection_path: Path) -> tuple[int, dict]:
    selection = read_json(selection_path)
    metrics = read_json(Path(selection["selected_run_dir"]) / "metrics.json")
    candidates = selection["candidates"]
    if {int(row["stage"]) for row in candidates} != {1, 2}:
        raise AssertionError(f"two complete candidates required: {selection_path}")
    winner = max(candidates, key=lambda row: (row["best_validation_score"], -row["stage"]))
    if (int(selection["selected_stage"]) != int(winner["stage"])
            or selection["selected_run_dir"] != winner["run_dir"]
            or metrics["checkpoint_sha256"] != winner["checkpoint_sha256"]):
        raise AssertionError(f"validation selection mismatch: {selection_path}")
    for candidate in candidates:
        candidate_metrics = read_json(Path(candidate["run_dir"]) / "metrics.json")
        if (candidate_metrics["evaluation_parameters"] != metrics["evaluation_parameters"]
                or candidate_metrics["best_validation_score"] != candidate["best_validation_score"]):
            raise AssertionError(f"candidate metadata mismatch: {selection_path}")
    return int(selection["selected_stage"]), metrics


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    count = len(array)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    half = float(t.ppf(0.975, count - 1) * std / math.sqrt(count)) if count > 1 else 0.0
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "values": array.tolist(),
    }


def aggregate_records(records: dict, datasets: tuple[str, ...], methods: tuple[str, ...]) -> dict:
    aggregate: dict[str, dict] = {}
    for dataset in datasets:
        aggregate[dataset] = {}
        for method in methods:
            aggregate[dataset][method] = {}
            for condition in CONDITIONS:
                aggregate[dataset][method][condition] = {
                    metric: summarize(
                        [records[dataset][str(seed)][method][condition][metric] for seed in SEEDS]
                    )
                    for metric in METRICS
                }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.server_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    ablation_records: dict[str, dict] = {}
    for dataset in DATASETS:
        ablation_records[dataset] = {}
        for seed in SEEDS:
            core_stage, core = selected_metrics(
                root / "runs_final_selection" / dataset / f"seed_{seed}" / "selection.json"
            )
            row: dict[str, object] = {
                "selected_stages": {"core": core_stage},
                "core": core["results"],
            }
            for tag in ABLATIONS:
                stage, metrics = selected_metrics(
                    root / "runs_final_ablations" / tag / "selection"
                    / dataset / f"seed_{seed}" / "selection.json"
                )
                row["selected_stages"][tag] = stage
                if metrics["evaluation_parameters"] != core["evaluation_parameters"]:
                    raise AssertionError(f"unmatched evaluation parameters: {dataset}/{seed}/{tag}")
                row[tag] = metrics["results"]
            spectral_clean = read_json(
                root / "runs_final_ablations" / "spectral_only" / "clean_stage"
                / dataset / "clean" / f"seed_{seed}" / "metrics.json"
            )
            row["spectral_only_clean"] = spectral_clean["results"]
            ablation_records[dataset][str(seed)] = row

    ablation_methods = ("core", *ABLATIONS, "spectral_only_clean")
    ablation_aggregate = aggregate_records(ablation_records, DATASETS, ablation_methods)

    buffered_records: dict[str, dict] = {}
    for dataset in BUFFERED_DATASETS:
        buffered_records[dataset] = {}
        for seed in SEEDS:
            clean = read_json(
                root / "runs_buffered_full_no_overlap" / "clean_stage"
                / dataset / "clean" / f"seed_{seed}" / "metrics.json"
            )
            stage, robust = selected_metrics(
                root / "runs_buffered_full_no_overlap" / "selection"
                / dataset / f"seed_{seed}" / "selection.json"
            )
            if robust["evaluation_parameters"] != clean["evaluation_parameters"]:
                raise AssertionError(f"unmatched no-overlap evaluation: {dataset}/{seed}")
            split_path = (root / "runs_buffered_full_no_overlap" / "clean_stage"
                          / dataset / "clean" / f"seed_{seed}" / "split.json")
            split = read_json(split_path)
            buffered_records[dataset][str(seed)] = {
                "selected_stage": stage,
                "partition_counts": {key: len(split[key]) for key in ("train", "val", "test")},
                "clean": clean["results"],
                "pids_ccf": robust["results"],
            }
    buffered_aggregate = aggregate_records(
        buffered_records, BUFFERED_DATASETS, ("clean", "pids_ccf")
    )

    paired_ablations = {}
    for dataset in DATASETS:
        paired_ablations[dataset] = {
            tag: {condition: paired_statistics(ablation_aggregate[dataset]["core"][condition]["oa"]["values"],
                                               ablation_aggregate[dataset][tag][condition]["oa"]["values"])
                  for condition in CONDITIONS} for tag in ABLATIONS}
        corrected = holm_adjust([paired_ablations[dataset][tag]["L5"]["wilcoxon_p"] for tag in ABLATIONS])
        for tag, value in zip(ABLATIONS, corrected):
            paired_ablations[dataset][tag]["L5"]["holm_p_across_ablations"] = value
    paired_buffered = {
        dataset: {condition: paired_statistics(buffered_aggregate[dataset]["pids_ccf"][condition]["oa"]["values"],
                                               buffered_aggregate[dataset]["clean"][condition]["oa"]["values"])
                  for condition in CONDITIONS} for dataset in BUFFERED_DATASETS}

    payload = {
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "statistical_unit": "independent training seed",
        "ablations": {
            "records": ablation_records,
            "aggregate": ablation_aggregate,
            "secondary_paired_core_vs_ablation": paired_ablations,
        },
        "buffered_sensitivity": {
            "datasets": BUFFERED_DATASETS,
            "patch_size": 11,
            "buffer_radius": 10,
            "overlap_guarantee": (
                "No pixel can occur in both a development-centered and test-centered 11x11 patch."
            ),
            "indian_pines_exclusion": (
                "A class-complete radius-10 test set is impossible under the locked compact-development "
                "construction because several rare classes have no remaining test center."
            ),
            "records": buffered_records,
            "aggregate": buffered_aggregate,
            "paired_selected_vs_clean_within_buffered_split": paired_buffered,
            "comparison_boundary": "Random and buffered splits use different test populations; their absolute difference is descriptive, not an isolated leakage effect.",
        },
    }
    (output / "sensitivity_aggregate.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    with (output / "sensitivity_long.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["analysis", "dataset", "method", "condition", "metric", "mean", "std", "ci95_low", "ci95_high"]
        )
        for dataset in DATASETS:
            for method in ablation_methods:
                for condition in CONDITIONS:
                    for metric in METRICS:
                        row = ablation_aggregate[dataset][method][condition][metric]
                        writer.writerow(
                            ["ablation", dataset, method, condition, metric,
                             row["mean"], row["std"], row["ci95_low"], row["ci95_high"]]
                        )
        for dataset in BUFFERED_DATASETS:
            for method in ("clean", "pids_ccf"):
                for condition in CONDITIONS:
                    for metric in METRICS:
                        row = buffered_aggregate[dataset][method][condition][metric]
                        writer.writerow(
                            ["buffered", dataset, method, condition, metric,
                             row["mean"], row["std"], row["ci95_low"], row["ci95_high"]]
                        )

    print(output / "sensitivity_aggregate.json")


if __name__ == "__main__":
    main()
