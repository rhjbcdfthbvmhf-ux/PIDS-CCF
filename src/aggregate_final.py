#!/usr/bin/env python3
"""Aggregate seed-level final experiments into immutable JSON/CSV/LaTeX tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t, ttest_rel


DATASETS = ("IndianPines", "PaviaU", "Salinas")
SEEDS = (23, 37, 51, 71, 101)
CONDITIONS = ("Clean", "L1", "L2", "L3", "L4", "L5")
METRICS = ("oa", "aa", "kappa", "macro_f1", "worst_class_f1", "ece", "nll")
METHODS = ("clean", "stage1", "selected_core", "routed")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def confidence_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    count = len(array)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if count > 1 else 0.0
    half_width = float(t.ppf(0.975, count - 1) * std / math.sqrt(count)) if count > 1 else 0.0
    return {
        "n": count,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "values": array.tolist(),
    }


def paired_statistics(candidate: list[float], reference: list[float]) -> dict:
    candidate_array = np.asarray(candidate, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    difference = candidate_array - reference_array
    if not np.isfinite(difference).all() or len(difference) == 0:
        raise ValueError("paired observations must be finite and nonempty")
    # Enumerate signs explicitly so zero/tied differences never silently invoke
    # a library's normal approximation under an 'exact' method label.
    nonzero = difference[difference != 0]
    if len(nonzero) > 20:
        raise ValueError("exact sign enumeration is limited to 20 nonzero pairs")
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    wilcoxon_statistic = min(positive, negative)
    observed = abs(positive - negative)
    null_values = [abs(float(np.dot(signs, ranks))) for signs in product((-1, 1), repeat=len(ranks))]
    wilcoxon_p = float(np.mean(np.asarray(null_values) >= observed - 1e-12))
    t_statistic, t_p = None, None
    if len(difference) > 1 and float(difference.std(ddof=1)) > 0:
        t_result = ttest_rel(candidate_array, reference_array)
        t_statistic, t_p = float(t_result.statistic), float(t_result.pvalue)
    return {
        "mean_difference": float(difference.mean()),
        "difference_ci": confidence_summary(difference.tolist()),
        "wins": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
        "losses": int(np.sum(difference < 0)),
        "paired_t_statistic": t_statistic,
        "paired_t_p": t_p,
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p": wilcoxon_p,
        "wilcoxon_method": "two-sided exact sign enumeration of average ranks; zero differences omitted",
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def format_percent(summary: dict) -> str:
    return f"{100 * summary['mean']:.2f} $\\pm$ {100 * summary['std']:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.server_root.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_records: dict[str, dict] = {}
    for dataset in DATASETS:
        seed_records[dataset] = {}
        for seed in SEEDS:
            clean = read_json(
                root / "runs_final_core_clean" / dataset / "clean" / f"seed_{seed}" / "metrics.json"
            )
            stage1 = read_json(
                root / "runs_final_core_stage1" / dataset / "pids" / f"seed_{seed}" / "metrics.json"
            )
            selection = read_json(
                root / "runs_final_selection" / dataset / f"seed_{seed}" / "selection.json"
            )
            selected = read_json(Path(selection["selected_run_dir"]) / "metrics.json")
            stage2 = read_json(
                root / "runs_final_core_stage2" / dataset / "pids" / f"seed_{seed}" / "metrics.json"
            )
            router = read_json(
                root / "runs_final_router" / dataset / f"seed_{seed}" / "router.json"
            )
            methods = {
                "clean": clean["results"],
                "stage1": stage1["results"],
                "selected_core": selected["results"],
                "routed": {
                    condition: router["results"][condition]["routed"]
                    for condition in CONDITIONS
                },
            }
            seed_records[dataset][str(seed)] = {
                "selected_stage": selection["selected_stage"],
                "router_enabled": router["router_enabled"],
                "expert_policy": router["expert_policy"],
                "run_metadata": {
                    "clean": {
                        "best_epoch": int(clean["best_epoch"]),
                        "end_to_end_seconds": float(clean["training_seconds"]),
                    },
                    "stage1": {
                        "best_epoch": int(stage1["best_epoch"]),
                        "end_to_end_seconds": float(stage1["training_seconds"]),
                    },
                    "selected_core": {
                        "best_epoch": int(selected["best_epoch"]),
                        "selected_stage": int(selection["selected_stage"]),
                        "stage2_end_to_end_seconds": float(stage2["training_seconds"]),
                        "two_stage_selection_seconds": float(stage1["training_seconds"]) + float(stage2["training_seconds"]),
                    },
                },
                "methods": methods,
            }

    aggregate: dict[str, dict] = {}
    for dataset in DATASETS:
        aggregate[dataset] = {}
        for method in METHODS:
            aggregate[dataset][method] = {}
            for condition in CONDITIONS:
                aggregate[dataset][method][condition] = {
                    metric: confidence_summary(
                        [
                            seed_records[dataset][str(seed)]["methods"][method][condition][metric]
                            for seed in SEEDS
                        ]
                    )
                    for metric in METRICS
                }
            clean_oa = np.asarray(
                [
                    seed_records[dataset][str(seed)]["methods"][method]["Clean"]["oa"]
                    for seed in SEEDS
                ]
            )
            l5_oa = np.asarray(
                [
                    seed_records[dataset][str(seed)]["methods"][method]["L5"]["oa"]
                    for seed in SEEDS
                ]
            )
            severity_oa = np.asarray(
                [
                    [
                        seed_records[dataset][str(seed)]["methods"][method][condition]["oa"]
                        for condition in CONDITIONS
                    ]
                    for seed in SEEDS
                ]
            )
            aggregate[dataset][method]["derived"] = {
                "l5_retention": confidence_summary((l5_oa / clean_oa).tolist()),
                "severity_auc": confidence_summary(
                    (np.trapz(severity_oa, dx=1.0, axis=1) / 5.0).tolist()
                ),
            }

    runtime = {}
    for dataset in DATASETS:
        runtime[dataset] = {
            "clean": confidence_summary(
                [
                    seed_records[dataset][str(seed)]["run_metadata"]["clean"]["end_to_end_seconds"]
                    for seed in SEEDS
                ]
            ),
            "stage1": confidence_summary(
                [
                    seed_records[dataset][str(seed)]["run_metadata"]["stage1"]["end_to_end_seconds"]
                    for seed in SEEDS
                ]
            ),
            "selected_core_total": confidence_summary(
                [
                    seed_records[dataset][str(seed)]["run_metadata"]["selected_core"]["two_stage_selection_seconds"]
                    for seed in SEEDS
                ]
            ),
        }

    comparisons = {
        "primary_selected_core_vs_clean": {},
        "secondary_selected_core_vs_stage1": {},
        "secondary_router_vs_selected_core": {},
    }
    primary_p = []
    primary_keys = []
    for dataset in DATASETS:
        comparisons["primary_selected_core_vs_clean"][dataset] = {}
        comparisons["secondary_selected_core_vs_stage1"][dataset] = {}
        comparisons["secondary_router_vs_selected_core"][dataset] = {}
        for condition in CONDITIONS:
            comparisons["primary_selected_core_vs_clean"][dataset][condition] = {}
            comparisons["secondary_selected_core_vs_stage1"][dataset][condition] = {}
            comparisons["secondary_router_vs_selected_core"][dataset][condition] = {}
            for metric in ("oa", "aa", "macro_f1", "worst_class_f1"):
                core = aggregate[dataset]["selected_core"][condition][metric]["values"]
                clean = aggregate[dataset]["clean"][condition][metric]["values"]
                stage1 = aggregate[dataset]["stage1"][condition][metric]["values"]
                routed = aggregate[dataset]["routed"][condition][metric]["values"]
                comparisons["primary_selected_core_vs_clean"][dataset][condition][metric] = paired_statistics(
                    core, clean
                )
                comparisons["secondary_selected_core_vs_stage1"][dataset][condition][metric] = paired_statistics(
                    core, stage1
                )
                comparisons["secondary_router_vs_selected_core"][dataset][condition][metric] = paired_statistics(
                    routed, core
                )
        primary_p.append(
            comparisons["primary_selected_core_vs_clean"][dataset]["L5"]["oa"]["wilcoxon_p"]
        )
        primary_keys.append(dataset)
    for dataset, adjusted in zip(primary_keys, holm_adjust(primary_p)):
        comparisons["primary_selected_core_vs_clean"][dataset]["L5"]["oa"][
            "holm_wilcoxon_p_across_datasets"
        ] = adjusted

    payload = {
        "datasets": DATASETS,
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "metrics": METRICS,
        "statistical_unit": "independent training seed",
        "draws_per_degraded_condition": 10,
        "seed_records": seed_records,
        "aggregate": aggregate,
        "runtime": runtime,
        "runtime_scope": (
            "wall-clock time from the first training epoch through validation, checkpoint loading, "
            "51-condition/draw final test passes, and prediction-file writing; excludes initial "
            "data loading/model construction; legacy raw key 'training_seconds' is not pure training time. "
            "selected_core_total includes both candidate stages regardless of selected stage."
        ),
        "comparison_policy": (
            "PIDS-CCF selected core versus clean CCF is primary. The scene router is "
            "a prespecified secondary extension and cannot replace the core merely "
            "because routing happens to improve final test results."
        ),
        "comparisons": comparisons,
    }
    (output_dir / "final_aggregate.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )

    with (output_dir / "final_long.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["dataset", "method", "condition", "metric", "mean", "std", "ci95_low", "ci95_high"]
        )
        for dataset in DATASETS:
            for method in METHODS:
                for condition in CONDITIONS:
                    for metric in METRICS:
                        row = aggregate[dataset][method][condition][metric]
                        writer.writerow(
                            [
                                dataset,
                                method,
                                condition,
                                metric,
                                row["mean"],
                                row["std"],
                                row["ci95_low"],
                                row["ci95_high"],
                            ]
                        )

    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Dataset & Method & Clean & L1 & L2 & L3 & L4 & L5 \\",
        r"\hline",
    ]
    method_labels = {
        "clean": "Clean CCF",
        "stage1": "PIDS-CCF (stage 1)",
        "selected_core": "PIDS-CCF (selected)",
        "routed": "PIDS-CCF-R",
    }
    for dataset in DATASETS:
        for method in METHODS:
            cells = [
                format_percent(aggregate[dataset][method][condition]["oa"])
                for condition in CONDITIONS
            ]
            lines.append(
                " & ".join([dataset, method_labels[method], *cells]) + r" \\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    (output_dir / "table_oa_by_severity.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(output_dir / "final_aggregate.json")


if __name__ == "__main__":
    main()
