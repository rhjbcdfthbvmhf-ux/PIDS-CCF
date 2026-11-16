#!/usr/bin/env python3
"""Independently recompute all reported metrics from immutable prediction NPZs.

Read-only with respect to experiment runs. No model import, CUDA access, training,
checkpoint selection, or result repair is performed by this audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.ndimage import maximum_filter


DATASETS = {
    "IndianPines": ("Indian_pines_gt.mat", "indian_pines_gt", 16),
    "PaviaU": ("PaviaU_gt.mat", "paviaU_gt", 9),
    "Salinas": ("Salinas_gt.mat", "salinas_gt", 16),
}
SEEDS = (23, 37, 51, 71, 101)
CONDITIONS = ("Clean", "L1", "L2", "L3", "L4", "L5")
METRICS = ("oa", "aa", "kappa", "macro_f1", "worst_class_f1", "ece", "nll")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def recompute(arrays: dict, classes: int) -> dict:
    truth, prediction = arrays["truth"], arrays["prediction"]
    cm = np.bincount(classes * truth + prediction, minlength=classes * classes).reshape(classes, classes)
    tp = np.diag(cm).astype(np.float64)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(classes), where=(support + predicted) > 0)
    oa = float(tp.sum() / cm.sum())
    expected = float(np.dot(support.astype(float), predicted) / float(cm.sum()) ** 2)
    confidence = arrays["confidence"]
    correct = truth == prediction
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 16)
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if np.any(mask):
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {
        "oa": oa,
        "aa": float(recall.mean()),
        "kappa": float((oa - expected) / (1.0 - expected)),
        "macro_f1": float(f1.mean()),
        "worst_class_f1": float(f1.min()),
        "ece": ece,
        "nll": float(-np.log(np.maximum(arrays["true_probability"], 1e-12)).mean()),
        "confusion_matrix": cm,
        "per_class_f1": f1,
    }


def audit_run(run: Path, gt: np.ndarray, classes: int) -> dict:
    config, metrics, split = (load(run / name) for name in ("config.json", "metrics.json", "split.json"))
    seed = int(config["seed"])
    if seed not in SEEDS or seed != int(metrics["seed"]) or seed != int(split["seed"]):
        raise AssertionError(f"unexpected or mismatched seed in {run}")
    if digest(run / "split.json") != config["split_sha256"]:
        raise AssertionError(f"split hash mismatch in {run}")
    if digest(run / "best.pt") != metrics["checkpoint_sha256"]:
        raise AssertionError(f"checkpoint hash mismatch in {run}")
    scaling = load(run / "scaling.json")
    if scaling["fit_scope"] != "training center spectra only":
        raise AssertionError(f"scaling scope mismatch in {run}")
    partitions = {name: np.asarray(split[name], dtype=np.int64) for name in ("train", "val", "test")}
    index_sets = {}
    missing_class_ids = {}
    for name, coords in partitions.items():
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) != config[f"{name}_samples"]:
            raise AssertionError(f"invalid {name} coordinates in {run}")
        if (np.any(coords < 0) or np.any(coords[:, 0] >= gt.shape[0])
                or np.any(coords[:, 1] >= gt.shape[1])):
            raise AssertionError(f"out-of-bounds {name} coordinates in {run}")
        indices = coords[:, 0] * gt.shape[1] + coords[:, 1]
        if len(np.unique(indices)) != len(indices):
            raise AssertionError(f"duplicate {name} centers in {run}")
        index_sets[name] = set(indices.tolist())
        labels = gt[coords[:, 0], coords[:, 1]]
        counts = {str(c): int(np.sum(labels == c)) for c in np.unique(labels)}
        missing_class_ids[name] = sorted(set(range(1, classes + 1)) - set(labels.tolist()))
        if counts != split["counts"][name]:
            raise AssertionError(f"class/count mismatch for {name} in {run}")
        if name in ("train", "test") and missing_class_ids[name]:
            raise AssertionError(f"missing {name} classes in {run}: {missing_class_ids[name]}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if index_sets[left] & index_sets[right]:
            raise AssertionError(f"overlapping {left}/{right} centers in {run}")
    test_coords = partitions["test"]
    spatial_check = "not claimed for random-pixel protocol"
    if config.get("split_mode") == "buffered":
        patch_size = int(config["patch_size"])
        radius = int(config["buffer_radius"])
        if radius < patch_size - 1:
            raise AssertionError(f"buffer cannot guarantee full patch non-overlap in {run}")
        development = np.concatenate([partitions["train"], partitions["val"]])
        mask = np.zeros(gt.shape, dtype=np.uint8)
        mask[development[:, 0], development[:, 1]] = 1
        excluded = maximum_filter(mask, size=2 * radius + 1, mode="constant", cval=0)
        if np.any(excluded[test_coords[:, 0], test_coords[:, 1]]):
            raise AssertionError(f"development/test contextual overlap in {run}")
        spatial_check = f"PASS: radius {radius}, patch width {patch_size}, no development/test support overlap"
    expected_truth = gt[test_coords[:, 0], test_coords[:, 1]] - 1
    wanted = {f"{condition}_draw_{draw}.npz" for condition in CONDITIONS
              for draw in range(1 if condition == "Clean" else 10)}
    observed = {path.name for path in (run / "predictions").glob("*.npz")}
    if wanted != observed:
        raise AssertionError(f"missing/extra prediction records in {run}: {wanted ^ observed}")
    maximum_error = 0.0
    for condition in CONDITIONS:
        recomputed = []
        count = 1 if condition == "Clean" else 10
        stored = metrics["results"][condition]
        if len(stored["draws"]) != count:
            raise AssertionError(f"draw count mismatch in {run}/{condition}")
        for draw in range(count):
            path = run / "predictions" / f"{condition}_draw_{draw}.npz"
            with np.load(path, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            required = {"truth", "prediction", "confidence", "true_probability", "dataset_index"}
            if set(arrays) != required or any(a.shape != (len(expected_truth),) for a in arrays.values()):
                raise AssertionError(f"array schema mismatch in {path}")
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise AssertionError(f"nonfinite prediction array in {path}")
            for name in ("truth", "prediction", "dataset_index"):
                if not np.issubdtype(arrays[name].dtype, np.integer):
                    raise AssertionError(f"noninteger {name} in {path}")
            indices = arrays["dataset_index"]
            if not np.array_equal(np.sort(indices), np.arange(len(expected_truth))):
                raise AssertionError(f"test sample permutation mismatch in {path}")
            if not np.array_equal(arrays["truth"], expected_truth[indices]):
                raise AssertionError(f"ground-truth mismatch in {path}")
            pred = arrays["prediction"]
            if pred.min() < 0 or pred.max() >= classes:
                raise AssertionError(f"out-of-range prediction in {path}")
            confidence, probability = arrays["confidence"], arrays["true_probability"]
            if np.any(confidence < 1 / classes - 2e-6) or np.any(confidence > 1 + 2e-6):
                raise AssertionError(f"invalid confidence in {path}")
            if np.any(probability < 0) or np.any(probability > confidence + 2e-6):
                raise AssertionError(f"invalid true-class probability in {path}")
            correct = arrays["truth"] == pred
            if not np.allclose(probability[correct], confidence[correct], atol=2e-6, rtol=0):
                raise AssertionError(f"correct-label probability mismatch in {path}")
            result = recompute(arrays, classes)
            record = stored["draws"][draw]
            if not np.array_equal(result["confusion_matrix"], record["confusion_matrix"]):
                raise AssertionError(f"confusion-matrix mismatch in {path}")
            if not np.allclose(result["per_class_f1"], record["per_class_f1"], atol=2e-12, rtol=0):
                raise AssertionError(f"per-class F1 mismatch in {path}")
            for metric in METRICS:
                error = abs(result[metric] - float(record[metric]))
                maximum_error = max(maximum_error, error)
                tolerance = 3e-6 if metric in ("ece", "nll") else 2e-12
                if not math.isfinite(error) or error > tolerance:
                    raise AssertionError(f"metric mismatch {metric} in {path}: {error}")
            recomputed.append(result)
        for metric in METRICS:
            values = np.asarray([result[metric] for result in recomputed])
            for key, expected in ((metric, values.mean()), (f"{metric}_std", values.std(ddof=1) if count > 1 else 0.0)):
                if abs(float(stored[key]) - float(expected)) > 3e-6:
                    raise AssertionError(f"draw aggregation mismatch {key} in {run}/{condition}")
    return {"dataset": config["dataset"], "seed": seed, "run": str(run),
            "prediction_files": len(wanted), "test_centers": len(expected_truth),
            "max_absolute_metric_error": maximum_error,
            "missing_class_ids": missing_class_ids,
            "spatial_support_check": spatial_check,
            "checkpoint_sha256": metrics["checkpoint_sha256"], "split_sha256": config["split_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=("core", "baselines", "sensitivities"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    prefixes = {"core": ("runs_final_core_clean", "runs_final_core_stage1", "runs_final_core_stage2"),
                "baselines": ("runs_final_baselines",),
                "sensitivities": ("runs_final_ablations", "runs_buffered_full_no_overlap")}[args.family]
    runs = sorted(path.parent for prefix in prefixes for path in (root / prefix).rglob("metrics.json"))
    expected_runs = {"core": 45, "baselines": 180, "sensitivities": 195}[args.family]
    if args.family == "core":
        expected_paths = {root / prefix / dataset / variant / f"seed_{seed}"
                          for prefix, variant in (("runs_final_core_clean", "clean"),
                                                  ("runs_final_core_stage1", "pids"),
                                                  ("runs_final_core_stage2", "pids"))
                          for dataset in DATASETS for seed in SEEDS}
    elif args.family == "baselines":
        expected_paths = {root / "runs_final_baselines" / backbone / stage / dataset / variant / f"seed_{seed}"
                          for backbone in ("mlp", "spectral_cnn", "hybrid3d", "spatial_resnet")
                          for stage, variant in (("clean_stage", "clean"), ("pids_stage1", "pids"), ("pids_stage2", "pids"))
                          for dataset in DATASETS for seed in SEEDS}
    else:
        expected_paths = {root / "runs_final_ablations" / tag / stage / dataset / "pids" / f"seed_{seed}"
                          for tag in ("uniform_risk", "random_subsets", "spectral_only", "patch7", "patch15")
                          for stage in ("pids_stage1", "pids_stage2") for dataset in DATASETS for seed in SEEDS}
        expected_paths |= {root / "runs_final_ablations" / "spectral_only" / "clean_stage" / dataset / "clean" / f"seed_{seed}"
                           for dataset in DATASETS for seed in SEEDS}
        expected_paths |= {root / "runs_buffered_full_no_overlap" / stage / dataset / variant / f"seed_{seed}"
                           for stage, variant in (("clean_stage", "clean"), ("pids_stage1", "pids"), ("pids_stage2", "pids"))
                           for dataset in ("PaviaU", "Salinas") for seed in SEEDS}
    if set(runs) - expected_paths:
        raise AssertionError(f"unexpected run paths: {set(runs) - expected_paths}")
    if not args.allow_partial and set(runs) != expected_paths:
        raise AssertionError(f"missing required runs: {expected_paths - set(runs)}")
    if not args.allow_partial and len(runs) != expected_runs:
        raise AssertionError(f"{args.family} incomplete: {len(runs)}/{expected_runs} runs")
    ground_truth = {name: np.asarray(loadmat(root / "data" / "raw" / f)[key], dtype=np.int64)
                    for name, (f, key, _) in DATASETS.items()}
    records = []
    for index, run in enumerate(runs, 1):
        dataset = load(run / "config.json")["dataset"]
        records.append(audit_run(run, ground_truth[dataset], DATASETS[dataset][2]))
        print(f"PASS {index}/{len(runs)} {run.relative_to(root)}", flush=True)
    payload = {"status": "PASS" if len(runs) == expected_runs else "PARTIAL_PASS_NOT_FINAL",
               "family": args.family, "audited_runs": len(runs), "expected_runs": expected_runs,
               "audited_prediction_files": sum(r["prediction_files"] for r in records),
               "scope": "all NPZ arrays; split and checkpoint hashes; exact ground-truth/center mapping; independent OA/AA/kappa/F1/ECE/NLL recomputation",
               "exclusions": "does not certify novelty, sensor realism, independent scenes, or absence of contextual overlap in random-pixel splits",
               "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
