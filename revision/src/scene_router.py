#!/usr/bin/env python3
"""Scene-level clean/degraded expert router trained only on synthetic condition labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import HSIPatchDataset, load_cube, make_loader, train_only_scale
from model import PIDSOPERA
from pids import apply_pids, sample_parameters
from reevaluate import load_coords, summarize
from train import classification_metrics, expected_calibration_error


def load_model(run_dir: Path, bands: int, classes: int, device: torch.device):
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model = PIDSOPERA(
        bands=bands,
        classes=classes,
        spatial=not bool(config["spectral_only"]),
        auxiliary=bool(config.get("auxiliary_heads", False)),
    ).to(device)
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def soft_statistics(probability: torch.Tensor) -> torch.Tensor:
    eps = 1e-12
    sorted_probability = probability.sort(dim=1).values
    confidence = sorted_probability[:, -1]
    margin = sorted_probability[:, -1] - sorted_probability[:, -2]
    entropy = -(probability * (probability + eps).log()).sum(dim=1)
    return torch.stack((confidence, margin, entropy), dim=1)


@torch.no_grad()
def condition_arrays(
    clean_model: PIDSOPERA,
    robust_model: PIDSOPERA,
    loader,
    device: torch.device,
    level: int,
    draw: int,
    base_seed: int,
    training_ranges: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    params = None
    if level > 0:
        params = sample_parameters(
            level,
            training=training_ranges,
            generator=torch.Generator().manual_seed(base_seed + 1000 * level + draw),
            compound=True,
        )
    noise_seed = base_seed + 100000 * level + draw
    noise_generator = torch.Generator(device=device).manual_seed(noise_seed)
    response_gain = None
    raw_rows, clean_probability, robust_probability, truth = [], [], [], []
    for patch, label, _ in loader:
        patch = patch.to(device, non_blocking=True)
        if params is not None:
            if response_gain is None and params.gain_active and params.response_sigma > 0:
                response_gain = torch.randn(
                    (1, patch.shape[1], 1, 1),
                    generator=torch.Generator(device=device).manual_seed(
                        noise_seed + 17_000_003
                    ),
                    device=device,
                )
            patch = apply_pids(
                patch,
                params,
                generator=noise_generator,
                response_gain=response_gain,
            )
        center = patch.shape[-1] // 2
        spectrum = patch[:, :, center, center]
        first = spectrum[:, 1:] - spectrum[:, :-1]
        second = first[:, 1:] - first[:, :-1]
        raw_rows.append(
            torch.stack(
                (
                    spectrum.mean(dim=1),
                    spectrum.std(dim=1),
                    first.abs().mean(dim=1),
                    first.std(dim=1),
                    second.abs().mean(dim=1),
                    (spectrum <= 1e-5).float().mean(dim=1),
                    (spectrum >= 1 - 1e-5).float().mean(dim=1),
                ),
                dim=1,
            ).cpu().numpy()
        )
        clean_probability.append(clean_model(patch)["logits"].softmax(dim=1))
        robust_probability.append(robust_model(patch)["logits"].softmax(dim=1))
        truth.append(label.numpy())
    clean_probability_tensor = torch.cat(clean_probability)
    robust_probability_tensor = torch.cat(robust_probability)
    midpoint = 0.5 * (clean_probability_tensor + robust_probability_tensor)
    eps = 1e-12
    js = 0.5 * (
        clean_probability_tensor
        * ((clean_probability_tensor + eps) / (midpoint + eps)).log()
    ).sum(dim=1) + 0.5 * (
        robust_probability_tensor
        * ((robust_probability_tensor + eps) / (midpoint + eps)).log()
    ).sum(dim=1)
    output_features = torch.cat(
        (
            soft_statistics(clean_probability_tensor),
            soft_statistics(robust_probability_tensor),
            js[:, None],
            (
                clean_probability_tensor.argmax(dim=1)
                != robust_probability_tensor.argmax(dim=1)
            ).float()[:, None],
        ),
        dim=1,
    ).cpu().numpy()
    patch_features = np.concatenate((np.concatenate(raw_rows), output_features), axis=1)
    return (
        patch_features,
        clean_probability_tensor.cpu().numpy(),
        robust_probability_tensor.cpu().numpy(),
        np.concatenate(truth),
    )


def scene_vector(patch_features: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    values = patch_features if indices is None else patch_features[indices]
    summaries = (
        values.mean(axis=0),
        values.std(axis=0),
        np.quantile(values, 0.10, axis=0),
        np.quantile(values, 0.50, axis=0),
        np.quantile(values, 0.90, axis=0),
    )
    return np.concatenate(summaries)


def probability_metrics(labels: np.ndarray, probability: np.ndarray, classes: int) -> dict:
    prediction = probability.argmax(axis=1)
    result = classification_metrics(labels, prediction, classes)
    result["ece"] = expected_calibration_error(
        labels, prediction, probability.max(axis=1)
    )
    result["nll"] = float(
        -np.log(np.maximum(probability[np.arange(len(labels)), labels], 1e-12)).mean()
    )
    return result


def balanced_metric_score(metrics: dict) -> float:
    values = np.maximum(
        np.asarray([metrics["oa"], metrics["aa"], metrics["macro_f1"]]), 1e-6
    )
    return float(3.0 / np.sum(1.0 / values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-run", type=Path, required=True)
    parser.add_argument("--robust-run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-draws", type=int, default=10)
    parser.add_argument("--validation-draws", type=int, default=3)
    parser.add_argument("--test-draws", type=int, default=10)
    parser.add_argument("--scene-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clean_dir = args.clean_run.resolve()
    robust_dir = args.robust_run.resolve()
    clean_config = json.loads((clean_dir / "config.json").read_text(encoding="utf-8"))
    robust_config = json.loads((robust_dir / "config.json").read_text(encoding="utf-8"))
    for key in ("dataset", "seed", "patch_size", "spectral_only"):
        if clean_config[key] != robust_config[key]:
            raise ValueError(f"expert mismatch for {key}")
    for split_name in ("train", "val", "test"):
        if not np.array_equal(
            load_coords(clean_dir / "split.json", split_name),
            load_coords(robust_dir / "split.json", split_name),
        ):
            raise ValueError(f"expert split mismatch for {split_name}")

    cube, gt = load_cube(args.data_root.resolve(), clean_config["dataset"])
    train_coords = load_coords(clean_dir / "split.json", "train")
    val_coords = load_coords(clean_dir / "split.json", "val")
    test_coords = load_coords(clean_dir / "split.json", "test")
    cube, _ = train_only_scale(cube, train_coords)
    patch_size = int(clean_config["patch_size"])
    seed = int(clean_config["seed"])
    loaders = {}
    for name, coords in (("train", train_coords), ("val", val_coords), ("test", test_coords)):
        dataset = HSIPatchDataset(cube, gt, coords, patch_size)
        loaders[name] = make_loader(
            dataset,
            args.batch_size,
            args.workers,
            seed,
            balanced=False,
            shuffle=False,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = int(gt.max())
    clean_model, _ = load_model(clean_dir, cube.shape[-1], classes, device)
    robust_model, _ = load_model(robust_dir, cube.shape[-1], classes, device)

    rng = np.random.default_rng(seed + 30303)
    train_vectors, train_targets = [], []
    clean_arrays = condition_arrays(
        clean_model,
        robust_model,
        loaders["train"],
        device,
        0,
        0,
        base_seed=seed + 500000,
        training_ranges=True,
    )
    bootstrap_size = min(args.scene_size, len(clean_arrays[0]))
    for _ in range(args.train_draws * 3):
        indices = rng.choice(len(clean_arrays[0]), bootstrap_size, replace=False)
        train_vectors.append(scene_vector(clean_arrays[0], indices))
        train_targets.append(0)
    for level in range(1, 4):
        for draw in range(args.train_draws):
            arrays = condition_arrays(
                clean_model,
                robust_model,
                loaders["train"],
                device,
                level,
                draw,
                base_seed=seed + 500000,
                training_ranges=True,
            )
            for _ in range(3):
                indices = rng.choice(len(arrays[0]), bootstrap_size, replace=False)
                train_vectors.append(scene_vector(arrays[0], indices))
                train_targets.append(1)
    router = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0),
    )
    router.fit(np.asarray(train_vectors), np.asarray(train_targets))

    validation_rows = []
    for level in range(6):
        for draw in range(1 if level == 0 else args.validation_draws):
            arrays = condition_arrays(
                clean_model,
                robust_model,
                loaders["val"],
                device,
                level,
                draw,
                base_seed=seed + 700000,
                training_ranges=False,
            )
            probability = float(router.predict_proba(scene_vector(arrays[0])[None, :])[0, 1])
            clean_metrics = probability_metrics(arrays[3], arrays[1], classes)
            robust_metrics = probability_metrics(arrays[3], arrays[2], classes)
            validation_rows.append(
                {
                    "level": level,
                    "draw": draw,
                    "degraded_probability": probability,
                    "clean_expert": clean_metrics,
                    "robust_expert": robust_metrics,
                }
            )
    thresholds = np.linspace(0.05, 0.95, 37)
    threshold_scores = []
    for threshold in thresholds:
        clean_correct = np.mean(
            [row["degraded_probability"] < threshold for row in validation_rows if row["level"] == 0]
        )
        degraded_correct = np.mean(
            [row["degraded_probability"] >= threshold for row in validation_rows if row["level"] > 0]
        )
        threshold_scores.append(
            {
                "threshold": float(threshold),
                "balanced_accuracy": float(0.5 * (clean_correct + degraded_correct)),
                "clean_correct": float(clean_correct),
                "degraded_correct": float(degraded_correct),
            }
        )
    selected = max(
        threshold_scores,
        key=lambda row: (row["balanced_accuracy"], row["clean_correct"], -abs(row["threshold"] - 0.5)),
    )
    clean_condition = [row for row in validation_rows if row["level"] == 0][0]
    clean_prefers_clean = balanced_metric_score(
        clean_condition["clean_expert"]
    ) > balanced_metric_score(clean_condition["robust_expert"])
    degraded_rows = [row for row in validation_rows if row["level"] > 0]
    degraded_clean_score = float(
        np.mean([balanced_metric_score(row["clean_expert"]) for row in degraded_rows])
    )
    degraded_robust_score = float(
        np.mean([balanced_metric_score(row["robust_expert"]) for row in degraded_rows])
    )
    degraded_prefers_robust = degraded_robust_score > degraded_clean_score
    router_enabled = bool(clean_prefers_clean and degraded_prefers_robust)

    results = {}
    for level in range(6):
        condition = "Clean" if level == 0 else f"L{level}"
        clean_rows, robust_rows, routed_rows, detector_rows = [], [], [], []
        for draw in range(1 if level == 0 else args.test_draws):
            features, clean_probability, robust_probability, labels = condition_arrays(
                clean_model,
                robust_model,
                loaders["test"],
                device,
                level,
                draw,
                base_seed=seed + 900000,
                training_ranges=False,
            )
            degraded_probability = float(
                router.predict_proba(scene_vector(features)[None, :])[0, 1]
            )
            choose_robust = (
                degraded_probability >= selected["threshold"]
                if router_enabled
                else True
            )
            routed_probability = robust_probability if choose_robust else clean_probability
            clean_rows.append(probability_metrics(labels, clean_probability, classes))
            robust_rows.append(probability_metrics(labels, robust_probability, classes))
            routed_rows.append(probability_metrics(labels, routed_probability, classes))
            detector_rows.append(
                {
                    "draw": draw,
                    "degraded_probability": degraded_probability,
                    "selected_expert": "robust" if choose_robust else "clean",
                }
            )
        results[condition] = {
            "clean_expert": summarize(clean_rows),
            "robust_expert": summarize(robust_rows),
            "routed": summarize(routed_rows),
            "detector": detector_rows,
        }

    output = {
        "clean_run": str(clean_dir),
        "robust_run": str(robust_dir),
        "feature_definition": "five summaries of seven raw spectral and eight expert-output statistics",
        "condition_label_only": True,
        "router_enabled": router_enabled,
        "expert_policy": (
            "scene-conditional clean/robust routing"
            if router_enabled
            else "robust expert for all conditions"
        ),
        "clean_prefers_clean_on_validation": clean_prefers_clean,
        "degraded_prefers_robust_on_validation": degraded_prefers_robust,
        "degraded_clean_validation_score": degraded_clean_score,
        "degraded_robust_validation_score": degraded_robust_score,
        "validation_rows": validation_rows,
        "threshold_scores": threshold_scores,
        "selected_threshold": selected,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
