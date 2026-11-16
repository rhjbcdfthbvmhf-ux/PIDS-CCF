#!/usr/bin/env python3
"""GPU execution of the revision experiment with CPU-matched perturbations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch.amp import GradScaler, autocast

from data import (
    HSIPatchDataset,
    load_cube,
    make_loader,
    save_split,
    seed_everything,
    spatial_cluster_split,
    stratified_split,
    train_only_scale,
)
from model import (
    PIDSOPERA,
    build_model,
    cosine_pair_loss,
    gradient_reverse,
    prototype_compactness,
    symmetric_kl,
)
from pids import PIDSParameters, apply_pids, sample_parameters


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(classes))
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    oa = float(tp.sum() / max(cm.sum(), 1))
    return {
        "oa": oa,
        "aa": float(recall.mean()),
        "kappa": float(cohen_kappa_score(y_true, y_pred, labels=np.arange(classes))),
        "macro_f1": float(f1.mean()),
        "worst_class_f1": float(f1.min()),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "confusion_matrix": cm.tolist(),
    }


def log_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


@torch.no_grad()
def predict(
    model: PIDSOPERA,
    loader,
    device: torch.device,
    params: PIDSParameters | None,
    noise_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    truth, prediction, confidence, true_probability, indices = [], [], [], [], []
    noise_generator = torch.Generator(device="cpu").manual_seed(noise_seed)
    response_gain = None
    for patch, label, sample_index in loader:
        if params is not None:
            if response_gain is None and params.gain_active and params.response_sigma > 0:
                gain_generator = torch.Generator(device="cpu").manual_seed(
                    noise_seed + 17_000_003
                )
                response_gain = torch.randn(
                    (1, patch.shape[1], 1, 1),
                    generator=gain_generator,
                    device="cpu",
                )
            patch = apply_pids(
                patch,
                params=params,
                generator=noise_generator,
                response_gain=response_gain,
            )
        patch = patch.to(device, non_blocking=True)
        content = model.content(patch)
        logits = model.classifier(content)
        probability = logits.softmax(dim=1)
        conf, pred = probability.max(dim=1)
        truth.append(label.numpy())
        prediction.append(pred.cpu().numpy())
        confidence.append(conf.cpu().numpy())
        true_probability.append(
            probability[
                torch.arange(label.shape[0], device=device),
                label.to(device, non_blocking=True),
            ]
            .cpu()
            .numpy()
        )
        indices.append(sample_index.numpy())
    return tuple(
        np.concatenate(x)
        for x in (truth, prediction, confidence, true_probability, indices)
    )


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    bins: int = 15,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y_true)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (confidence > left) & (confidence <= right)
        if not np.any(mask):
            continue
        accuracy = np.mean(y_true[mask] == y_pred[mask])
        ece += np.sum(mask) / total * abs(accuracy - np.mean(confidence[mask]))
    return float(ece)


def evaluate_condition(
    model: PIDSOPERA,
    loader,
    device: torch.device,
    classes: int,
    level: int,
    draw: int,
    base_seed: int,
) -> tuple[dict, dict[str, np.ndarray], dict | None]:
    if level == 0:
        params = None
        parameter_record = None
    else:
        parameter_generator = torch.Generator().manual_seed(base_seed + 1000 * level + draw)
        params = sample_parameters(
            level=level,
            training=False,
            generator=parameter_generator,
            compound=True,
        )
        parameter_record = params.as_dict()
        parameter_record["response_gain_seed"] = float(
            base_seed + 100000 * level + draw + 17_000_003
        )
    y_true, y_pred, confidence, true_probability, indices = predict(
        model,
        loader,
        device,
        params=params,
        noise_seed=base_seed + 100000 * level + draw,
    )
    metrics = classification_metrics(y_true, y_pred, classes)
    metrics["ece"] = expected_calibration_error(y_true, y_pred, confidence)
    metrics["nll"] = float(-np.log(np.maximum(true_probability, 1e-12)).mean())
    arrays = {
        "truth": y_true,
        "prediction": y_pred,
        "confidence": confidence,
        "true_probability": true_probability,
        "dataset_index": indices,
    }
    return metrics, arrays, parameter_record


def validation_score(results: dict[str, dict], objective: str = "robust") -> float:
    if objective == "clean":
        values = np.maximum(
            np.asarray(
                [
                    results["Clean"]["oa"],
                    results["Clean"]["aa"],
                    results["Clean"]["macro_f1"],
                ]
            ),
            1e-6,
        )
        return float(3.0 / np.sum(1.0 / values))
    if objective != "robust":
        raise ValueError(objective)
    clean = results["Clean"]["oa"]
    degraded = 0.5 * (results["L3"]["oa"] + results["L5"]["oa"])
    severe_aa = results["L5"]["aa"]
    values = np.maximum(np.asarray([clean, degraded, severe_aa]), 1e-6)
    return float(3.0 / np.sum(1.0 / values))


def run_validation(
    model: PIDSOPERA,
    loader,
    device: torch.device,
    classes: int,
    seed: int,
    objective: str = "robust",
) -> tuple[float, dict]:
    results = {}
    for level, label in ((0, "Clean"), (3, "L3"), (5, "L5")):
        metrics, _, _ = evaluate_condition(
            model, loader, device, classes, level, draw=0, base_seed=seed + 700000
        )
        results[label] = metrics
    return validation_score(results, objective=objective), results


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "This execution adapter requires CUDA"
    assert args.variant in ("clean", "pids") and args.backbone == "s2mamba"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data_root = args.data_root.resolve()
    run_dir = (args.output_root / args.dataset / args.variant / f"seed_{args.seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    cube, gt = load_cube(data_root, args.dataset)
    if args.split_mode == "random":
        split = stratified_split(gt, seed=args.seed)
    else:
        split = spatial_cluster_split(
            gt, seed=args.seed, buffer_radius=args.buffer_radius
        )
    split_path = run_dir / "split.json"
    save_split(split, gt, split_path)
    cube, scaling = train_only_scale(cube, split.train)
    (run_dir / "scaling.json").write_text(json.dumps(scaling, indent=2), encoding="utf-8")

    train_set = HSIPatchDataset(cube, gt, split.train, args.patch_size)
    val_set = HSIPatchDataset(cube, gt, split.val, args.patch_size)
    test_set = HSIPatchDataset(cube, gt, split.test, args.patch_size)
    train_loader = make_loader(
        train_set, args.batch_size, args.workers, args.seed, balanced=True, shuffle=False
    )
    val_loader = make_loader(
        val_set, args.eval_batch_size, args.workers, args.seed, balanced=False, shuffle=False
    )
    test_loader = make_loader(
        test_set, args.eval_batch_size, args.workers, args.seed, balanced=False, shuffle=False
    )

    classes = int(gt.max())
    bands = cube.shape[-1]
    spatial = not args.spectral_only
    use_auxiliary_heads = args.variant == "opera" and any(
        weight > 0
        for weight in (
            args.lambda_operator,
            args.lambda_adversary,
            args.lambda_severity,
            args.lambda_severity_adversary,
        )
    )
    model = build_model(
        backbone=args.backbone,
        bands=bands,
        classes=classes,
        spatial=spatial,
        auxiliary=use_auxiliary_heads,
    ).to(device)

    allowed_legacy_missing = {
        "severity_predictor.net.0.weight",
        "severity_predictor.net.0.bias",
        "severity_predictor.net.1.weight",
        "severity_predictor.net.1.bias",
        "severity_predictor.net.4.weight",
        "severity_predictor.net.4.bias",
        "content_severity_adversary.net.0.weight",
        "content_severity_adversary.net.0.bias",
        "content_severity_adversary.net.1.weight",
        "content_severity_adversary.net.1.bias",
        "content_severity_adversary.net.4.weight",
        "content_severity_adversary.net.4.bias",
    }

    def load_compatible(source: Path, destination: PIDSOPERA) -> dict:
        source = source.resolve()
        source_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        for key, expected in (
            ("dataset", args.dataset),
            ("seed", args.seed),
            ("patch_size", args.patch_size),
            ("spectral_only", args.spectral_only),
            ("backbone", args.backbone),
        ):
            observed = source_config.get(key, "ccf" if key == "backbone" else None)
            if observed != expected:
                raise ValueError(
                    f"checkpoint configuration mismatch for {key}: "
                    f"{observed} != {expected}"
                )
        source_split = json.loads((source / "split.json").read_text(encoding="utf-8"))
        current_split = json.loads(split_path.read_text(encoding="utf-8"))
        if source_split != current_split:
            raise ValueError("checkpoint and current data splits differ")
        checkpoint = torch.load(source / "best.pt", map_location=device, weights_only=False)
        destination_state = destination.state_dict()
        compatible_state = {
            key: value
            for key, value in checkpoint["model"].items()
            if key in destination_state and value.shape == destination_state[key].shape
        }
        incompatible = destination.load_state_dict(compatible_state, strict=False)
        unexpected_missing = set(incompatible.missing_keys) - allowed_legacy_missing
        if unexpected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"checkpoint mismatch: missing={unexpected_missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        return {
            "run": str(source),
            "checkpoint_sha256": file_digest(source / "best.pt"),
            "best_epoch": checkpoint.get("epoch"),
        }

    initialization = None
    if args.initial_run is not None:
        initialization = load_compatible(args.initial_run, model)

    teacher = None
    teacher_source = None
    if args.teacher_run is not None:
        teacher = build_model(
            backbone=args.backbone,
            bands=bands,
            classes=classes,
            spatial=spatial,
            auxiliary=use_auxiliary_heads,
        ).to(device)
        teacher_source = load_compatible(args.teacher_run, teacher)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    warmup = max(1, int(args.epochs * 0.08))

    def lr_factor(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / warmup
        progress = (epoch - warmup) / max(args.epochs - warmup - 1, 1)
        return args.minimum_lr_ratio + (1 - args.minimum_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = GradScaler("cuda", enabled=False)
    if args.degraded_only_dro:
        if not args.class_severity_dro:
            raise ValueError("--degraded-only-dro requires --class-severity-dro")
        group_count = 3 * classes
    else:
        group_count = 4 * classes if args.class_severity_dro else 4
    group_weights = torch.ones(group_count, device=device) / group_count
    group_loss_ema = torch.zeros(group_count, device=device)
    group_seen = torch.zeros(group_count, dtype=torch.bool, device=device)
    cpu_generator = torch.Generator().manual_seed(args.seed + 4242)
    noise_generator = torch.Generator(device="cpu").manual_seed(args.seed + 8484)
    best_score = -float("inf")
    best_epoch = 0
    patience_count = 0
    history_path = run_dir / "history.jsonl"
    if history_path.exists():
        history_path.unlink()

    selection_objective = args.selection_objective
    if selection_objective == "auto":
        selection_objective = "clean" if args.variant == "clean" else "robust"

    config = vars(args).copy()
    config.update(
        {
            "device": str(device),
            "bands": bands,
            "classes": classes,
            "train_samples": len(train_set),
            "val_samples": len(val_set),
            "test_samples": len(test_set),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "split_sha256": file_digest(split_path),
            "auxiliary_heads": use_auxiliary_heads,
            "initialization": initialization,
            "teacher_source": teacher_source,
            "resolved_selection_objective": selection_objective,
        }
    )
    for key, value in list(config.items()):
        if isinstance(value, Path):
            config[key] = str(value)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        sums: dict[str, float] = {}
        examples = 0
        adversarial_scale = args.adversarial_scale * (
            2.0 / (1.0 + math.exp(-10.0 * epoch / max(args.epochs - 1, 1))) - 1.0
        )
        for patch, label, _ in train_loader:
            patch = patch.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=False):
                clean_output = model(patch, adversarial_scale=adversarial_scale)
                clean_per_sample = F.cross_entropy(
                    clean_output["logits"], label, reduction="none"
                )
                clean_cls = clean_per_sample.mean()
                total = clean_cls
                components = {"clean_cls": clean_cls}

                clean_distill = torch.zeros((), device=device, dtype=clean_cls.dtype)
                if teacher is not None and args.lambda_clean_distill > 0:
                    with torch.no_grad():
                        teacher_logits = teacher(patch)["logits"]
                    temperature = args.distill_temperature
                    clean_distill = (
                        F.kl_div(
                            F.log_softmax(clean_output["logits"] / temperature, dim=1),
                            F.softmax(teacher_logits / temperature, dim=1),
                            reduction="batchmean",
                        )
                        * temperature
                        * temperature
                    )
                    components["clean_distill"] = clean_distill

                if args.variant != "clean":
                    level = int(torch.randint(1, 4, (), generator=cpu_generator).item())
                    params = sample_parameters(
                        level=level,
                        training=True,
                        generator=cpu_generator,
                        compound=args.training_compound,
                    )
                    degraded = apply_pids(patch.cpu(), params=params, generator=noise_generator).to(device)
                    degraded_output = model(degraded, adversarial_scale=adversarial_scale)
                    degraded_per_sample = F.cross_entropy(
                        degraded_output["logits"], label, reduction="none"
                    )
                    degraded_cls = degraded_per_sample.mean()

                    with torch.no_grad():
                        if args.class_severity_dro:
                            observed_values = []
                            observed_indices = []
                            for class_index in label.unique(sorted=True).tolist():
                                class_mask = label == class_index
                                if args.degraded_only_dro:
                                    observed_values.append(
                                        degraded_per_sample[class_mask].mean().detach()
                                    )
                                    observed_indices.append(
                                        (level - 1) * classes + class_index
                                    )
                                else:
                                    observed_values.extend(
                                        (
                                            clean_per_sample[class_mask].mean().detach(),
                                            degraded_per_sample[class_mask].mean().detach(),
                                        )
                                    )
                                    observed_indices.extend(
                                        (
                                            class_index,
                                            level * classes + class_index,
                                        )
                                    )
                            observed = torch.stack(observed_values)
                            indices = torch.tensor(observed_indices, device=device)
                        else:
                            observed = torch.stack((clean_cls.detach(), degraded_cls.detach()))
                            indices = torch.tensor([0, level], device=device)
                        for group_index, group_loss in zip(indices.tolist(), observed):
                            if group_seen[group_index]:
                                group_loss_ema[group_index] = (
                                    args.group_ema * group_loss_ema[group_index]
                                    + (1.0 - args.group_ema) * group_loss
                                )
                            else:
                                group_loss_ema[group_index] = group_loss
                                group_seen[group_index] = True
                        visible_losses = group_loss_ema.clone()
                        # Unseen groups remain neutral rather than receiving an
                        # artificial zero-loss advantage early in epoch one.
                        if group_seen.any():
                            visible_losses[~group_seen] = visible_losses[group_seen].mean()
                        risk_weights = torch.softmax(
                            args.group_temperature * visible_losses, dim=0
                        )
                        uniform_weights = torch.full_like(
                            risk_weights, 1.0 / risk_weights.numel()
                        )
                        group_weights = (
                            args.group_mixture * risk_weights
                            + (1.0 - args.group_mixture) * uniform_weights
                        )
                    if args.class_severity_dro:
                        active_losses = []
                        active_indices = []
                        for class_index in label.unique(sorted=True).tolist():
                            class_mask = label == class_index
                            if args.degraded_only_dro:
                                active_losses.append(
                                    degraded_per_sample[class_mask].mean()
                                )
                                active_indices.append(
                                    (level - 1) * classes + class_index
                                )
                            else:
                                active_losses.extend(
                                    (
                                        clean_per_sample[class_mask].mean(),
                                        degraded_per_sample[class_mask].mean(),
                                    )
                                )
                                active_indices.extend(
                                    (class_index, level * classes + class_index)
                                )
                        active_losses_tensor = torch.stack(active_losses)
                        active_weights = group_weights[
                            torch.tensor(active_indices, device=device)
                        ]
                        active_weights = active_weights / active_weights.sum()
                        robust_group_loss = torch.sum(
                            active_weights * active_losses_tensor
                        )
                        if args.degraded_only_dro:
                            group_cls = (
                                args.clean_anchor_weight * clean_cls
                                + (1.0 - args.clean_anchor_weight)
                                * robust_group_loss
                            )
                        else:
                            group_cls = robust_group_loss
                    else:
                        active_weights = group_weights[
                            torch.tensor([0, level], device=device)
                        ]
                        active_weights = active_weights / active_weights.sum()
                        group_cls = active_weights[0] * clean_cls + active_weights[1] * degraded_cls
                    total = group_cls
                    components.update({"degraded_cls": degraded_cls, "group_cls": group_cls})

                    if args.variant == "opera":
                        pair = cosine_pair_loss(
                            clean_output["content"], degraded_output["content"]
                        )
                        consistency = symmetric_kl(
                            clean_output["logits"], degraded_output["logits"]
                        )
                        if args.hard_prototype:
                            prototypes = F.normalize(model.classifier.weight, dim=1)
                            clean_distance = 1.0 - torch.sum(
                                clean_output["content"] * prototypes[label], dim=1
                            )
                            degraded_distance = 1.0 - torch.sum(
                                degraded_output["content"] * prototypes[label], dim=1
                            )
                            degraded_confidence = degraded_output["logits"].softmax(dim=1)[
                                torch.arange(label.shape[0], device=device), label
                            ]
                            difficulty = (1.0 - degraded_confidence).detach()
                            difficulty = difficulty / difficulty.mean().clamp_min(1e-6)
                            proto = 0.5 * (
                                clean_distance.mean()
                                + torch.mean(difficulty * degraded_distance)
                            )
                        else:
                            proto = 0.5 * (
                                prototype_compactness(
                                    clean_output["content"], label, model.classifier
                                )
                                + prototype_compactness(
                                    degraded_output["content"], label, model.classifier
                                )
                            )
                        nuisance_loss = torch.zeros_like(group_cls)
                        adversary_loss = torch.zeros_like(group_cls)
                        severity_loss = torch.zeros_like(group_cls)
                        severity_adversary_loss = torch.zeros_like(group_cls)
                        if use_auxiliary_heads:
                            # Diagnostic legacy branch. It is excluded from the
                            # locked CARD configuration because pilot probes did
                            # not identify the simulated condition reliably.
                            phi = params.normalized(
                                batch=1, device=device, dtype=patch.dtype
                            )
                            zeros = torch.zeros_like(phi)
                            nuisance_clean_prediction = model.nuisance_predictor(
                                clean_output["nuisance"].mean(dim=0, keepdim=True)
                            )
                            nuisance_degraded_prediction = model.nuisance_predictor(
                                degraded_output["nuisance"].mean(dim=0, keepdim=True)
                            )
                            content_clean_prediction = model.content_adversary(
                                gradient_reverse(
                                    clean_output["content"].mean(dim=0, keepdim=True),
                                    adversarial_scale,
                                )
                            )
                            content_degraded_prediction = model.content_adversary(
                                gradient_reverse(
                                    degraded_output["content"].mean(dim=0, keepdim=True),
                                    adversarial_scale,
                                )
                            )
                            nuisance_loss = 0.5 * (
                                F.smooth_l1_loss(nuisance_degraded_prediction, phi)
                                + F.smooth_l1_loss(nuisance_clean_prediction, zeros)
                            )
                            adversary_loss = 0.5 * (
                                F.smooth_l1_loss(content_degraded_prediction, phi)
                                + F.smooth_l1_loss(content_clean_prediction, zeros)
                            )
                            severity_target_clean = torch.zeros(
                                1, dtype=torch.long, device=device
                            )
                            severity_target_degraded = torch.full(
                                (1,), level, dtype=torch.long, device=device
                            )
                            severity_loss = 0.5 * (
                                F.cross_entropy(
                                    model.severity_predictor(
                                        clean_output["nuisance"].mean(dim=0, keepdim=True)
                                    ),
                                    severity_target_clean,
                                )
                                + F.cross_entropy(
                                    model.severity_predictor(
                                        degraded_output["nuisance"].mean(dim=0, keepdim=True)
                                    ),
                                    severity_target_degraded,
                                )
                            )
                            severity_adversary_loss = 0.5 * (
                                F.cross_entropy(
                                    model.content_severity_adversary(
                                        gradient_reverse(
                                            clean_output["content"].mean(dim=0, keepdim=True),
                                            adversarial_scale,
                                        )
                                    ),
                                    severity_target_clean,
                                )
                                + F.cross_entropy(
                                    model.content_severity_adversary(
                                        gradient_reverse(
                                            degraded_output["content"].mean(dim=0, keepdim=True),
                                            adversarial_scale,
                                        )
                                    ),
                                    severity_target_degraded,
                                )
                            )
                        total = (
                            group_cls
                            + args.lambda_pair * pair
                            + args.lambda_kl * consistency
                            + args.lambda_proto * proto
                            + args.lambda_operator * nuisance_loss
                            + args.lambda_adversary * adversary_loss
                            + args.lambda_severity * severity_loss
                            + args.lambda_severity_adversary
                            * severity_adversary_loss
                        )
                        components.update(
                            {
                                "pair": pair,
                                "kl": consistency,
                                "proto": proto,
                                "operator": nuisance_loss,
                                "adversary": adversary_loss,
                                "severity": severity_loss,
                                "severity_adversary": severity_adversary_loss,
                            }
                        )

                total = total + args.lambda_clean_distill * clean_distill

            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
            scaler.step(optimizer)
            scaler.update()

            batch_count = patch.shape[0]
            examples += batch_count
            sums["total"] = sums.get("total", 0.0) + float(total.detach()) * batch_count
            for name, value in components.items():
                sums[name] = sums.get(name, 0.0) + float(value.detach()) * batch_count

        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "group_weights": group_weights.detach().cpu().tolist(),
            "group_loss_ema": group_loss_ema.detach().cpu().tolist(),
            **{name: value / max(examples, 1) for name, value in sums.items()},
        }

        check = (epoch + 1) % args.validation_interval == 0 or epoch + 1 == args.epochs
        if check:
            score, val_results = run_validation(
                model,
                val_loader,
                device,
                classes,
                args.seed,
                objective=selection_objective,
            )
            record["validation_score"] = score
            record["validation"] = val_results
            if score > best_score + args.minimum_improvement:
                best_score = score
                best_epoch = epoch + 1
                patience_count = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": best_epoch,
                        "validation_score": best_score,
                        "config": config,
                    },
                    run_dir / "best.pt",
                )
            else:
                patience_count += args.validation_interval
        log_json(history_path, record)
        console_record = {
            key: record[key]
            for key in ("epoch", "learning_rate", "total", "clean_cls")
            if key in record
        }
        console_record["group_weight_range"] = [
            float(group_weights.min()),
            float(group_weights.max()),
        ]
        if "validation" in record:
            console_record["validation_score"] = record["validation_score"]
            console_record["validation"] = {
                condition: {
                    metric: values[metric] for metric in ("oa", "aa", "macro_f1")
                }
                for condition, values in record["validation"].items()
            }
        print(json.dumps(console_record, sort_keys=True), flush=True)
        if patience_count >= args.patience:
            break

    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    all_results = {}
    parameter_log = {}
    prediction_dir = run_dir / "predictions"
    prediction_dir.mkdir(exist_ok=True)
    for level in range(0, 6):
        label = "Clean" if level == 0 else f"L{level}"
        draw_results = []
        draws = 1 if level == 0 else args.evaluation_draws
        for draw in range(draws):
            metrics, arrays, parameters = evaluate_condition(
                model,
                test_loader,
                device,
                classes,
                level,
                draw,
                base_seed=args.seed + 900000,
            )
            draw_results.append(metrics)
            np.savez_compressed(prediction_dir / f"{label}_draw_{draw}.npz", **arrays)
            if parameters is not None:
                parameter_log[f"{label}_draw_{draw}"] = parameters
        scalar_keys = [
            "oa",
            "aa",
            "kappa",
            "macro_f1",
            "worst_class_f1",
            "ece",
            "nll",
        ]
        summary = {}
        for key in scalar_keys:
            values = np.asarray([row[key] for row in draw_results], dtype=np.float64)
            summary[key] = float(values.mean())
            summary[key + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary["draws"] = draw_results
        all_results[label] = summary

    final = {
        "dataset": args.dataset,
        "variant": args.variant,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "training_seconds": time.time() - start_time,
        "checkpoint_sha256": file_digest(run_dir / "best.pt"),
        "results": all_results,
        "evaluation_parameters": parameter_log,
    }
    (run_dir / "metrics.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "variant": args.variant,
                "seed": args.seed,
                "best_epoch": best_epoch,
                "best_validation_score": best_score,
                "training_seconds": final["training_seconds"],
                "results": {
                    condition: {
                        metric: values[metric]
                        for metric in ("oa", "aa", "kappa", "macro_f1", "worst_class_f1")
                    }
                    for condition, values in all_results.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=["IndianPines", "PaviaU", "Salinas"], required=True)
    parser.add_argument("--variant", choices=["clean", "pids", "opera"], required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--split-mode", choices=["random", "buffered"], default="random")
    parser.add_argument("--buffer-radius", type=int, default=5)
    parser.add_argument(
        "--backbone",
        choices=["ccf", "mlp", "spectral_cnn", "hybrid3d", "spatial_resnet"],
        default="ccf",
    )
    parser.add_argument("--spectral-only", action="store_true")
    parser.add_argument("--training-compound", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.03)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--lambda-pair", type=float, default=0.20)
    parser.add_argument("--lambda-kl", type=float, default=0.10)
    parser.add_argument("--lambda-proto", type=float, default=0.05)
    parser.add_argument("--lambda-operator", type=float, default=0.20)
    parser.add_argument("--lambda-adversary", type=float, default=0.05)
    parser.add_argument("--lambda-severity", type=float, default=0.0)
    parser.add_argument("--lambda-severity-adversary", type=float, default=0.0)
    parser.add_argument("--adversarial-scale", type=float, default=0.25)
    parser.add_argument("--group-ema", type=float, default=0.90)
    parser.add_argument("--group-temperature", type=float, default=0.5)
    parser.add_argument("--group-mixture", type=float, default=0.75)
    parser.add_argument("--class-severity-dro", action="store_true")
    parser.add_argument("--degraded-only-dro", action="store_true")
    parser.add_argument("--clean-anchor-weight", type=float, default=0.50)
    parser.add_argument("--hard-prototype", action="store_true")
    parser.add_argument(
        "--initial-run",
        type=Path,
        default=None,
        help="Compatible run whose best checkpoint initializes the student.",
    )
    parser.add_argument(
        "--teacher-run",
        type=Path,
        default=None,
        help="Compatible frozen teacher used to anchor clean predictions.",
    )
    parser.add_argument("--lambda-clean-distill", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--validation-interval", type=int, default=5)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--minimum-improvement", type=float, default=1e-4)
    parser.add_argument("--evaluation-draws", type=int, default=1)
    parser.add_argument(
        "--selection-objective",
        choices=["auto", "clean", "robust"],
        default="auto",
    )
    return parser


if __name__ == "__main__":
    train(build_parser().parse_args())
