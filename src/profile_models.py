#!/usr/bin/env python3
"""Profile parameters, multiply-accumulate counts, throughput, and CUDA memory."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import build_model


DATASETS = {
    "IndianPines": {"bands": 200, "classes": 16},
    "PaviaU": {"bands": 103, "classes": 9},
    "Salinas": {"bands": 204, "classes": 16},
}
BACKBONES = ("mlp", "spectral_cnn", "hybrid3d", "spatial_resnet", "ccf")


def macs_per_patch(model: nn.Module, sample: torch.Tensor, classes: int) -> int:
    total = 0
    handles = []

    def convolution_hook(module: nn.Module, inputs, output) -> None:
        nonlocal total
        del inputs
        batch = int(output.shape[0])
        output_elements = int(output.numel() // batch)
        kernel = int(np.prod(module.kernel_size))
        per_output = kernel * int(module.in_channels // module.groups)
        total += output_elements * per_output

    def linear_hook(module: nn.Linear, inputs, output) -> None:
        nonlocal total
        del inputs
        batch = int(output.shape[0]) if output.ndim > 1 else 1
        output_elements = int(output.numel() // batch)
        total += output_elements * int(module.in_features)

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            handles.append(module.register_forward_hook(convolution_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    with torch.no_grad():
        model(sample)
    for handle in handles:
        handle.remove()
    # CosineClassifier performs an explicit matrix product rather than nn.Linear.
    total += 128 * int(classes)
    return int(total)


def throughput_profile(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int,
    blocks: int,
    iterations: int,
) -> dict:
    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rates = []
        for _ in range(blocks):
            start = time.perf_counter()
            for _ in range(iterations):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            duration = time.perf_counter() - start
            rates.append(float(sample.shape[0] * iterations / duration))
    peak = None
    if device.type == "cuda":
        peak = float(torch.cuda.max_memory_allocated(device) / (1024**2))
    return {
        "throughput_patches_per_second_mean": float(np.mean(rates)),
        "throughput_patches_per_second_std": float(np.std(rates, ddof=1)) if len(rates) > 1 else 0.0,
        "throughput_blocks": int(blocks),
        "iterations_per_block": int(iterations),
        "batch_size": int(sample.shape[0]),
        "peak_allocated_memory_mib": peak,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--patch-size", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA profiling requested but no CUDA device is available")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for dataset, specification in DATASETS.items():
        for backbone in BACKBONES:
            torch.manual_seed(20260827)
            model = build_model(
                backbone,
                bands=specification["bands"],
                classes=specification["classes"],
                spatial=True,
                auxiliary=False,
            ).to(device)
            sample_for_macs = torch.zeros(
                1,
                specification["bands"],
                args.patch_size,
                args.patch_size,
                device=device,
            )
            macs = macs_per_patch(model, sample_for_macs, specification["classes"])
            sample = torch.randn(
                args.batch_size,
                specification["bands"],
                args.patch_size,
                args.patch_size,
                device=device,
            )
            timing = throughput_profile(
                model,
                sample,
                device,
                warmup=args.warmup,
                blocks=args.blocks,
                iterations=args.iterations,
            )
            record = {
                "dataset": dataset,
                "backbone": backbone,
                "bands": specification["bands"],
                "classes": specification["classes"],
                "patch_size": args.patch_size,
                "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                "macs_per_patch": macs,
                "gmacs_per_patch": macs / 1e9,
                **timing,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            del model, sample, sample_for_macs
            if device.type == "cuda":
                torch.cuda.empty_cache()
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    payload = {
        "device": str(device),
        "device_name": device_name,
        "precision": "float32",
        "tf32_allowed": False,
        "warmup_iterations": args.warmup,
        "environment": {"python": platform.python_version(), "pytorch": torch.__version__,
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()},
        "mac_definition": (
            "one multiplication-accumulation in Conv1d/2d/3d or Linear is counted as one MAC; "
            "the 128-by-class cosine-head matrix product is included"
        ),
        "mac_exclusions": (
            "normalization, activation, pooling, tensor concatenation, feature normalization, "
            "and PIDS data transformation are excluded"
        ),
        "throughput_scope": "model forward pass in evaluation mode, including input tensor resident on device",
        "records": records,
    }
    (args.output / "model_profile.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (args.output / "model_profile.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
