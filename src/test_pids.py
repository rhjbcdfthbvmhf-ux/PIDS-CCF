#!/usr/bin/env python3
"""Determinism and severity sanity checks for the degradation simulator."""

from __future__ import annotations

import json

import torch

from pids import PIDSParameters, apply_pids, sample_parameters


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bands = 204
    wavelength_axis = torch.linspace(0, 1, bands, device=device)
    spectrum = (
        0.15
        + 0.65 * torch.exp(-((wavelength_axis - 0.32) ** 2) / 0.006)
        + 0.45 * torch.exp(-((wavelength_axis - 0.72) ** 2) / 0.012)
    ).clamp(0, 1)
    clean = spectrum[None, :, None, None].expand(16, -1, 7, 7).contiguous()

    identity = PIDSParameters(0, 1, 0, 1000, 0, 0, 0, 0, 0)
    assert torch.equal(clean, apply_pids(clean, identity)), "inactive PIDS is not identity"

    # Replicate boundary extension plus normalized interpolation/convolution
    # must preserve a constant radiance field exactly up to floating precision.
    constant = torch.full((4, bands, 5, 5), 0.37, device=device)
    response_only = PIDSParameters(4.3, 1.4, 0, 1000, 0, 1, 0, 0, 0)
    response_output = apply_pids(constant, response_only)
    if not torch.allclose(response_output, constant, atol=2e-6, rtol=0):
        raise AssertionError("normalized response operator does not preserve constants")

    gain_only = PIDSParameters(0, 1, 0, 1000, 0.04, 0, 0, 0, 1)
    fixed_gain = torch.randn(
        (1, bands, 1, 1),
        generator=torch.Generator(device=device).manual_seed(8080),
        device=device,
    )
    full_gain = apply_pids(clean, gain_only, response_gain=fixed_gain)
    split_gain = torch.cat(
        [
            apply_pids(chunk, gain_only, response_gain=fixed_gain)
            for chunk in clean.chunk(4)
        ],
        dim=0,
    )
    if not torch.equal(full_gain, split_gain):
        raise AssertionError("fixed scene response depends on DataLoader batch partition")

    gaussian_only = PIDSParameters(0, 1, 0.02, 1000, 0, 0, 1, 0, 0)
    gaussian_output = apply_pids(
        torch.full((32, bands, 7, 7), 0.5, device=device),
        gaussian_only,
        generator=torch.Generator(device=device).manual_seed(9090),
    )
    empirical_sigma = float((gaussian_output - 0.5).std())
    if not 0.019 <= empirical_sigma <= 0.021:
        raise AssertionError(f"Gaussian sigma mismatch: {empirical_sigma}")

    rows = []
    for level in range(1, 6):
        parameter_seed = 1000 + level
        noise_seed = 2000 + level
        params_a = sample_parameters(
            level,
            training=False,
            generator=torch.Generator().manual_seed(parameter_seed),
            compound=True,
        )
        params_b = sample_parameters(
            level,
            training=False,
            generator=torch.Generator().manual_seed(parameter_seed),
            compound=True,
        )
        out_a = apply_pids(
            clean,
            params_a,
            generator=torch.Generator(device=device).manual_seed(noise_seed),
        )
        out_b = apply_pids(
            clean,
            params_b,
            generator=torch.Generator(device=device).manual_seed(noise_seed),
        )
        assert torch.equal(out_a, out_b), f"L{level} is not seed-reproducible"
        rows.append(
            {
                "level": level,
                "mean_absolute_change": float((out_a - clean).abs().mean()),
                "root_mean_square_change": float(((out_a - clean) ** 2).mean().sqrt()),
                "minimum": float(out_a.min()),
                "maximum": float(out_a.max()),
                "parameters": params_a.as_dict(),
            }
        )
    if not all(0.0 <= row["minimum"] <= row["maximum"] <= 1.0 for row in rows):
        raise AssertionError("degraded values escaped [0,1]")

    severity_mean = []
    for level in range(1, 6):
        changes = []
        for draw in range(32):
            params = sample_parameters(
                level,
                training=False,
                generator=torch.Generator().manual_seed(10000 + 100 * level + draw),
                compound=True,
            )
            output = apply_pids(
                clean,
                params,
                generator=torch.Generator(device=device).manual_seed(
                    20000 + 100 * level + draw
                ),
            )
            changes.append(float(((output - clean) ** 2).mean().sqrt()))
        severity_mean.append(float(sum(changes) / len(changes)))
    if any(b <= a for a, b in zip(severity_mean, severity_mean[1:])):
        raise AssertionError(f"mean severity is not ordered: {severity_mean}")

    print(
        json.dumps(
            {
                "status": "ok",
                "identity_exact": True,
                "constant_response_max_abs_error": float(
                    (response_output - constant).abs().max()
                ),
                "fixed_gain_batch_partition_invariant": True,
                "gaussian_empirical_sigma": empirical_sigma,
                "mean_rms_change_by_level": severity_mean,
                "conditions": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
