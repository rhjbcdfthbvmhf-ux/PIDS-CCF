"""Compound sensor-inspired spectral degradation with observable parameters."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SeverityRange:
    shift: tuple[float, float]
    width: tuple[float, float]
    gaussian: tuple[float, float]
    poisson_gain: float
    response_sigma: float


TRAIN_RANGES = {
    1: SeverityRange((-1, 1), (0.9, 1.1), (0, 0.01), 1000, 0.02),
    2: SeverityRange((-3, 3), (0.85, 1.3), (0, 0.03), 500, 0.04),
    3: SeverityRange((-5, 5), (0.8, 1.5), (0, 0.05), 200, 0.06),
}

EVAL_RANGES = {
    1: SeverityRange((-1, 1), (0.95, 1.05), (0, 0.005), 1000, 0.02),
    2: SeverityRange((-2, 2), (0.90, 1.10), (0, 0.010), 800, 0.03),
    3: SeverityRange((-3, 3), (0.85, 1.15), (0, 0.020), 500, 0.04),
    4: SeverityRange((-4, 4), (0.80, 1.20), (0, 0.030), 350, 0.05),
    5: SeverityRange((-5, 5), (0.80, 1.50), (0, 0.050), 200, 0.06),
}


@dataclass
class PIDSParameters:
    shift: float
    width: float
    gaussian_sigma: float
    poisson_gain: float
    response_sigma: float
    response_active: float
    gaussian_active: float
    poisson_active: float
    gain_active: float

    def normalized(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        inverse_gain = 0.0
        if self.poisson_active:
            inverse_gain = ((1000.0 / self.poisson_gain) - 1.0) / 4.0
        values = torch.tensor(
            [
                self.shift / 5.0,
                (self.width - 1.0) / 0.5,
                self.gaussian_sigma / 0.05,
                inverse_gain,
                self.response_sigma / 0.06,
                self.response_active,
                self.gaussian_active,
                self.poisson_active,
                self.gain_active,
            ],
            device=device,
            dtype=dtype,
        )
        return values.unsqueeze(0).expand(batch, -1)

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in self.__dict__.items()}


def _uniform(bounds: tuple[float, float], generator: torch.Generator | None) -> float:
    value = torch.rand((), generator=generator).item()
    return bounds[0] + value * (bounds[1] - bounds[0])


def sample_parameters(
    level: int,
    training: bool,
    generator: torch.Generator | None = None,
    compound: bool = True,
) -> PIDSParameters:
    ranges = TRAIN_RANGES if training else EVAL_RANGES
    spec = ranges[level]
    if compound:
        active = [1.0, 1.0, 1.0, 1.0]
    else:
        active_tensor = torch.rand(4, generator=generator) < 0.65
        if not bool(active_tensor.any()):
            active_tensor[int(torch.randint(0, 4, (), generator=generator))] = True
        active = [float(x) for x in active_tensor.tolist()]
    shift = _uniform(spec.shift, generator) if active[0] else 0.0
    width = _uniform(spec.width, generator) if active[0] else 1.0
    gaussian = _uniform(spec.gaussian, generator) if active[1] else 0.0
    return PIDSParameters(
        shift=shift,
        width=width,
        gaussian_sigma=gaussian,
        poisson_gain=spec.poisson_gain,
        response_sigma=spec.response_sigma if active[3] else 0.0,
        response_active=active[0],
        gaussian_active=active[1],
        poisson_active=active[2],
        gain_active=active[3],
    )


def _spectral_shift(x: torch.Tensor, shift: float) -> torch.Tensor:
    if abs(shift) < 1e-8:
        return x
    batch, bands, height, width = x.shape
    signal = x.permute(0, 2, 3, 1).reshape(batch * height * width, 1, 1, bands)
    gx = torch.linspace(-1.0, 1.0, bands, device=x.device, dtype=x.dtype)
    gx = gx - (2.0 * shift / max(bands - 1, 1))
    gy = torch.zeros_like(gx)
    grid = torch.stack((gx, gy), dim=-1).reshape(1, 1, bands, 2)
    grid = grid.expand(signal.shape[0], -1, -1, -1)
    shifted = F.grid_sample(
        signal,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return shifted.reshape(batch, height, width, bands).permute(0, 3, 1, 2)


def _spectral_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 1e-6:
        return x
    batch, bands, height, width = x.shape
    radius = max(2, int(round(3.0 * sigma)))
    coordinate = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(coordinate**2) / (2.0 * sigma**2))
    kernel = (kernel / kernel.sum()).reshape(1, 1, -1)
    signal = x.permute(0, 2, 3, 1).reshape(batch * height * width, 1, bands)
    signal = F.pad(signal, (radius, radius), mode="replicate")
    blurred = F.conv1d(signal, kernel)
    return blurred.reshape(batch, height, width, bands).permute(0, 3, 1, 2)


@torch.no_grad()
def apply_pids(
    x: torch.Tensor,
    params: PIDSParameters,
    generator: torch.Generator | None = None,
    response_gain: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one sensor condition consistently over each patch in the batch."""
    degraded = x.float()
    if params.response_active:
        degraded = _spectral_shift(degraded, params.shift)
        degraded = _spectral_blur(degraded, params.width)
    if params.gaussian_active and params.gaussian_sigma > 0:
        noise = torch.randn(
            degraded.shape,
            generator=generator,
            device=degraded.device,
            dtype=degraded.dtype,
        )
        degraded = degraded + noise * params.gaussian_sigma
    if params.poisson_active:
        lam = degraded.clamp(0.0, 1.0) * params.poisson_gain
        degraded = torch.poisson(lam, generator=generator) / params.poisson_gain
    if params.gain_active and params.response_sigma > 0:
        if response_gain is None:
            gain = torch.randn(
                (1, degraded.shape[1], 1, 1),
                generator=generator,
                device=degraded.device,
                dtype=degraded.dtype,
            )
        else:
            gain = response_gain.to(device=degraded.device, dtype=degraded.dtype)
            if gain.shape != (1, degraded.shape[1], 1, 1):
                raise ValueError(
                    f"response_gain shape {tuple(gain.shape)} does not match "
                    f"(1, {degraded.shape[1]}, 1, 1)"
                )
        degraded = degraded * (1.0 + params.response_sigma * gain)
    return degraded.clamp_(0.0, 1.0).to(dtype=x.dtype)
