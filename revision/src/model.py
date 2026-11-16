"""Spectral-spatial backbone and optional diagnostic heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def gradient_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    return GradientReverse.apply(x, scale)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SpatialResidual(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=dilation,
                dilation=dilation,
                groups=channels,
                bias=False,
            ),
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            SqueezeExcite(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SpectralCenterEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 7, padding=3, bias=False),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv1d(64, 96, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 96),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        center = patch.shape[-1] // 2
        spectrum = patch[:, :, center, center].unsqueeze(1)
        return self.net(spectrum)


class ContentEncoder(nn.Module):
    def __init__(self, bands: int, feature_dim: int = 128, spatial: bool = True) -> None:
        super().__init__()
        self.spatial = spatial
        self.spectral = SpectralCenterEncoder(64)
        if spatial:
            self.project = nn.Sequential(
                nn.Conv2d(bands, 64, 1, bias=False),
                nn.GroupNorm(8, 64),
                nn.GELU(),
            )
            self.trunk = nn.Sequential(
                SpatialResidual(64, 1),
                SpatialResidual(64, 2),
                SpatialResidual(64, 1),
            )
            fusion_dim = 64 * 4
        else:
            fusion_dim = 64
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, feature_dim),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        spectral = self.spectral(patch)
        if not self.spatial:
            return F.normalize(self.head(spectral), dim=1)
        feature_map = self.trunk(self.project(patch))
        center = feature_map.shape[-1] // 2
        center_feature = feature_map[:, :, center, center]
        mean_feature = feature_map.mean(dim=(-2, -1))
        max_feature = feature_map.amax(dim=(-2, -1))
        fused = torch.cat((spectral, center_feature, mean_feature, max_feature), dim=1)
        return F.normalize(self.head(fused), dim=1)


class CosineClassifier(nn.Module):
    def __init__(self, feature_dim: int, classes: int, scale: float = 20.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, feature_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.log_scale = nn.Parameter(torch.tensor(math.log(scale)))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        weight = F.normalize(self.weight, dim=1)
        return self.log_scale.exp().clamp(max=100.0) * feature @ weight.t()


class ParameterHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 9) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PIDSOPERA(nn.Module):
    def __init__(
        self,
        bands: int,
        classes: int,
        feature_dim: int = 128,
        nuisance_dim: int = 64,
        spatial: bool = True,
        auxiliary: bool = True,
    ) -> None:
        super().__init__()
        self.auxiliary = auxiliary
        self.content = ContentEncoder(bands, feature_dim=feature_dim, spatial=spatial)
        self.classifier = CosineClassifier(feature_dim, classes)
        if auxiliary:
            self.nuisance = SpectralCenterEncoder(nuisance_dim)
            self.nuisance_predictor = ParameterHead(nuisance_dim)
            self.content_adversary = ParameterHead(feature_dim)
            self.severity_predictor = ParameterHead(nuisance_dim, output_dim=4)
            self.content_severity_adversary = ParameterHead(feature_dim, output_dim=4)

    def forward(
        self,
        patch: torch.Tensor,
        adversarial_scale: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        content = self.content(patch)
        output = {
            "content": content,
            "logits": self.classifier(content),
        }
        if self.auxiliary:
            nuisance = self.nuisance(patch)
            output.update(
                {
                    "nuisance": nuisance,
                    "nuisance_phi": self.nuisance_predictor(nuisance),
                    "content_phi": self.content_adversary(
                        gradient_reverse(content, adversarial_scale)
                    ),
                    "nuisance_severity": self.severity_predictor(nuisance),
                    "content_severity": self.content_severity_adversary(
                        gradient_reverse(content, adversarial_scale)
                    ),
                }
            )
        return output


class MLPContent(nn.Module):
    """Single-spectrum multilayer perceptron baseline."""

    spatial = False

    def __init__(self, bands: int, feature_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bands, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, feature_dim),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        center = patch.shape[-1] // 2
        return F.normalize(self.net(patch[:, :, center, center]), dim=1)


class SpectralCNNContent(nn.Module):
    """Single-spectrum 1D-CNN baseline."""

    spatial = False

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = SpectralCenterEncoder(feature_dim)

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encoder(patch), dim=1)


class Hybrid3DContent(nn.Module):
    """Compact HybridSN-style 3D-to-2D spectral-spatial baseline."""

    spatial = True

    def __init__(self, feature_dim: int = 128) -> None:
        super().__init__()
        self.spectral_spatial = nn.Sequential(
            nn.Conv3d(1, 8, (7, 3, 3), padding=(3, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.GELU(),
            nn.Conv3d(8, 16, (5, 3, 3), padding=(2, 1, 1), bias=False),
            nn.BatchNorm3d(16),
            nn.GELU(),
            nn.Conv3d(16, 24, (3, 3, 3), padding=1, bias=False),
            nn.BatchNorm3d(24),
            nn.GELU(),
        )
        self.spatial_head = nn.Sequential(
            nn.Conv2d(24, 64, 3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, feature_dim),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        feature = self.spectral_spatial(patch.unsqueeze(1)).mean(dim=2)
        return F.normalize(self.spatial_head(feature), dim=1)


class SpatialResNetContent(nn.Module):
    """Residual spatial baseline without the explicit center-spectrum path."""

    spatial = True

    def __init__(self, bands: int, feature_dim: int = 128) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(bands, 64, 1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            SpatialResidual(64, 1),
            SpatialResidual(64, 2),
            SpatialResidual(64, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(256, feature_dim),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        feature = self.trunk(self.project(patch))
        center = feature.shape[-1] // 2
        fused = torch.cat(
            (feature[:, :, center, center], feature.mean(dim=(-2, -1))), dim=1
        )
        return F.normalize(self.head(fused), dim=1)


class FeatureClassifier(nn.Module):
    """Common classifier interface for input-matched baseline encoders."""

    auxiliary = False

    def __init__(self, content: nn.Module, classes: int, feature_dim: int = 128) -> None:
        super().__init__()
        self.content = content
        self.classifier = CosineClassifier(feature_dim, classes)

    def forward(
        self, patch: torch.Tensor, adversarial_scale: float = 0.0
    ) -> dict[str, torch.Tensor]:
        del adversarial_scale
        content = self.content(patch)
        return {"content": content, "logits": self.classifier(content)}


def build_model(
    backbone: str,
    bands: int,
    classes: int,
    spatial: bool = True,
    auxiliary: bool = False,
) -> nn.Module:
    if backbone == "ccf":
        return PIDSOPERA(
            bands=bands,
            classes=classes,
            spatial=spatial,
            auxiliary=auxiliary,
        )
    if auxiliary:
        raise ValueError("auxiliary diagnostic heads are supported only by ccf")
    if backbone == "mlp":
        return FeatureClassifier(MLPContent(bands), classes)
    if backbone == "spectral_cnn":
        return FeatureClassifier(SpectralCNNContent(), classes)
    if backbone == "hybrid3d":
        return FeatureClassifier(Hybrid3DContent(), classes)
    if backbone == "spatial_resnet":
        return FeatureClassifier(SpatialResNetContent(bands), classes)
    raise ValueError(f"unknown backbone: {backbone}")


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    log_a = F.log_softmax(logits_a, dim=1)
    log_b = F.log_softmax(logits_b, dim=1)
    prob_a = log_a.exp()
    prob_b = log_b.exp()
    return 0.5 * (
        F.kl_div(log_a, prob_b, reduction="batchmean")
        + F.kl_div(log_b, prob_a, reduction="batchmean")
    )


def cosine_pair_loss(feature_a: torch.Tensor, feature_b: torch.Tensor) -> torch.Tensor:
    return (1.0 - (feature_a * feature_b).sum(dim=1)).mean()


def prototype_compactness(
    feature: torch.Tensor,
    labels: torch.Tensor,
    classifier: CosineClassifier,
) -> torch.Tensor:
    prototypes = F.normalize(classifier.weight, dim=1)
    return (1.0 - (feature * prototypes[labels]).sum(dim=1)).mean()
