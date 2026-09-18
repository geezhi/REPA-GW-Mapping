"""Parameter-free adapter from diffusion-transformer tokens to the teacher's token grid.

The student (CogVideoX) and the teacher (VideoMAEv2 / VJEPA / ...) produce tokens on
different spatio-temporal grids::

    student: 13 x 30 x 45   (latent frames x latent H/2 x latent W/2, at 49x480x720)
    teacher: 24 x 10 x 15   (48 frames / tubelet 2, 160/16, 240/16)

A learnable projector or stride-3 convolution would absorb part of the alignment
gradient, so we deliberately use **only non-parametric operations**: trilinear
temporal upsampling + average pooling. This keeps the alignment zero-parameter and
sends 100% of the gradient into the diffusion backbone.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

__all__ = ["ZeroParamFeatureAdapter"]


class ZeroParamFeatureAdapter(nn.Module):
    """Resample student hidden states onto the teacher's token grid.

    Args:
        grid: ``(F, H, W)`` token grid of the student *before* adaptation, i.e. the
            latent grid after patch embedding (e.g. ``(13, 30, 45)``).
        temporal_scale: Temporal upsampling factor (student latent frames are 4x
            compressed vs. pixels while the teacher tubelet is 2x, hence ``2.0``).
        spatial_downsample: Average-pooling factor matching the teacher patch size
            ratio (``3`` maps ``30 x 45`` to ``10 x 15``).
        drop_first_frame: Drop the first latent frame so that the student covers the
            same temporal window as the teacher (which encodes frames ``1..T``).

    Shapes:
        Input ``(B, F*H*W, D)`` -> output ``(B, F'*H'*W', D)``.
    """

    def __init__(
        self,
        grid: Tuple[int, int, int],
        temporal_scale: float = 2.0,
        spatial_downsample: int = 3,
        drop_first_frame: bool = True,
    ) -> None:
        super().__init__()
        if len(grid) != 3:
            raise ValueError(f"grid must be (F, H, W), got {grid}")
        if any(int(v) <= 0 for v in grid):
            raise ValueError(f"grid entries must be positive, got {grid}")
        if spatial_downsample < 1:
            raise ValueError("spatial_downsample must be >= 1")

        self.grid = tuple(int(v) for v in grid)
        self.temporal_scale = float(temporal_scale)
        self.spatial_downsample = int(spatial_downsample)
        self.drop_first_frame = bool(drop_first_frame)

    # -- shape helpers -----------------------------------------------------
    @property
    def output_grid(self) -> Tuple[int, int, int]:
        """``(F', H', W')`` grid produced by :meth:`forward`."""
        frames, height, width = self.grid
        if self.drop_first_frame:
            frames -= 1
        frames = int(round(frames * self.temporal_scale))
        return frames, height // self.spatial_downsample, width // self.spatial_downsample

    @property
    def num_output_tokens(self) -> int:
        frames, height, width = self.output_grid
        return frames * height * width

    @classmethod
    def for_cogvideox(
        cls,
        latent_shape: Tuple[int, int, int, int, int],
        patch_size: int,
        temporal_scale: float = 2.0,
        spatial_downsample: int = 3,
        drop_first_frame: bool = True,
    ) -> "ZeroParamFeatureAdapter":
        """Convenience constructor from a VAE latent ``(B, C, F, H, W)`` shape.

        Args:
            latent_shape: Shape of the *encoded* video latent.
            patch_size: Patch size of the transformer's patch embedding (2 for CogVideoX).
        """
        _, _, frames, height, width = latent_shape
        grid = (frames, height // patch_size, width // patch_size)
        return cls(
            grid=grid,
            temporal_scale=temporal_scale,
            spatial_downsample=spatial_downsample,
            drop_first_frame=drop_first_frame,
        )

    # -- forward -----------------------------------------------------------
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Resample flattened transformer hidden states onto the teacher grid.

        Args:
            hidden: ``(B, F*H*W, D)`` hidden states at the alignment layer.

        Returns:
            ``(B, F'*H'*W', D)`` resampled features.
        """
        if hidden.ndim != 3:
            raise ValueError(f"expected (B, N, D), got shape {tuple(hidden.shape)}")

        batch, num_tokens, dim = hidden.shape
        frames, height, width = self.grid
        if num_tokens != frames * height * width:
            raise ValueError(
                f"hidden has {num_tokens} tokens but grid {self.grid} implies "
                f"{frames * height * width}"
            )

        # (B, F, H, W, D) -> (B, D, F, H, W)
        x = hidden.view(batch, frames, height, width, dim).permute(0, 4, 1, 2, 3)

        if self.drop_first_frame:
            x = x[:, :, 1:]

        if self.temporal_scale != 1.0:
            x = nn.functional.interpolate(
                x, scale_factor=(self.temporal_scale, 1.0, 1.0), mode="trilinear"
            )

        # Average-pool space per frame: (B, D, F', H, W) -> (B*F', D, H, W) -> ...
        out_frames, out_h, out_w = x.shape[2], x.shape[3], x.shape[4]
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, dim, out_h, out_w)
        if self.spatial_downsample > 1:
            x = nn.functional.avg_pool2d(
                x, kernel_size=self.spatial_downsample, stride=self.spatial_downsample
            )
        out_h, out_w = x.shape[-2], x.shape[-1]
        x = x.reshape(batch, out_frames, dim, out_h, out_w)

        # Back to token-major: (B, F', H', W', D) -> (B, F'*H'*W', D)
        return x.permute(0, 1, 3, 4, 2).reshape(batch, -1, dim)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"grid={self.grid}, temporal_scale={self.temporal_scale}, "
            f"spatial_downsample={self.spatial_downsample}, "
            f"drop_first_frame={self.drop_first_frame}, out_grid={self.output_grid}"
        )
