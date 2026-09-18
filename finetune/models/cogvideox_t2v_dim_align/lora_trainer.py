"""
Feature-dimension level alignment trainer for CogVideoX.

Unlike token-level alignment (which aligns the relational structure between tokens),
this trainer aligns the relational structure between feature DIMENSIONS.

Key idea:
- Token-level: sim_X = X @ X.T (N×N), sim_Y = Y @ Y.T (N×N) → direct MSE
- Dimension-level: sim_X = X.T @ X (D1×D1), sim_Y = Y.T @ Y (D2×D2)
  Since D1≠D2 (1920 vs 768), we need GW to find the optimal transport plan T: (D1, D2)
  that maps dimensions from one space to the other.

This captures "which dimensions co-activate together" and aligns that structure
across the two models.
"""
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
)
from finetune.models.cogvideox_t2v_gw_align.models.cogvideox_align import CogVideoXTransformer3DModelAlign, CogVideoXPipelineAlign
from finetune.models.cogvideox_t2v_gw_align.models.ssl.VideoMAEv2 import vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch16_224, vit_giant_patch14_224
from finetune.models.cogvideox_t2v_gw_align.models.ssl.VideoMAE import vit_base_patch16_224 as VideoMAE_vit_base_patch16_224

from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model
import torch.nn as nn
from ..utils import register
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


# ============================================================================
# GW utilities for dimension-level alignment
# ============================================================================

def pairwise_cosine_distance_dims(X):
    """
    Compute pairwise cosine distance between DIMENSIONS (columns).
    X: (N, D) -> treat each column as a "sample" of N observations
    Output: (D, D) distance matrix
    """
    # Normalize each column (dimension) across the token axis
    X_col_norm = F.normalize(X, dim=0)  # (N, D), each column has unit norm
    sim = X_col_norm.T @ X_col_norm  # (D, D) - similarity between dimensions
    dist = 1.0 - sim  # cosine distance in [0, 2]
    return dist


def sinkhorn_log_dim(M, reg, a=None, b=None, num_iters=50, tol=1e-4):
    """
    Sinkhorn algorithm in log-domain for dimension-level transport.
    Solves: min <T, M> - reg * H(T)  s.t. T1=a, T^T1=b

    Args:
        M: (D1, D2) cost matrix between dimensions
        reg: entropic regularization coefficient
        a: (D1,) source marginal (uniform if None)
        b: (D2,) target marginal (uniform if None)
        num_iters: maximum number of Sinkhorn iterations
        tol: convergence tolerance

    Returns:
        T: (D1, D2) transport plan between dimensions
    """
    D1, D2 = M.shape
    device = M.device
    dtype = M.dtype

    if a is None:
        a = torch.ones(D1, device=device, dtype=dtype) / D1
    if b is None:
        b = torch.ones(D2, device=device, dtype=dtype) / D2

    log_a = torch.log(a + 1e-20)
    log_b = torch.log(b + 1e-20)
    log_K = -M / reg  # (D1, D2)

    u = torch.zeros(D1, device=device, dtype=dtype)
    v = torch.zeros(D2, device=device, dtype=dtype)

    for i in range(num_iters):
        u_prev = u.clone()
        u = log_a - torch.logsumexp(log_K + v.unsqueeze(0), dim=1)
        v = log_b - torch.logsumexp(log_K + u.unsqueeze(1), dim=0)

        if tol > 0 and i % 5 == 0:
            diff = torch.max(torch.abs(u - u_prev)).item()
            if diff < tol:
                break

    T = torch.exp(u.unsqueeze(1) + log_K + v.unsqueeze(0))
    return T


def entropic_gw_dimensions(C_X, C_Y, reg=0.1, num_outer_iters=20, num_sinkhorn_iters=50,
                           outer_tol=1e-4, sinkhorn_tol=1e-4):
    """
    Entropic Gromov-Wasserstein for dimension-level alignment.

    Finds transport plan T: (D1, D2) that aligns the relational structure
    between dimensions of two different-sized feature spaces.

    Args:
        C_X: (D1, D1) intra-distance matrix of source dimensions
        C_Y: (D2, D2) intra-distance matrix of target dimensions
        reg: entropic regularization
        num_outer_iters: max outer iterations
        num_sinkhorn_iters: max inner Sinkhorn iterations
        outer_tol: convergence tolerance for outer loop
        sinkhorn_tol: convergence tolerance for Sinkhorn

    Returns:
        T: (D1, D2) transport plan between dimensions
    """
    D1 = C_X.shape[0]
    D2 = C_Y.shape[0]
    device = C_X.device
    dtype = C_X.dtype

    # Uniform marginals over dimensions
    a = torch.ones(D1, device=device, dtype=dtype) / D1
    b = torch.ones(D2, device=device, dtype=dtype) / D2

    # Initialize T as outer product of marginals
    T = a.unsqueeze(1) * b.unsqueeze(0)  # (D1, D2)

    # Precompute squared distance matrices
    C_X_sq = C_X ** 2  # (D1, D1)
    C_Y_sq = C_Y ** 2  # (D2, D2)

    for outer_iter in range(num_outer_iters):
        # Linearized cost matrix
        term1 = (C_X_sq @ a).unsqueeze(1)  # (D1, 1)
        term2 = (C_Y_sq @ b).unsqueeze(0)  # (1, D2)
        term3 = C_X @ T @ C_Y.T  # (D1, D2)

        M_lin = term1 + term2 - 2.0 * term3  # (D1, D2)

        # Solve linear OT with Sinkhorn
        T_new = sinkhorn_log_dim(M_lin, reg=reg, a=a, b=b,
                                 num_iters=num_sinkhorn_iters, tol=sinkhorn_tol)

        # Check convergence
        if outer_tol > 0:
            T_diff = torch.norm(T_new - T, p='fro').item()
            T_norm = torch.norm(T, p='fro').item() + 1e-20
            if T_diff / T_norm < outer_tol:
                T = T_new
                break

        T = T_new

    return T


@torch.no_grad()
def compute_dim_transport_plan(X, Y, reg=0.1, num_outer_iters=20, num_sinkhorn_iters=50,
                               dim_sample_size=None, token_sample_size=None,
                               outer_tol=1e-4, sinkhorn_tol=1e-4):
    """
    Compute GW transport plan between feature dimensions.

    Args:
        X: (N, D1) source features (CogVideoX, D1=1920)
        Y: (N, D2) target features (VideoMAEv2, D2=768)
        reg: entropic regularization
        num_outer_iters: max outer GW iterations
        num_sinkhorn_iters: max inner Sinkhorn iterations
        dim_sample_size: subsample dimensions (if D1/D2 too large)
        token_sample_size: subsample tokens for computing dimension distance matrices
        outer_tol: convergence tolerance for outer loop
        sinkhorn_tol: convergence tolerance for Sinkhorn

    Returns:
        T: (D1, D2) or (dim_sample_size_X, dim_sample_size_Y) transport plan
        indices_X: sampled dimension indices for X (or None)
        indices_Y: sampled dimension indices for Y (or None)
    """
    N, D1 = X.shape
    _, D2 = Y.shape

    # Optional: subsample tokens to reduce cost of computing (D×D) distance matrices
    # Computing C_X requires X.T @ X which is O(N * D1^2), so subsampling N helps
    if token_sample_size is not None and token_sample_size > 0 and token_sample_size < N:
        token_indices = torch.randperm(N, device=X.device)[:token_sample_size]
        X_for_dist = X[token_indices]
        Y_for_dist = Y[token_indices]
    else:
        X_for_dist = X
        Y_for_dist = Y

    # Optional: subsample dimensions if D1 or D2 is too large
    indices_X = None
    indices_Y = None
    if dim_sample_size is not None and dim_sample_size > 0:
        if dim_sample_size < D1:
            indices_X = torch.randperm(D1, device=X.device)[:dim_sample_size]
            X_for_dist = X_for_dist[:, indices_X]
        if dim_sample_size < D2:
            indices_Y = torch.randperm(D2, device=Y.device)[:dim_sample_size]
            Y_for_dist = Y_for_dist[:, indices_Y]

    # Compute intra-dimension distance matrices
    # C_X[i,j] = cosine distance between dimension i and dimension j of X
    C_X = pairwise_cosine_distance_dims(X_for_dist.float())  # (D1', D1')
    C_Y = pairwise_cosine_distance_dims(Y_for_dist.float())  # (D2', D2')

    # Solve GW to find dimension correspondence
    T = entropic_gw_dimensions(
        C_X, C_Y,
        reg=reg,
        num_outer_iters=num_outer_iters,
        num_sinkhorn_iters=num_sinkhorn_iters,
        outer_tol=outer_tol,
        sinkhorn_tol=sinkhorn_tol
    )

    return T, indices_X, indices_Y


def compute_dim_relational_loss(X, Y, T_dim, indices_X=None, indices_Y=None,
                                token_sample_size=None):
    """
    Compute dimension-level relational alignment loss using the GW transport plan.

    The transport plan T_dim: (D1, D2) tells us which dimensions should correspond.
    We use it to align the dimension-level similarity structures.

    Args:
        X: (N, D1) source features (requires grad)
        Y: (N, D2) target features (no grad)
        T_dim: (D1', D2') dimension transport plan (detached)
        indices_X: sampled dimension indices for X (or None if full)
        indices_Y: sampled dimension indices for Y (or None if full)
        token_sample_size: subsample tokens for computing similarity matrices

    Returns:
        loss: scalar tensor (differentiable w.r.t. X)
    """
    N, D1 = X.shape

    # Subsample tokens if needed (to reduce memory for N×N intermediate computations)
    if token_sample_size is not None and token_sample_size > 0 and token_sample_size < N:
        token_indices = torch.randperm(N, device=X.device)[:token_sample_size]
        X = X[token_indices]
        Y = Y[token_indices]

    # Select dimensions if subsampled during T computation
    if indices_X is not None:
        X = X[:, indices_X]
    if indices_Y is not None:
        Y = Y[:, indices_Y]

    # Compute dimension-level similarity matrices
    # Normalize along token axis (dim=0): each dimension becomes a unit vector in token space
    X_norm = F.normalize(X.float(), dim=0)  # (N, D1') - each column normalized
    Y_norm = F.normalize(Y.float(), dim=0)  # (N, D2') - each column normalized

    # sim_X[i,j] = how correlated are dimension i and dimension j across all tokens
    sim_X = X_norm.T @ X_norm  # (D1', D1')
    sim_Y = Y_norm.T @ Y_norm  # (D2', D2')

    # Use T_dim to transport Y's dimension structure to X's dimension ordering
    # T has marginals: row sum = 1/D1, col sum = 1/D2
    # We need to row-normalize T so each row sums to 1 (soft assignment of each X-dim to Y-dims)
    # This ensures T @ sim_Y @ T.T has the same scale as sim_X (values in [-1, 1])
    T_float = T_dim.float()
    T_row_norm = T_float / (T_float.sum(dim=1, keepdim=True) + 1e-20)  # (D1', D2'), each row sums to 1
    T_sim_Y = T_row_norm @ sim_Y @ T_row_norm.T  # (D1', D1')

    # Loss: MSE between X's dimension structure and transported Y's dimension structure
    loss = F.mse_loss(sim_X, T_sim_Y)

    return loss


class DimGWAlignmentHelper:
    """
    Helper class for dimension-level GW alignment.
    Manages the transport plan T: (D1, D2) that maps dimensions across spaces.
    """

    def __init__(self, reg=0.1, num_outer_iters=30, num_sinkhorn_iters=100,
                 dim_sample_size=None, token_sample_size=512,
                 update_interval=50, outer_tol=1e-4, sinkhorn_tol=1e-4):
        """
        Args:
            reg: entropic regularization (larger = smoother T)
            num_outer_iters: max GW outer iterations
            num_sinkhorn_iters: max Sinkhorn inner iterations
            dim_sample_size: subsample dimensions (None = use all D1, D2)
            token_sample_size: subsample tokens for distance matrix computation
            update_interval: recompute T every N training steps
            outer_tol: convergence tolerance for outer GW loop
            sinkhorn_tol: convergence tolerance for Sinkhorn
        """
        self.reg = reg
        self.num_outer_iters = num_outer_iters
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.dim_sample_size = dim_sample_size
        self.token_sample_size = token_sample_size
        self.update_interval = update_interval
        self.outer_tol = outer_tol
        self.sinkhorn_tol = sinkhorn_tol

        self._cached_T = None
        self._cached_indices_X = None
        self._cached_indices_Y = None
        self._step_count = 0

    def should_update(self):
        """Check if transport plan should be recomputed."""
        return self._cached_T is None or (self._step_count % self.update_interval == 0)

    def update_transport_plan(self, X, Y):
        """
        Recompute the dimension transport plan (no gradients).
        X: (N, D1), Y: (N, D2)
        """
        T, indices_X, indices_Y = compute_dim_transport_plan(
            X, Y,
            reg=self.reg,
            num_outer_iters=self.num_outer_iters,
            num_sinkhorn_iters=self.num_sinkhorn_iters,
            dim_sample_size=self.dim_sample_size,
            token_sample_size=self.token_sample_size,
            outer_tol=self.outer_tol,
            sinkhorn_tol=self.sinkhorn_tol
        )
        self._cached_T = T.detach()
        self._cached_indices_X = indices_X
        self._cached_indices_Y = indices_Y

    def compute_loss(self, X, Y, update_T=True, token_sample_size=None):
        """
        Compute dimension-level relational alignment loss.

        X: (N, D1) source features (requires grad)
        Y: (N, D2) target features (no grad)

        Returns:
            loss: scalar tensor (differentiable w.r.t. X)
        """
        if update_T:
            self._step_count += 1

        # Update transport plan periodically
        if self.should_update():
            self.update_transport_plan(X.detach(), Y.detach())

        T = self._cached_T
        indices_X = self._cached_indices_X
        indices_Y = self._cached_indices_Y

        # Use provided token_sample_size or default
        ts = token_sample_size if token_sample_size is not None else self.token_sample_size

        loss = compute_dim_relational_loss(
            X, Y, T,
            indices_X=indices_X,
            indices_Y=indices_Y,
            token_sample_size=ts
        )

        return loss

    def compute_loss_batched(self, X, Y, batch_size, token_sample_size=None):
        """
        Compute dimension-level alignment loss for batched inputs.

        X: (B*N, D1) source features
        Y: (B*N, D2) target features
        batch_size: B

        Returns:
            loss: scalar tensor averaged over batch
        """
        N = X.shape[0] // batch_size
        total_loss = 0.0

        self._step_count += 1

        for b in range(batch_size):
            X_b = X[b * N: (b + 1) * N]
            Y_b = Y[b * N: (b + 1) * N]
            total_loss = total_loss + self.compute_loss(X_b, Y_b, update_T=False,
                                                        token_sample_size=token_sample_size)

        return total_loss / batch_size


# ============================================================================
# Trainer class
# ============================================================================

class CogVideoXT2VDimAlignLoraTrainer(Trainer):
    """
    Trainer using feature-dimension level GW alignment.

    Key differences from token-level alignment:
    - Aligns the relational structure between DIMENSIONS, not tokens
    - Uses GW to find correspondence T: (D1=1920, D2=768) between dimension spaces
    - Captures "which dimensions co-activate together" patterns

    Comparison:
    - Token-level (direct): sim_X = X @ X.T (N×N), sim_Y = Y @ Y.T (N×N) → MSE
    - Dimension-level (this): sim_X = X.T @ X (D1×D1), sim_Y = Y.T @ Y (D2×D2)
      → GW finds T:(D1,D2), then MSE(sim_X, T @ sim_Y @ T.T)
    """
    UNLOAD_LIST = ["text_encoder", "vae"]

    def initialize_vision_encoder(self):
        assert len(self.args.align_models) == 1, 'Currently support one alignment model'
        if self.args.align_models[0] == "VideoMAEv2":
            self.vision_encoder = vit_base_patch16_224().to(self.accelerator.device)
            self.vision_encoder.from_pretrained('/efs/zixianhuang/ckpt/VideoMAEv2/vit_b_k710_dl_from_giant.pth')
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.require_grad = False
        elif self.args.align_models[0] == "VideoMAE":
            self.vision_encoder = VideoMAE_vit_base_patch16_224().to(self.accelerator.device)
            self.vision_encoder.from_pretrained('/efs/zixianhuang/ckpt/VideoMAE/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth')
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.require_grad = False
        elif self.args.align_models[0] == 'OminiMAE':
            from finetune.models.cogvideox_t2v_gw_align.models.ssl.omini_mae import vit_base_mae_pretraining
            self.vision_encoder = vit_base_mae_pretraining().to(self.accelerator.device)
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.require_grad = False
            self.vision_encoder.tubelet_size = 2
            self.vision_encoder.patch_size = 16
            self.vision_encoder.embed_dim = 768
        elif self.args.align_models[0] == 'VJEPA':
            from finetune.models.cogvideox_t2v_gw_align.models.ssl.JEPA import load_VJEPA
            self.vision_encoder = load_VJEPA(device=self.accelerator.device, pretrained_path='/efs/zixianhuang/ckpt/vjepa_l/vitl16.pth.tar')
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.require_grad = False
        elif self.args.align_models[0] == "VJEPA2":
            self.vision_encoder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_vit_large')
            self.vision_encoder = self.vision_encoder.to(self.accelerator.device)
            self.vision_encoder.eval()
            del self.vision_encoder.norm
            self.vision_encoder.norm = torch.nn.Identity()
            for param in self.vision_encoder.parameters():
                param.require_grad = False
        else:
            raise NotImplementedError

    def initialize_dim_gw_helper(self):
        """Initialize the dimension-level GW alignment helper."""
        # Convert 0 to None (0 means use all dimensions, represented as None internally)
        dim_sample_size = getattr(self.args, 'dim_gw_dim_sample_size', 0)
        dim_sample_size = None if dim_sample_size == 0 else dim_sample_size
        self.dim_gw_helper = DimGWAlignmentHelper(
            reg=getattr(self.args, 'dim_gw_reg', 0.1),
            num_outer_iters=getattr(self.args, 'dim_gw_outer_iters', 30),
            num_sinkhorn_iters=getattr(self.args, 'dim_gw_sinkhorn_iters', 100),
            dim_sample_size=dim_sample_size,
            token_sample_size=getattr(self.args, 'dim_gw_token_sample_size', 512),
            update_interval=getattr(self.args, 'dim_gw_update_interval', 50),
            outer_tol=getattr(self.args, 'dim_gw_outer_tol', 1e-4),
            sinkhorn_tol=getattr(self.args, 'dim_gw_sinkhorn_tol', 1e-4),
        )

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipelineAlign

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        load_kwargs = dict(
            align_layer=self.args.align_layer,
            align_dims=self.args.align_dims,
            projector_dim=getattr(self.args, 'projector_dim', 2048),
            align_residual=self.args.align_residual,
            align_attn_residual=self.args.align_attn_residual,
        )
        if self.args.align_layer_secondary is not None:
            load_kwargs["align_layer_secondary"] = self.args.align_layer_secondary
        components.transformer = CogVideoXTransformer3DModelAlign.from_pretrained(model_path, subfolder="transformer", **load_kwargs)

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        return components

    @override
    def initialize_pipeline(self) -> CogVideoXPipelineAlign:
        pipe = CogVideoXPipelineAlign(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )
        return pipe

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    @override
    def encode_text(self, prompt: str) -> torch.Tensor:
        prompt_token_ids = self.components.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.state.transformer_config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        prompt_token_ids = prompt_token_ids.input_ids
        prompt_embedding = self.components.text_encoder(prompt_token_ids.to(self.accelerator.device))[0]
        return prompt_embedding

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "raw_frames": []}

        for sample in samples:
            encoded_video = sample["encoded_video"]
            prompt_embedding = sample["prompt_embedding"]
            raw_frames = sample["raw_frames"]

            ret["encoded_videos"].append(encoded_video)
            ret["prompt_embedding"].append(prompt_embedding)
            ret["raw_frames"].append(raw_frames)

        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["raw_frames"] = torch.stack(ret["raw_frames"])

        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        raw_frames = batch["raw_frames"]  # [B, C, F, H, W] value range [-1, 1]

        # Initialize dim GW helper on first call
        if not hasattr(self, 'dim_gw_helper'):
            self.initialize_dim_gw_helper()

        # Pre-process for vision encoder
        B, C, F, H, W = raw_frames.shape
        raw_frames = raw_frames.transpose(1, 2).flatten(0, 1)  # B * F, C, H, W
        if self.args.align_models[0] in ['VideoMAEv2', 'VideoMAE', 'OminiMAE', "VJEPA", "VJEPA2"]:
            raw_frames = (raw_frames + 1.0) / 2.0
            raw_frames = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(raw_frames)
        else:
            raise NotImplementedError("Dim alignment currently only supports VideoMAEv2/VideoMAE/OminiMAE/VJEPA/VJEPA2")
        raw_frames = raw_frames.reshape(B, F, C, H, W).transpose(1, 2)

        # Pre-process frames for Video Foundation Models
        assert len(self.args.align_models) == 1, "Support only align one model currently"
        repa_raw_frames = raw_frames[:, :, 1:]  # remove the first frame
        B, C, F, H, W = repa_raw_frames.shape

        repa_raw_frames = repa_raw_frames.transpose(1, 2).flatten(0, 1)
        # 480x720 -> 160x240
        repa_raw_frames = torch.nn.functional.interpolate(repa_raw_frames, (H // 3, W // 3), mode='bicubic')
        repa_raw_frames = repa_raw_frames.reshape(B, F, C, H // 3, W // 3).transpose(1, 2)  # B, C, F, H, W

        # Encode frames with frozen vision encoder
        with torch.no_grad():
            B, C, F, H, W = repa_raw_frames.shape
            align_target = self.vision_encoder(repa_raw_frames)
            align_target = align_target.transpose(1, 2).reshape(B, -1, F // self.vision_encoder.tubelet_size, H // self.vision_encoder.patch_size, W // self.vision_encoder.patch_size)

        # Flatten target: B, N_tokens, D  (N_tokens = 24*10*15 = 3600, D=768)
        align_target = align_target.flatten(2).transpose(1, 2)  # B, 3600, D

        # Prepare latent for diffusion
        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            raise NotImplementedError("This is for CogVideoX1.5 but the 1.5 is not used in VideoREPA")

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Sample random timestep
        num_train_timesteps = self.components.scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=self.accelerator.device).long()

        # Add noise
        latent = latent.permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
        noise = torch.randn_like(latent)
        latent_added_noise = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Prepare rotary embeds
        vae_scale_factor_spatial = 2 ** (len(self.components.vae.config.block_out_channels) - 1)
        transformer_config = self.state.transformer_config
        rotary_emb = (
            self.prepare_rotary_positional_embeddings(
                height=height * vae_scale_factor_spatial,
                width=width * vae_scale_factor_spatial,
                num_frames=num_frames,
                transformer_config=transformer_config,
                vae_scale_factor_spatial=vae_scale_factor_spatial,
                device=self.accelerator.device,
            )
            if transformer_config.use_rotary_positional_embeddings
            else None
        )

        # Forward pass - get predicted noise and raw alignment features
        predicted_noises, aligns = self.components.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )
        predicted_noise = predicted_noises[0]

        # Process alignment features (raw hidden_states, dim=1920 for 2B model)
        align = aligns[0]
        align = align.reshape(B, 13, 60 // 2, 90 // 2, -1)  # B, 13, 30, 45, inner_dim
        # Remove first frame to match VideoMAEv2 (which processes frames 1..48)
        align = align[:, 1:]  # B, 12, 30, 45, inner_dim

        # Spatial downsampling: 30x45 -> 10x15 (to match VideoMAEv2 spatial resolution)
        align = align.permute(0, 4, 1, 2, 3)  # B, inner_dim, 12, 30, 45
        # Temporal upsample: 12 -> 24 to match VideoMAEv2
        align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')
        B_a, C_a, F_a, H_a, W_a = align.shape
        align = align.permute(0, 2, 1, 3, 4).reshape(B_a * F_a, C_a, H_a, W_a)
        align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))  # 30x45 -> 10x15
        align = align.reshape(B_a, F_a, C_a, H_a // 3, W_a // 3).permute(0, 2, 1, 3, 4)  # B, inner_dim, 24, 10, 15

        # Flatten: B, N_tokens, inner_dim  (N_tokens = 24*10*15 = 3600, inner_dim=1920)
        align = align.flatten(2).transpose(1, 2)  # B, 3600, 1920

        # Compute dimension-level GW alignment loss
        # Flatten batch: (B*3600, 1920) and (B*3600, 768)
        align_flat = align.flatten(0, 1)  # (B*3600, 1920)
        align_target_flat = align_target.flatten(0, 1)  # (B*3600, 768)

        # Compute loss using dimension-level GW helper
        proj_loss = self.dim_gw_helper.compute_loss_batched(
            align_flat, align_target_flat, batch_size=batch_size,
            token_sample_size=getattr(self.args, 'dim_gw_token_sample_size', 512)
        )

        # Compute diffusion loss
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_added_noise, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        return [loss, proj_loss]

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipelineAlign
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        prompt, image, video = eval_data["prompt"], eval_data["image"], eval_data["video"]

        video_generate = pipe(
            num_frames=self.state.train_frames,
            height=self.state.train_height,
            width=self.state.train_width,
            prompt=prompt,
            generator=self.state.generator,
        ).frames[0]
        return [("video", video_generate)]

    def prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        transformer_config: Dict,
        vae_scale_factor_spatial: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)

        if transformer_config.patch_size_t is None:
            base_num_frames = num_frames
        else:
            base_num_frames = (num_frames + transformer_config.patch_size_t - 1) // transformer_config.patch_size_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )

        return freqs_cos, freqs_sin


register("cogvideox-t2v-dim-align", "lora", CogVideoXT2VDimAlignLoraTrainer)
