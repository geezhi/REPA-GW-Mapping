"""
Dimension-guided token alignment trainer for CogVideoX (Plan A).

Key idea:
1. Use GW to find the dimension transport plan T: (D1, D2) that maps
   dimensions from Y's space (D2=768) to X's space (D1=1920).
2. Row-normalize T to get a soft assignment matrix T_norm: (D1, D2),
   where each row sums to 1 (each X-dimension is a weighted combination of Y-dimensions).
3. Map Y into X's dimension space: Y_mapped = Y @ T_norm.T → (N, D1)
4. Compute per-token alignment loss in the shared D1-dimensional space:
   loss = mean(1 - cosine_sim(X_i, Y_mapped_i))

This combines the benefits of:
- Dimension-level GW: finds which dimensions correspond across spaces
- Token-level alignment: aligns individual token features directly

Compared to pure relational alignment (sim matrix MSE), this provides
a stronger signal because it aligns actual feature values, not just
their relational structure.
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
from finetune.paths import ckpt


# ============================================================================
# GW utilities for dimension-level transport plan computation
# ============================================================================

def pairwise_cosine_distance_dims(X):
    """
    Compute pairwise cosine distance between DIMENSIONS (columns).
    X: (N, D) -> treat each column as a "sample" of N observations
    Output: (D, D) distance matrix
    """
    X_col_norm = F.normalize(X, dim=0)  # (N, D), each column has unit norm
    sim = X_col_norm.T @ X_col_norm  # (D, D)
    dist = 1.0 - sim  # cosine distance in [0, 2]
    return dist


def sinkhorn_log_dim(M, reg, a=None, b=None, num_iters=50, tol=1e-4):
    """
    Sinkhorn algorithm in log-domain for dimension-level transport.
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
    Finds transport plan T: (D1, D2) between dimensions.
    """
    D1 = C_X.shape[0]
    D2 = C_Y.shape[0]
    device = C_X.device
    dtype = C_X.dtype

    a = torch.ones(D1, device=device, dtype=dtype) / D1
    b = torch.ones(D2, device=device, dtype=dtype) / D2

    T = a.unsqueeze(1) * b.unsqueeze(0)  # (D1, D2)

    C_X_sq = C_X ** 2
    C_Y_sq = C_Y ** 2

    for outer_iter in range(num_outer_iters):
        term1 = (C_X_sq @ a).unsqueeze(1)  # (D1, 1)
        term2 = (C_Y_sq @ b).unsqueeze(0)  # (1, D2)
        term3 = C_X @ T @ C_Y.T  # (D1, D2)

        M_lin = term1 + term2 - 2.0 * term3

        T_new = sinkhorn_log_dim(M_lin, reg=reg, a=a, b=b,
                                 num_iters=num_sinkhorn_iters, tol=sinkhorn_tol)

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
                               token_sample_size=None,
                               outer_tol=1e-4, sinkhorn_tol=1e-4):
    """
    Compute GW transport plan between feature dimensions.

    Args:
        X: (N, D1) source features (CogVideoX, D1=1920)
        Y: (N, D2) target features (VideoMAEv2, D2=768)
        reg: entropic regularization
        num_outer_iters: max outer GW iterations
        num_sinkhorn_iters: max inner Sinkhorn iterations
        token_sample_size: subsample tokens for computing dimension distance matrices
        outer_tol: convergence tolerance for outer loop
        sinkhorn_tol: convergence tolerance for Sinkhorn

    Returns:
        T: (D1, D2) transport plan between dimensions
    """
    N, D1 = X.shape
    _, D2 = Y.shape

    # Subsample tokens to reduce cost of computing (D×D) distance matrices
    if token_sample_size is not None and token_sample_size > 0 and token_sample_size < N:
        token_indices = torch.randperm(N, device=X.device)[:token_sample_size]
        X_for_dist = X[token_indices]
        Y_for_dist = Y[token_indices]
    else:
        X_for_dist = X
        Y_for_dist = Y

    # Compute intra-dimension distance matrices
    C_X = pairwise_cosine_distance_dims(X_for_dist.float())  # (D1, D1)
    C_Y = pairwise_cosine_distance_dims(Y_for_dist.float())  # (D2, D2)

    # Solve GW to find dimension correspondence
    T = entropic_gw_dimensions(
        C_X, C_Y,
        reg=reg,
        num_outer_iters=num_outer_iters,
        num_sinkhorn_iters=num_sinkhorn_iters,
        outer_tol=outer_tol,
        sinkhorn_tol=sinkhorn_tol
    )

    return T


def compute_dim_token_loss(X, Y, T_dim, token_sample_size=None, margin=0.0):
    """
    Plan A: Use dimension transport plan to map Y into X's space,
    then compute per-token cosine alignment loss with optional margin.

    Args:
        X: (N, D1) source features (requires grad, CogVideoX 1920-dim)
        Y: (N, D2) target features (no grad, VideoMAEv2 768-dim)
        T_dim: (D1, D2) dimension transport plan (detached)
        token_sample_size: subsample tokens for loss computation
        margin: margin threshold. Tokens with cosine distance < margin are
                considered "already aligned" and contribute zero loss/gradient.
                E.g. margin=0.1 means tokens with cos_sim > 0.9 are ignored.

    Returns:
        loss: scalar tensor (differentiable w.r.t. X)
    """
    N, D1 = X.shape

    # Optional token subsampling
    if token_sample_size is not None and token_sample_size > 0 and token_sample_size < N:
        token_indices = torch.randperm(N, device=X.device)[:token_sample_size]
        X = X[token_indices]
        Y = Y[token_indices]

    # Row-normalize T: each row sums to 1
    # T_row_norm[i, j] = "how much does X-dimension i correspond to Y-dimension j"
    T_float = T_dim.float()
    T_row_norm = T_float / (T_float.sum(dim=1, keepdim=True) + 1e-20)  # (D1, D2)

    # Map Y into X's dimension space using the transport plan
    # Y_mapped[n, i] = sum_j T_row_norm[i, j] * Y[n, j]
    Y_mapped = Y.float() @ T_row_norm.T  # (N, D2) @ (D2, D1) -> (N, D1)

    # Per-token cosine similarity loss
    X_norm = F.normalize(X.float(), dim=-1)  # (N, D1)
    Y_mapped_norm = F.normalize(Y_mapped, dim=-1)  # (N, D1)

    # Cosine distance per token: d_i = 1 - cos(X_i, Y_mapped_i)
    cos_sim = (X_norm * Y_mapped_norm).sum(dim=-1)  # (N,)
    cos_dist = 1.0 - cos_sim  # (N,), in [0, 2]

    # Margin: only penalize tokens with distance > margin
    # relu(d_i - margin) = 0 if already aligned (d_i < margin), else d_i - margin
    if margin > 0:
        loss = torch.nn.functional.relu(cos_dist - margin).mean()
    else:
        loss = cos_dist.mean()

    return loss


class DimTokenAlignmentHelper:
    """
    Helper class for dimension-guided token alignment (Plan A).

    Computes the dimension transport plan T: (D1, D2) per sample using GW,
    then uses T to map Y into X's dimension space for per-token cosine loss.

    Key design: T is recomputed for EVERY sample (no caching), using ALL tokens
    (no subsampling) to compute the dimension distance matrices. This ensures
    the most accurate dimension correspondence for each individual sample.
    """

    def __init__(self, reg=0.1, num_outer_iters=30, num_sinkhorn_iters=100,
                 token_sample_size_for_loss=0, outer_tol=1e-4, sinkhorn_tol=1e-4,
                 margin=0.0):
        """
        Args:
            reg: entropic regularization for GW (larger = smoother T)
            num_outer_iters: max GW outer iterations
            num_sinkhorn_iters: max Sinkhorn inner iterations
            token_sample_size_for_loss: subsample tokens for loss computation (0 = use all)
            outer_tol: convergence tolerance for outer GW loop
            sinkhorn_tol: convergence tolerance for Sinkhorn
            margin: margin for per-token cosine loss. Tokens with distance < margin
                    contribute zero loss. Default 0 = no margin.
        """
        self.reg = reg
        self.num_outer_iters = num_outer_iters
        self.num_sinkhorn_iters = num_sinkhorn_iters
        self.token_sample_size_for_loss = token_sample_size_for_loss
        self.outer_tol = outer_tol
        self.sinkhorn_tol = sinkhorn_tol
        self.margin = margin

    def compute_loss(self, X, Y):
        """
        Compute dimension-guided per-token alignment loss for a single sample.

        For each sample:
        1. Compute GW transport plan T: (D1, D2) using ALL tokens (no subsampling)
        2. Row-normalize T and map Y into X's space
        3. Compute per-token cosine loss

        X: (N, D1) source features (requires grad)
        Y: (N, D2) target features (no grad)

        Returns:
            loss: scalar tensor (differentiable w.r.t. X)
        """
        # Compute T for this sample using all tokens (token_sample_size=None)
        T = compute_dim_transport_plan(
            X.detach(), Y.detach(),
            reg=self.reg,
            num_outer_iters=self.num_outer_iters,
            num_sinkhorn_iters=self.num_sinkhorn_iters,
            token_sample_size=None,  # use ALL tokens for dimension distance
            outer_tol=self.outer_tol,
            sinkhorn_tol=self.sinkhorn_tol
        )

        token_sample_size = self.token_sample_size_for_loss if self.token_sample_size_for_loss > 0 else None
        loss = compute_dim_token_loss(X, Y, T.detach(), token_sample_size=token_sample_size, margin=self.margin)
        return loss

    def compute_loss_batched(self, X, Y, batch_size):
        """
        Compute loss for batched inputs. GW is computed per sample.

        X: (B*N, D1) source features
        Y: (B*N, D2) target features
        batch_size: B

        Returns:
            loss: scalar tensor averaged over batch
        """
        N = X.shape[0] // batch_size
        total_loss = 0.0

        for b in range(batch_size):
            X_b = X[b * N: (b + 1) * N]
            Y_b = Y[b * N: (b + 1) * N]
            total_loss = total_loss + self.compute_loss(X_b, Y_b)

        return total_loss / batch_size


# ============================================================================
# Trainer class
# ============================================================================

class CogVideoXT2VDimTokenAlignLoraTrainer(Trainer):
    """
    Trainer using dimension-guided token alignment (Plan A).

    Strategy:
    1. Compute GW transport plan T: (D1=1920, D2=768) between dimensions
    2. Row-normalize T to get soft assignment: each X-dim is a weighted combo of Y-dims
    3. Map Y into X's space: Y_mapped = Y @ T_norm.T → (N, 1920)
    4. Per-token cosine loss: mean(1 - cos(X_i, Y_mapped_i))

    Advantages:
    - Stronger signal than relational loss (aligns actual values, not just structure)
    - No learnable projector needed (T is computed from data)
    - T is recomputed periodically (adapts as X changes during training)
    - Computationally efficient: GW only runs every update_interval steps
    """
    UNLOAD_LIST = ["text_encoder", "vae"]

    def initialize_vision_encoder(self):
        assert len(self.args.align_models) == 1, 'Currently support one alignment model'
        if self.args.align_models[0] == "VideoMAEv2":
            self.vision_encoder = vit_base_patch16_224().to(self.accelerator.device)
            self.vision_encoder.from_pretrained(ckpt("VideoMAEv2", "vit_b_k710_dl_from_giant.pth"))
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.require_grad = False
        elif self.args.align_models[0] == "VideoMAE":
            self.vision_encoder = VideoMAE_vit_base_patch16_224().to(self.accelerator.device)
            self.vision_encoder.from_pretrained(ckpt("VideoMAE", "k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth"))
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
            self.vision_encoder = load_VJEPA(device=self.accelerator.device, pretrained_path=ckpt("vjepa_l", "vitl16.pth.tar"))
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

    def initialize_dim_token_helper(self):
        """Initialize the dimension-guided token alignment helper."""
        token_sample_for_loss = getattr(self.args, 'dim_token_sample_size_for_loss', 0)
        self.dim_token_helper = DimTokenAlignmentHelper(
            reg=getattr(self.args, 'dim_gw_reg', 0.1),
            num_outer_iters=getattr(self.args, 'dim_gw_outer_iters', 30),
            num_sinkhorn_iters=getattr(self.args, 'dim_gw_sinkhorn_iters', 100),
            token_sample_size_for_loss=token_sample_for_loss,
            outer_tol=getattr(self.args, 'dim_gw_outer_tol', 1e-4),
            sinkhorn_tol=getattr(self.args, 'dim_gw_sinkhorn_tol', 1e-4),
            margin=getattr(self.args, 'margin', 0.0),
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

        # Initialize helper on first call
        if not hasattr(self, 'dim_token_helper'):
            self.initialize_dim_token_helper()

        # Pre-process for vision encoder
        B, C, F, H, W = raw_frames.shape
        raw_frames = raw_frames.transpose(1, 2).flatten(0, 1)  # B * F, C, H, W
        if self.args.align_models[0] in ['VideoMAEv2', 'VideoMAE', 'OminiMAE', "VJEPA", "VJEPA2"]:
            raw_frames = (raw_frames + 1.0) / 2.0
            raw_frames = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(raw_frames)
        else:
            raise NotImplementedError("Currently only supports VideoMAEv2/VideoMAE/OminiMAE/VJEPA/VJEPA2")
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

        # Compute dimension-guided token alignment loss (Plan A)
        # Flatten batch: (B*3600, 1920) and (B*3600, 768)
        align_flat = align.flatten(0, 1)  # (B*3600, 1920)
        align_target_flat = align_target.flatten(0, 1)  # (B*3600, 768)

        # Dimension-guided per-token cosine loss (uses GW transport plan only)
        proj_loss = self.dim_token_helper.compute_loss_batched(
            align_flat, align_target_flat, batch_size=batch_size
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


register("cogvideox-t2v-dim-token-align", "lora", CogVideoXT2VDimTokenAlignLoraTrainer)
