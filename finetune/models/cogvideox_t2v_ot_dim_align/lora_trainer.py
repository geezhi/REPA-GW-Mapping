"""
OT Dimension Alignment trainer for CogVideoX.

Key idea: Use standard Optimal Transport (Sinkhorn) to find dimension-level
transport plan T: (D1, D2) by directly computing cross-space cost matrix.

Since tokens are spatially aligned (same resolution after downsampling),
we can directly compute cosine distance between student dim i and teacher dim j
using their corresponding token activations. No need for Gromov-Wasserstein.

This is "OT used in reverse":
- Dimensions are the "mass" being transported
- Tokens are the "features" describing each dimension's behavior
- Cost[i,j] = cosine_distance(student_dim_i, teacher_dim_j) in token space

After finding T, we row-normalize it and use it to map teacher into student's
dimension space, then compute per-token cosine alignment loss.
"""
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
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
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from finetune.paths import ckpt


# ============================================================================
# OT utilities for dimension-level transport plan computation
# ============================================================================

def compute_cross_space_cost(X, Y):
    """
    Compute cross-space cost matrix between dimensions using token correspondences.
    
    X: (N, D1) student features — N tokens, D1 dims
    Y: (N, D2) teacher features — N tokens, D2 dims (same N, spatially aligned)
    
    Returns:
        C: (D1, D2) cost matrix, C[i,j] = cosine_distance(X[:,i], Y[:,j])
    """
    # Normalize each column (dimension) to unit norm
    X_col_norm = F.normalize(X, dim=0)  # (N, D1), each column has unit norm
    Y_col_norm = F.normalize(Y, dim=0)  # (N, D2), each column has unit norm
    
    # Cross-space cosine similarity: (D1, D2)
    sim = X_col_norm.T @ Y_col_norm  # (D1, N) @ (N, D2) -> (D1, D2)
    
    # Cosine distance
    cost = 1.0 - sim  # in [0, 2]
    return cost


def sinkhorn_log(C, reg, a=None, b=None, num_iters=100, tol=1e-4):
    """
    Sinkhorn algorithm in log-domain for standard OT.
    
    C: (D1, D2) cost matrix
    reg: entropic regularization
    a: (D1,) source marginal (uniform if None)
    b: (D2,) target marginal (uniform if None)
    
    Returns:
        T: (D1, D2) transport plan
    """
    D1, D2 = C.shape
    device = C.device
    dtype = C.dtype

    if a is None:
        a = torch.ones(D1, device=device, dtype=dtype) / D1
    if b is None:
        b = torch.ones(D2, device=device, dtype=dtype) / D2

    log_a = torch.log(a + 1e-20)
    log_b = torch.log(b + 1e-20)
    log_K = -C / reg  # (D1, D2)

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


@torch.no_grad()
def compute_ot_dim_transport_plan(X, Y, reg=0.1, num_iters=100, tol=1e-4):
    """
    Compute OT transport plan between dimensions using direct cross-space cost.
    
    X: (N, D1) student features (detached)
    Y: (N, D2) teacher features (detached)
    
    Returns:
        T: (D1, D2) transport plan
    """
    # Direct cross-space cost matrix
    C = compute_cross_space_cost(X.float(), Y.float())  # (D1, D2)
    
    # Sinkhorn OT
    T = sinkhorn_log(C, reg=reg, num_iters=num_iters, tol=tol)
    
    return T


def compute_ot_dim_token_loss(X, Y, T_dim, margin=0.0):
    """
    Use OT dimension transport plan to map Y into X's space,
    then compute per-token cosine alignment loss.
    
    X: (N, D1) student features (requires grad)
    Y: (N, D2) teacher features (no grad)
    T_dim: (D1, D2) transport plan (detached)
    margin: cosine distance threshold below which no loss is applied
    
    Returns:
        loss: scalar tensor (differentiable w.r.t. X)
    """
    # Row-normalize T: each row sums to 1
    T_float = T_dim.float()
    T_row_norm = T_float / (T_float.sum(dim=1, keepdim=True) + 1e-20)  # (D1, D2)

    # Map Y into X's dimension space
    Y_mapped = Y.float() @ T_row_norm.T  # (N, D2) @ (D2, D1) -> (N, D1)

    # Per-token cosine loss
    X_norm = F.normalize(X.float(), dim=-1)  # (N, D1)
    Y_mapped_norm = F.normalize(Y_mapped, dim=-1)  # (N, D1)

    cos_sim = (X_norm * Y_mapped_norm).sum(dim=-1)  # (N,)
    cos_dist = 1.0 - cos_sim

    if margin > 0:
        loss = torch.nn.functional.relu(cos_dist - margin).mean()
    else:
        loss = cos_dist.mean()

    return loss


class OTDimAlignmentHelper:
    """
    Helper class for OT-based dimension alignment.
    
    Directly computes cross-space cost matrix (no GW needed since tokens are aligned),
    then uses Sinkhorn OT to find dimension transport plan.
    """

    def __init__(self, reg=0.1, num_iters=100, tol=1e-4, margin=0.0):
        self.reg = reg
        self.num_iters = num_iters
        self.tol = tol
        self.margin = margin

    def compute_loss(self, X, Y):
        """
        Compute OT-based per-token alignment loss for a single sample.
        
        X: (N, D1) student features (requires grad)
        Y: (N, D2) teacher features (no grad)
        """
        T = compute_ot_dim_transport_plan(
            X.detach(), Y.detach(),
            reg=self.reg, num_iters=self.num_iters, tol=self.tol
        )
        loss = compute_ot_dim_token_loss(X, Y, T.detach(), margin=self.margin)
        return loss

    def compute_loss_batched(self, X, Y, batch_size):
        """
        Compute loss for batched inputs. OT is computed per sample.
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

class CogVideoXT2VOTDimAlignLoraTrainer(Trainer):
    """
    Trainer using OT-based dimension alignment.
    
    Key differences from GW-based dim_token_align:
    - Uses direct cross-space cost matrix (cosine distance between dims)
    - Only needs ONE Sinkhorn solve (no GW outer loop)
    - ~50x faster than GW for the same result quality
    - More principled: directly uses token correspondences
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
        else:
            raise NotImplementedError

    def initialize_ot_helper(self):
        """Initialize OT dimension alignment helper."""
        self.ot_helper = OTDimAlignmentHelper(
            reg=getattr(self.args, 'ot_reg', 0.1),
            num_iters=getattr(self.args, 'ot_sinkhorn_iters', 100),
            tol=getattr(self.args, 'ot_tol', 1e-4),
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
        components.transformer = CogVideoXTransformer3DModelAlign.from_pretrained(
            model_path, subfolder="transformer", **load_kwargs
        )
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
            ret["encoded_videos"].append(sample["encoded_video"])
            ret["prompt_embedding"].append(sample["prompt_embedding"])
            ret["raw_frames"].append(sample["raw_frames"])
        ret["encoded_videos"] = torch.stack(ret["encoded_videos"])
        ret["prompt_embedding"] = torch.stack(ret["prompt_embedding"])
        ret["raw_frames"] = torch.stack(ret["raw_frames"])
        return ret

    @override
    def compute_loss(self, batch) -> torch.Tensor:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        raw_frames = batch["raw_frames"]

        # Initialize helper on first call
        if not hasattr(self, 'ot_helper'):
            self.initialize_ot_helper()

        # Pre-process for vision encoder
        B, C, F, H, W = raw_frames.shape
        raw_frames = raw_frames.transpose(1, 2).flatten(0, 1)
        if self.args.align_models[0] in ['VideoMAEv2', 'VideoMAE']:
            raw_frames = (raw_frames + 1.0) / 2.0
            raw_frames = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(raw_frames)
        else:
            raise NotImplementedError
        raw_frames = raw_frames.reshape(B, F, C, H, W).transpose(1, 2)

        # Remove first frame, resize
        repa_raw_frames = raw_frames[:, :, 1:]
        B, C, F, H, W = repa_raw_frames.shape
        repa_raw_frames = repa_raw_frames.transpose(1, 2).flatten(0, 1)
        repa_raw_frames = torch.nn.functional.interpolate(repa_raw_frames, (H // 3, W // 3), mode='bicubic')
        repa_raw_frames = repa_raw_frames.reshape(B, F, C, H // 3, W // 3).transpose(1, 2)

        # Encode with frozen vision encoder
        with torch.no_grad():
            B, C, F, H, W = repa_raw_frames.shape
            align_target = self.vision_encoder(repa_raw_frames)
            align_target = align_target.transpose(1, 2).reshape(
                B, -1, F // self.vision_encoder.tubelet_size,
                H // self.vision_encoder.patch_size, W // self.vision_encoder.patch_size
            )

        # Flatten target: B, 3600, 768
        align_target = align_target.flatten(2).transpose(1, 2)

        # Prepare latent for diffusion
        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            raise NotImplementedError

        batch_size, num_channels, num_frames, height, width = latent.shape
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Sample timestep, add noise
        num_train_timesteps = self.components.scheduler.config.num_train_timesteps
        timesteps = torch.randint(0, num_train_timesteps, (batch_size,), device=self.accelerator.device).long()
        latent = latent.permute(0, 2, 1, 3, 4)
        noise = torch.randn_like(latent)
        latent_added_noise = self.components.scheduler.add_noise(latent, noise, timesteps)

        # Rotary embeddings
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

        # Forward pass
        predicted_noises, aligns = self.components.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )
        predicted_noise = predicted_noises[0]

        # Process alignment features
        align = aligns[0]
        align = align.reshape(B, 13, 60 // 2, 90 // 2, -1)  # B, 13, 30, 45, 1920
        align = align[:, 1:]  # remove first frame -> B, 12, 30, 45, 1920

        # Spatial downsampling + temporal upsample
        align = align.permute(0, 4, 1, 2, 3)  # B, 1920, 12, 30, 45
        align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')
        B_a, C_a, F_a, H_a, W_a = align.shape
        align = align.permute(0, 2, 1, 3, 4).reshape(B_a * F_a, C_a, H_a, W_a)
        align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))
        align = align.reshape(B_a, F_a, C_a, H_a // 3, W_a // 3).permute(0, 2, 1, 3, 4)

        # Flatten: B, 3600, 1920
        align = align.flatten(2).transpose(1, 2)

        # Compute OT dimension alignment loss
        align_flat = align.flatten(0, 1)  # (B*3600, 1920)
        align_target_flat = align_target.flatten(0, 1)  # (B*3600, 768)

        proj_loss = self.ot_helper.compute_loss_batched(
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
        self, height, width, num_frames, transformer_config, vae_scale_factor_spatial, device
    ):
        grid_height = height // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_width = width // (vae_scale_factor_spatial * transformer_config.patch_size)
        base_size_width = 720 // (vae_scale_factor_spatial * transformer_config.patch_size)
        base_size_height = 480 // (vae_scale_factor_spatial * transformer_config.patch_size)
        grid_crops_coords = (0, 0, base_size_height, base_size_width)
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )
        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

