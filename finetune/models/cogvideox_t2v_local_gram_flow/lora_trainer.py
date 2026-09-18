"""
Local Gram Flow trainer for CogVideoX.
Aligns LOCAL structural dynamics (how local feature correlations change over time)
between student (CogVideoX) and teacher (VideoMAEv2), without any learnable projector.

Key insight: Instead of aligning absolute features (REPA) or global structure (Gram),
align how LOCAL feature structures CHANGE across time — capturing motion/dynamics
rather than static appearance.

Theoretical basis:
- Local Gram matrix captures spatial co-activation patterns within a patch
- Temporal difference of Gram captures how these patterns evolve (motion)
- Dimension-independent: student (1920d) and teacher (768d) produce same-shaped
  local Gram matrices (p*p × p*p), directly comparable without projector
"""
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.nn.functional as F_func  # alias to avoid shadowing by variable 'F' (num_frames)
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
)
from finetune.models.cogvideox_t2v_local_gram_flow.models.cogvideox_align import CogVideoXTransformer3DModelAlign, CogVideoXPipelineAlign
from finetune.models.cogvideox_t2v_local_gram_flow.models.ssl.VideoMAEv2 import vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch16_224, vit_giant_patch14_224
from finetune.models.cogvideox_t2v_local_gram_flow.models.ssl.VideoMAE import vit_base_patch16_224 as VideoMAE_vit_base_patch16_224

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


def compute_local_gram_flow_loss(student_feat, teacher_feat, patch_size_h=5, patch_size_w=5, 
                                  alpha=1.0, beta=0.0):
    """
    Compute Local Gram Flow alignment loss.
    
    Args:
        student_feat: [B, F, H, W, C_s]  (e.g. B, 24, 10, 15, 1920)
        teacher_feat: [B, F, H, W, C_t]  (e.g. B, 24, 10, 15, 768)
        patch_size_h: patch height (default 5, divides H=10 into 2 patches)
        patch_size_w: patch width (default 5, divides W=15 into 3 patches)
        alpha: weight for temporal flow loss (delta_G alignment)
        beta: weight for static Gram loss (G alignment, 0 = pure flow)
    
    Returns:
        loss: scalar tensor
    """
    B, T, H, W, C_s = student_feat.shape  # T=frames, avoid shadowing F (torch.nn.functional)
    C_t = teacher_feat.shape[-1]
    ph, pw = patch_size_h, patch_size_w
    
    assert H % ph == 0 and W % pw == 0, f"H={H} must be divisible by ph={ph}, W={W} by pw={pw}"
    
    num_patches_h = H // ph
    num_patches_w = W // pw
    num_patches = num_patches_h * num_patches_w  # e.g. 2*3=6
    tokens_per_patch = ph * pw  # e.g. 5*5=25
    
    # Reshape into patches: [B, T, num_patches_h, ph, num_patches_w, pw, C]
    # -> [B, T, num_patches, tokens_per_patch, C]
    def to_patches(x, C):
        x = x.reshape(B, T, num_patches_h, ph, num_patches_w, pw, C)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)  # [B, T, nph, npw, ph, pw, C]
        x = x.reshape(B, T, num_patches, tokens_per_patch, C)
        return x
    
    s_patches = to_patches(student_feat, C_s)  # [B, T, 6, 25, 1920]
    t_patches = to_patches(teacher_feat, C_t)  # [B, T, 6, 25, 768]
    
    # L2 normalize along feature dim -> cosine similarity basis
    # This makes Gram matrices dimension-independent
    s_patches = F_func.normalize(s_patches.float(), dim=-1)
    t_patches = F_func.normalize(t_patches.float(), dim=-1)
    
    # Compute local Gram matrices: [B, F, num_patches, tokens_per_patch, tokens_per_patch]
    # G[b,f,p,i,j] = cos_sim(token_i, token_j) within patch p at frame f
    G_s = torch.matmul(s_patches, s_patches.transpose(-1, -2))  # [B, F, 6, 25, 25]
    G_t = torch.matmul(t_patches, t_patches.transpose(-1, -2))  # [B, F, 6, 25, 25]
    
    loss = torch.tensor(0.0, device=student_feat.device, dtype=torch.float32)
    
    # === Temporal Flow Loss (main component) ===
    # Align how local structure CHANGES between consecutive frames
    if alpha > 0 and T > 1:
        delta_G_s = G_s[:, 1:] - G_s[:, :-1]  # [B, F-1, 6, 25, 25]
        delta_G_t = G_t[:, 1:] - G_t[:, :-1]  # [B, F-1, 6, 25, 25]
        flow_loss = F_func.mse_loss(delta_G_s, delta_G_t)
        loss = loss + alpha * flow_loss
    
    # === Static Gram Loss (optional, for spatial structure) ===
    # Align absolute local structure (not just changes)
    if beta > 0:
        static_loss = F_func.mse_loss(G_s, G_t)
        loss = loss + beta * static_loss
    
    return loss


def compute_multiscale_local_gram_flow_loss(student_feat, teacher_feat, 
                                             scales=None, alpha=1.0, beta=0.0):
    """
    Multi-scale Local Gram Flow: aggregate loss over multiple patch sizes.
    Captures motion at different spatial granularities simultaneously.
    
    Args:
        student_feat: [B, T, H, W, C_s]  (e.g. B, 24, 10, 15, 1920)
        teacher_feat: [B, T, H, W, C_t]  (e.g. B, 24, 10, 15, 768)
        scales: list of (ph, pw) tuples. Default: [(2,3), (5,5), (10,15)]
            - (2, 3): 5x5=25 patches, 6 tokens each → fine-grained local motion
            - (5, 5): 2x3=6 patches, 25 tokens each → medium region structure
            - (10,15): 1 patch = global, 150 tokens → global relation flow
        alpha: weight for temporal flow loss
        beta: weight for static Gram loss
    
    Returns:
        loss: scalar tensor (averaged over scales)
    """
    if scales is None:
        scales = [(2, 3), (5, 5), (10, 15)]
    
    total_loss = torch.tensor(0.0, device=student_feat.device, dtype=torch.float32)
    num_valid = 0
    
    for (ph, pw) in scales:
        H, W = student_feat.shape[2], student_feat.shape[3]
        if H % ph != 0 or W % pw != 0:
            continue  # skip incompatible scales
        total_loss = total_loss + compute_local_gram_flow_loss(
            student_feat, teacher_feat,
            patch_size_h=ph, patch_size_w=pw,
            alpha=alpha, beta=beta,
        )
        num_valid += 1
    
    if num_valid > 0:
        total_loss = total_loss / num_valid
    
    return total_loss


class CogVideoXT2VLocalGramFlowLoraTrainer(Trainer):
    """
    Trainer using Local Gram Flow alignment.
    Key design:
    - No MLP projector: features aligned in native dimensions via local Gram matrices
    - Aligns temporal CHANGES in local spatial structure (motion/dynamics)
    - Dimension-independent: cosine-normalized local Gram is the same shape regardless of C
    - Zero additional parameters, 100% gradient flows to transformer backbone
    """
    UNLOAD_LIST = ["text_encoder", "vae"]

    @override
    def prepare_trainable_parameters(self):
        """Override to register temporal diff projector BEFORE optimizer/DeepSpeed init."""
        super().prepare_trainable_parameters()
        # If temporal diff loss is enabled, register projector as transformer submodule
        # so DeepSpeed includes it in optimizer automatically.
        # dtype is NOT specified here — DeepSpeed will cast it to bf16 along with the transformer.
        lgf_temporal_diff_weight = getattr(self.args, 'lgf_temporal_diff_weight', 0.0)
        if lgf_temporal_diff_weight > 0:
            D_s = 1920  # CogVideoX-2B inner_dim = 30 heads * 64 dim
            D_t = self.args.align_dims[0] if hasattr(self.args, 'align_dims') else 768
            projector = nn.Sequential(
                nn.Linear(D_s, 2048), nn.SiLU(),
                nn.Linear(2048, 2048), nn.SiLU(),
                nn.Linear(2048, D_t),
            )
            self.components.transformer._temporal_diff_projector = projector
            self.components.transformer._temporal_diff_projector.requires_grad_(True)
            print(f"[LGF] Registered temporal diff projector: {D_s} -> {D_t} (before optimizer init)")

    def initialize_vision_encoder(self):
        """Lazy initialization: mark for loading, actual load happens in first compute_loss.
        This avoids CUDA init before DataLoader workers fork."""
        self._vision_encoder_initialized = False
        self.vision_encoder = None

    def _lazy_init_vision_encoder(self):
        """Actually load vision encoder to GPU. Called once from compute_loss."""
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
            from finetune.models.cogvideox_t2v_local_gram_flow.models.ssl.omini_mae import vit_base_mae_pretraining
            self.vision_encoder = vit_base_mae_pretraining().to(self.accelerator.device)
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.require_grad = False
            self.vision_encoder.tubelet_size = 2
            self.vision_encoder.patch_size = 16
            self.vision_encoder.embed_dim = 768
        elif self.args.align_models[0] == 'VJEPA':
            from finetune.models.cogvideox_t2v_local_gram_flow.models.ssl.JEPA import load_VJEPA
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
        self._vision_encoder_initialized = True

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipelineAlign

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        # No projector needed, but still pass align_layer for feature extraction
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

        # If temporal diff cosine is enabled, attach projector to transformer
        # so it gets registered with accelerator.prepare() and the optimizer
        if getattr(self.args, 'lgf_temporal_diff_weight', 0.0) > 0:
            D_s = components.transformer.inner_dim  # 1920
            D_t = self.args.align_dims[0]  # 768
            components.transformer._td_projector = nn.Sequential(
                nn.Linear(D_s, 2048), nn.SiLU(),
                nn.Linear(2048, 2048), nn.SiLU(),
                nn.Linear(2048, D_t),
            )
            print(f"[LGF] Attached temporal diff projector to transformer: {D_s} -> {D_t}")

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
        # Lazy-init vision encoder on first call (after DataLoader workers have forked)
        if not self._vision_encoder_initialized:
            self._lazy_init_vision_encoder()

        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]
        raw_frames = batch["raw_frames"]  # [B, C, F, H, W] value range [-1, 1]

        # Pre-process for vision encoder
        B, C, F, H, W = raw_frames.shape
        raw_frames = raw_frames.transpose(1, 2).flatten(0, 1)  # B * F, C, H, W
        if self.args.align_models[0] in ['VideoMAEv2', 'VideoMAE', 'OminiMAE', "VJEPA", "VJEPA2"]:
            raw_frames = (raw_frames + 1.0) / 2.0
            raw_frames = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(raw_frames)
        else:
            raise NotImplementedError("Local Gram Flow currently only supports VideoMAEv2/VideoMAE/OminiMAE/VJEPA/VJEPA2")
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
            # encoding: B, 3, 48, 160, 240 -> B, 24x10x15, D (D=768 for ViT-B)
            align_target = self.vision_encoder(repa_raw_frames)
            # B, 24x10x15, D -> B, D, 24, 10, 15
            align_target = align_target.transpose(1, 2).reshape(B, -1, F // self.vision_encoder.tubelet_size, H // self.vision_encoder.patch_size, W // self.vision_encoder.patch_size)

        # Reshape target to [B, F_t, H_t, W_t, D]: (B, 24, 10, 15, 768)
        D_t = align_target.shape[1]
        F_t = F // self.vision_encoder.tubelet_size  # 24
        H_t = H // self.vision_encoder.patch_size    # 10
        W_t = W // self.vision_encoder.patch_size    # 15
        align_target_spatial = align_target.permute(0, 2, 3, 4, 1)  # [B, 24, 10, 15, 768]

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

        # === Zero-parameter spatial preprocessing ===
        align = align.permute(0, 4, 1, 2, 3)  # B, inner_dim, 12, 30, 45
        # Temporal upsample: 12 -> 24 to match VideoMAEv2 frame count
        align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')
        B_a, C_a, F_a, H_a, W_a = align.shape  # B, 1920, 24, 30, 45

        # Spatial downsampling via Conv2d (stride=3, learnable):
        # 30x45 -> 10x15, matching teacher's spatial resolution.
        align_2d = align.permute(0, 2, 1, 3, 4).reshape(B_a * F_a, C_a, H_a, W_a)  # (B*24, 1920, 30, 45)
        align_pooled = self.components.transformer.downsampler_cogvideo_output(align_2d.to(torch.bfloat16))  # (B*24, 1920, 10, 15)
        align_pooled = align_pooled.reshape(B_a, F_a, C_a, H_a // 3, W_a // 3)  # B, 24, 1920, 10, 15

        # Reshape student to [B, F, H, W, C]: (B, 24, 10, 15, 1920)
        student_spatial = align_pooled.permute(0, 1, 3, 4, 2)  # [B, 24, 10, 15, 1920]

        # === Local Gram Flow Loss ===
        # Get hyperparameters
        patch_size_h = getattr(self.args, 'lgf_patch_size_h', 5)
        patch_size_w = getattr(self.args, 'lgf_patch_size_w', 5)
        lgf_alpha = getattr(self.args, 'lgf_alpha', 1.0)  # temporal flow weight
        lgf_beta = getattr(self.args, 'lgf_beta', 0.0)    # static Gram weight
        lgf_multiscale = getattr(self.args, 'lgf_multiscale', False)
        lgf_temporal_diff_weight = getattr(self.args, 'lgf_temporal_diff_weight', 0.0)

        if lgf_multiscale:
            proj_loss = compute_multiscale_local_gram_flow_loss(
                student_feat=student_spatial,
                teacher_feat=align_target_spatial,
                alpha=lgf_alpha,
                beta=lgf_beta,
            )
        else:
            proj_loss = compute_local_gram_flow_loss(
                student_feat=student_spatial,
                teacher_feat=align_target_spatial,
                patch_size_h=patch_size_h,
                patch_size_w=patch_size_w,
                alpha=lgf_alpha,
                beta=lgf_beta,
            )

        # === Optional: Temporal Diff Cosine Loss (per-token motion direction alignment) ===
        if lgf_temporal_diff_weight > 0:
            # Use projector attached to transformer (registered via accelerator.prepare)
            td_projector = self.components.transformer._td_projector

            # Project student to 768d for temporal diff cosine
            student_projected = td_projector(student_spatial.to(torch.bfloat16))  # [B, 24, 10, 15, 768]
            teacher_for_diff = align_target_spatial.to(torch.bfloat16)  # [B, 24, 10, 15, 768]

            # Compute frame differences
            student_diff = student_projected[:, 1:] - student_projected[:, :-1]  # [B, 23, 10, 15, 768]
            teacher_diff = teacher_for_diff[:, 1:] - teacher_for_diff[:, :-1]    # [B, 23, 10, 15, 768]

            # Flatten and per-token cosine loss
            s_diff_flat = student_diff.reshape(-1, student_diff.shape[-1])  # [B*23*150, 768]
            t_diff_flat = teacher_diff.reshape(-1, teacher_diff.shape[-1])  # [B*23*150, 768]
            s_diff_norm = F_func.normalize(s_diff_flat.float(), dim=-1)
            t_diff_norm = F_func.normalize(t_diff_flat.float(), dim=-1)
            cos_sim = (s_diff_norm * t_diff_norm).sum(dim=-1)
            temporal_diff_loss = (1.0 - cos_sim).mean()

            proj_loss = proj_loss + lgf_temporal_diff_weight * temporal_diff_loss

        # === Optional: Global Gram Loss (pre-projector, in native 1920d space) ===
        lgf_gram_weight = getattr(self.args, 'lgf_gram_weight', 0.0)
        if lgf_gram_weight > 0:
            # Compute global Gram matrices in NATIVE space (before any projector)
            # student_spatial: [B, 24, 10, 15, 1920], teacher: [B, 24, 10, 15, 768]
            # Flatten to [B, 3600, C], normalize, compute Gram [B, 3600, 3600]
            B_g = student_spatial.shape[0]
            s_flat = student_spatial.reshape(B_g, -1, C_a).float()   # [B, 3600, 1920]
            t_flat = align_target_spatial.reshape(B_g, -1, align_target_spatial.shape[-1]).float()  # [B, 3600, 768]
            s_norm = F_func.normalize(s_flat, dim=-1)
            t_norm = F_func.normalize(t_flat, dim=-1)
            # Gram matrices (dimension-independent: both become [B, 3600, 3600])
            gram_s = torch.bmm(s_norm, s_norm.transpose(1, 2))  # [B, 3600, 3600]
            gram_t = torch.bmm(t_norm, t_norm.transpose(1, 2))  # [B, 3600, 3600]
            gram_loss = F_func.mse_loss(gram_s, gram_t)
            proj_loss = proj_loss + lgf_gram_weight * gram_loss

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


register("cogvideox-t2v-local-gram-flow", "lora", CogVideoXT2VLocalGramFlowLoraTrainer)
register("cogvideox-t2v-local-gram-flow-align", "lora", CogVideoXT2VLocalGramFlowLoraTrainer)
