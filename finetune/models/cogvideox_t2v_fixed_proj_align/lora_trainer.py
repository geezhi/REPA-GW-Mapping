"""
Fixed Random Projection alignment trainer for CogVideoX.

Key idea: Use a FROZEN random linear projection W: (D1=1920) -> (D2=768) to map
student features into teacher's dimension space, then do per-token cosine alignment.

This solves the "projector gradient absorption" problem:
- Unlike a learnable projector, W is fixed (never updated) -> 100% gradient flows to backbone
- Unlike Gram matrix alignment, this provides per-token feature-level supervision
  so the model learns actual dimensional semantics, not just relational structure

Theoretical justification: Johnson-Lindenstrauss lemma guarantees that random
projections approximately preserve pairwise distances. So W provides a valid
(approximate) mapping between the two spaces without needing to be learned.

Token downsampling: downsampler_cogvideo_output (30x45 -> 10x15), learnable Conv2d stride=3.
"""
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
)
from finetune.models.cogvideox_t2v_fixed_proj_align.models.cogvideox_align import CogVideoXTransformer3DModelAlign, CogVideoXPipelineAlign
from finetune.models.cogvideox_t2v_fixed_proj_align.models.ssl.VideoMAEv2 import vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch16_224, vit_giant_patch14_224
from finetune.models.cogvideox_t2v_fixed_proj_align.models.ssl.VideoMAE import vit_base_patch16_224 as VideoMAE_vit_base_patch16_224

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


class CogVideoXT2VFixedProjAlignLoraTrainer(Trainer):
    """
    Trainer using a FROZEN random linear projection for cross-space alignment.

    Key properties:
    - Fixed random W: (1920, 768), initialized once, NEVER trained
    - Token downsampling: downsampler_cogvideo_output (30x45 -> 10x15), learnable Conv2d stride=3
    - Loss: per-token cosine similarity between projected student and teacher
    - Gradient: 100% flows to transformer backbone (W is frozen, avg_pool has no params)
    - Learns: dimensional semantics (unlike Gram matrix which only learns structure)
    """
    UNLOAD_LIST = ["text_encoder", "vae"]

    def initialize_fixed_projection(self):
        """Initialize a frozen random projection matrix W: (1920, 768).
        
        Uses Kaiming-style initialization (scaled Gaussian) for stable magnitude.
        W is registered as a buffer (saved with model state but never updated).
        """
        D1 = 1920  # student dim (CogVideoX-2B inner_dim)
        D2 = self.args.align_dims[0]  # teacher dim (768 for VideoMAEv2-B)
        # Xavier/Kaiming-style: scale by sqrt(2 / (D1 + D2)) for balanced magnitude
        scale = (2.0 / (D1 + D2)) ** 0.5
        W = torch.randn(D1, D2, device=self.accelerator.device) * scale
        # Frozen: will not receive gradients
        self.fixed_proj_W = W.detach()  # (1920, 768)
        print(f"[FixedProjAlign] Initialized frozen random projection W: ({D1}, {D2}), scale={scale:.6f}")
        print(f"[FixedProjAlign] W norm: {W.norm().item():.4f}, mean: {W.mean().item():.6f}, std: {W.std().item():.6f}")

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



    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipelineAlign

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        # GW mode: no projector needed, but still pass align_layer for feature extraction
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

        # Initialize fixed projection on first call
        if not hasattr(self, 'fixed_proj_W'):
            self.initialize_fixed_projection()

        # Pre-process for vision encoder
        B, C, F, H, W = raw_frames.shape
        raw_frames = raw_frames.transpose(1, 2).flatten(0, 1)  # B * F, C, H, W
        if self.args.align_models[0] in ['VideoMAEv2', 'VideoMAE', 'OminiMAE', "VJEPA", "VJEPA2"]:
            raw_frames = (raw_frames + 1.0) / 2.0
            raw_frames = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(raw_frames)
        else:
            raise NotImplementedError("GW alignment currently only supports VideoMAEv2/VideoMAE/OminiMAE/VJEPA/VJEPA2")
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

        # Flatten target: B, F*H*W, D
        align_target = align_target.flatten(2).transpose(1, 2)  # B, 24x10x15, D

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
        # Temporal upsample: 12 -> 24 to match VideoMAEv2 frame count
        align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')
        B_a, C_a, F_a, H_a, W_a = align.shape  # B, 1920, 24, 30, 45

        # Spatial downsampling via downsampler_cogvideo_output (learnable Conv2d stride=3):
        # 30x45 -> 10x15, matching teacher's spatial resolution.
        align_2d = align.permute(0, 2, 1, 3, 4).reshape(B_a * F_a, C_a, H_a, W_a)  # (B*24, 1920, 30, 45)
        align_pooled = self.components.transformer.downsampler_cogvideo_output(align_2d.to(torch.bfloat16))  # 30x45 -> 10x15
        align_pooled = align_pooled.reshape(B_a, F_a, C_a, H_a // 3, W_a // 3)  # B, 24, 1920, 10, 15

        # Flatten to (B, N_tokens, D): student (B, 3600, 1920), teacher (B, 3600, 768)
        align_flat = align_pooled.permute(0, 1, 3, 4, 2).reshape(B_a, -1, C_a)  # (B, 3600, 1920)
        align_target_reshaped = align_target  # already (B, 3600, 768)

        # === Fixed Random Projection + Per-Token Cosine Alignment ===
        # Project student features into teacher's dimension space using frozen W.
        # W is (1920, 768), NEVER updated. Gradient flows through the matmul back
        # to align_flat (and thus to the transformer backbone) but NOT to W.
        # This gives per-token feature-level supervision (learns dimensional semantics)
        # while keeping 100% gradient flowing to the backbone.
        W = self.fixed_proj_W  # (1920, 768), frozen

        # Project: (B, 3600, 1920) @ (1920, 768) -> (B, 3600, 768)
        align_projected = align_flat.float() @ W  # gradient flows to align_flat, not W

        if self.args.loss == 'fixed_proj_temporal_diff':
            # === Temporal Difference mode ===
            # Instead of aligning absolute features, align FRAME DIFFERENCES:
            # delta_student[t] = student[t] - student[t-1]
            # delta_teacher[t] = teacher[t] - teacher[t-1]
            # This captures motion/dynamics rather than static appearance.
            F_align = F_a  # 24 frames
            H_align = H_a // 3  # 10
            W_align = W_a // 3  # 15
            tokens_per_frame = H_align * W_align  # 150

            # Reshape to [B, F, HW, 768]
            student_frames = align_projected.reshape(B_a, F_align, tokens_per_frame, -1)  # [B, 24, 150, 768]
            teacher_frames = align_target_reshaped.float().reshape(B_a, F_align, tokens_per_frame, -1)  # [B, 24, 150, 768]

            # Compute temporal differences: frame[t] - frame[t-1], for t=1..F-1
            student_diff = student_frames[:, 1:] - student_frames[:, :-1]  # [B, 23, 150, 768]
            teacher_diff = teacher_frames[:, 1:] - teacher_frames[:, :-1]  # [B, 23, 150, 768]

            # Flatten back and compute per-token cosine loss on differences
            student_diff_flat = student_diff.reshape(B_a, -1, student_diff.shape[-1])  # [B, 23*150, 768]
            teacher_diff_flat = teacher_diff.reshape(B_a, -1, teacher_diff.shape[-1])  # [B, 23*150, 768]

            proj_norm = torch.nn.functional.normalize(student_diff_flat, dim=-1)
            tgt_norm = torch.nn.functional.normalize(teacher_diff_flat, dim=-1)

            cos_sim = (proj_norm * tgt_norm).sum(dim=-1)  # [B, 23*150]
            proj_loss = (1.0 - cos_sim).mean()
        else:
            # === Original absolute alignment mode (fixed_proj_align) ===
            # Per-token cosine similarity loss
            proj_norm = torch.nn.functional.normalize(align_projected, dim=-1)  # (B, 3600, 768)
            tgt_norm = torch.nn.functional.normalize(align_target_reshaped.float(), dim=-1)  # (B, 3600, 768)

            # Cosine loss: mean(1 - cos(projected_student_i, teacher_i))
            cos_sim = (proj_norm * tgt_norm).sum(dim=-1)  # (B, 3600)
            proj_loss = (1.0 - cos_sim).mean()

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


register("cogvideox-t2v-fixed-proj-align", "lora", CogVideoXT2VFixedProjAlignLoraTrainer)
register("cogvideox-t2v-fixed-proj-align", "sft", CogVideoXT2VFixedProjAlignLoraTrainer)
