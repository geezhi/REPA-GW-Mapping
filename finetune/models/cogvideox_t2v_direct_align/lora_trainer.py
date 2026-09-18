"""
Direct relational alignment trainer for CogVideoX.
Since tokens from CogVideoX and VideoMAEv2 are spatially aligned (same 24x10x15 grid),
we directly compute relational (similarity) matrices and align them without GW.
This avoids the expensive GW transport plan computation entirely.
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


def compute_direct_relational_loss(X, Y, sample_size=None):
    """
    Compute direct relational alignment loss between spatially-aligned tokens.

    Since tokens are in the same spatial order (both are 24x10x15 grids),
    we directly compare their intra-space similarity matrices.

    Args:
        X: (N, D1) source features (CogVideoX, D1=1920), requires grad
        Y: (N, D2) target features (VideoMAEv2, D2=768), no grad
        sample_size: if > 0, subsample tokens to reduce memory for similarity matrix

    Returns:
        loss: scalar tensor (differentiable w.r.t. X)
    """
    N = X.shape[0]

    # Optional subsampling for memory efficiency (similarity matrix is N x N)
    if sample_size is not None and sample_size > 0 and sample_size < N:
        indices = torch.randperm(N, device=X.device)[:sample_size]
        X = X[indices]
        Y = Y[indices]

    # Compute cosine similarity matrices (dimension-independent)
    X_norm = F.normalize(X.float(), dim=-1)  # (S, D1)
    Y_norm = F.normalize(Y.float(), dim=-1)  # (S, D2)

    sim_X = X_norm @ X_norm.T  # (S, S) - source similarity structure
    sim_Y = Y_norm @ Y_norm.T  # (S, S) - target similarity structure

    # Direct relational loss: MSE between similarity matrices
    # Since tokens are spatially aligned, sim_X[i,j] should match sim_Y[i,j]
    loss = F.mse_loss(sim_X, sim_Y)

    return loss


class CogVideoXT2VDirectAlignLoraTrainer(Trainer):
    """
    Trainer using direct relational alignment (no GW, no projector).

    Key insight: CogVideoX and VideoMAEv2 tokens are spatially aligned after
    resolution matching (both produce 24x10x15 = 3600 tokens in the same order).
    Therefore, we can directly compare their relational structures without
    computing expensive GW transport plans.

    Advantages over GW-based alignment:
    - No expensive iterative optimization (50 outer x 100 Sinkhorn iterations)
    - No projector MLP needed (dimension-free relational comparison)
    - Simpler, faster, and theoretically equivalent when T ≈ Identity
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

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipelineAlign

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        # Direct align mode: no projector needed, but still pass align_layer for feature extraction
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

        # Pre-process for vision encoder
        B, C, F, H, W = raw_frames.shape
        raw_frames = raw_frames.transpose(1, 2).flatten(0, 1)  # B * F, C, H, W
        if self.args.align_models[0] in ['VideoMAEv2', 'VideoMAE', 'OminiMAE', "VJEPA", "VJEPA2"]:
            raw_frames = (raw_frames + 1.0) / 2.0
            raw_frames = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(raw_frames)
        else:
            raise NotImplementedError("Direct alignment currently only supports VideoMAEv2/VideoMAE/OminiMAE/VJEPA/VJEPA2")
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

        # Flatten target: B, N_tokens, D  (N_tokens = 24*10*15 = 3600)
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

        # Flatten: B, N_tokens, inner_dim  (N_tokens = 24*10*15 = 3600)
        align = align.flatten(2).transpose(1, 2)  # B, 3600, inner_dim=1920
        # align_target is already: B, 3600, D=768

        # Compute direct relational alignment loss (per sample, then average)
        # Token i in align corresponds to token i in align_target (same spatial position)
        sample_size = getattr(self.args, 'direct_align_sample_size', 0)
        total_proj_loss = 0.0
        for b in range(batch_size):
            total_proj_loss = total_proj_loss + compute_direct_relational_loss(
                align[b], align_target[b], sample_size=sample_size if sample_size > 0 else None
            )
        proj_loss = total_proj_loss / batch_size

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


register("cogvideox-t2v-direct-align", "lora", CogVideoXT2VDirectAlignLoraTrainer)
