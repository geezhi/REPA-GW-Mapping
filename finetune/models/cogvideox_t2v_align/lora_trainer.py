from typing import Any, Dict, List, Tuple

import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
    CogVideoXDDIMScheduler,
    # CogVideoXPipeline,
    # CogVideoXTransformer3DModel,
)
from finetune.models.cogvideox_t2v_align.models.cogvideox_align import CogVideoXTransformer3DModelAlign, CogVideoXPipelineAlign
from finetune.models.cogvideox_t2v_align.models.ssl.VideoMAEv2 import vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch16_224, vit_giant_patch14_224
from finetune.models.cogvideox_t2v_align.models.ssl.VideoMAE import vit_base_patch16_224 as VideoMAE_vit_base_patch16_224

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

class CogVideoXT2VAlignLoraTrainer(Trainer):
    UNLOAD_LIST = ["vae"]  # Keep text_encoder on GPU for uncached samples

    def initialize_vision_encoder(self):
        assert len(self.args.align_models) == 1, 'Currently support one alignment model'
        if self.args.align_models[0] == "VideoMAEv2":
            self.vision_encoder = vit_base_patch16_224().to(self.accelerator.device)
            self.vision_encoder.from_pretrained('/efs/zixianhuang/ckpt/VideoMAEv2/vit_b_k710_dl_from_giant.pth')  # The from pretrained return None
            # freeze the parameter
            self.vision_encoder.eval()
            # Actually no need to set False because it is not going through the optimizer
            for param in self.vision_encoder.parameters():  
                param.require_grad = False
            # Load teacher PCA matrix if specified
            if self.args.teacher_pca_path is not None:
                pca_data = torch.load(self.args.teacher_pca_path, map_location=self.accelerator.device)
                self.teacher_pca_W = pca_data["W_pca"].to(self.accelerator.device)  # (2304, 1920)
                self.teacher_pca_mean = pca_data["mean"].to(self.accelerator.device)  # (2304,)
                self.teacher_pca_layer_indices = pca_data["layer_indices"]
                print(f"Loaded teacher PCA matrix: {self.teacher_pca_W.shape}, layers={self.teacher_pca_layer_indices}")
        elif self.args.align_models[0] == "VideoMAE":
            self.vision_encoder = VideoMAE_vit_base_patch16_224().to(self.accelerator.device)
            self.vision_encoder.from_pretrained('/efs/zixianhuang/ckpt/VideoMAE/k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth')
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.require_grad = False
        elif self.args.align_models[0] == 'OminiMAE':
            from finetune.models.cogvideox_t2v_align.models.ssl.omini_mae import vit_base_mae_pretraining
            self.vision_encoder = vit_base_mae_pretraining().to(self.accelerator.device)
            self.vision_encoder.eval()
            for param in self.vision_encoder.parameters():
                param.require_grad = False          
            self.vision_encoder.tubelet_size = 2
            self.vision_encoder.patch_size = 16
            self.vision_encoder.embed_dim = 768    
        elif self.args.align_models[0] == 'DINOv2':
            self.vision_encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14').to(self.accelerator.device)
            self.vision_encoder.eval()
            del self.vision_encoder.head
            self.vision_encoder.head = torch.nn.Identity()
            for param in self.vision_encoder.parameters():
                param.require_grad = False 
        elif self.args.align_models[0] == 'VJEPA':
            from finetune.models.cogvideox_t2v_align.models.ssl.JEPA import load_VJEPA
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

    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipelineAlign

        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")

        components.text_encoder = T5EncoderModel.from_pretrained(model_path, subfolder="text_encoder")

        # add interface for the parameter of feature alignment
        load_kwargs = dict(align_layer=self.args.align_layer, align_dims=self.args.align_dims, projector_dim=self.args.projector_dim, align_residual=self.args.align_residual, align_attn_residual=self.args.align_attn_residual)
        if self.args.align_layer_secondary is not None:
            load_kwargs["align_layer_secondary"] = self.args.align_layer_secondary
        components.transformer = CogVideoXTransformer3DModelAlign.from_pretrained(model_path, subfolder="transformer", **load_kwargs)
        
        # If using teacher PCA, skip projector entirely (align in 1920d space)
        if self.args.teacher_pca_path is not None:
            components.transformer.skip_projector = True
            print("Teacher PCA mode: projector skipped, aligning in 1920d space.")

        # Dimension-independent relation losses (token similarity matrix / CKA):
        # use RAW student features in the native space, no projector needed (C_s may != C_t).
        if self.args.loss in ['token_similarity_matrix', 'cka_alignment']:
            components.transformer.skip_projector = True
            print("Dimension-independent relation loss: projector skipped, aligning in native student/teacher spaces.")

        # Load pretrained projector if specified
        elif self.args.pretrained_projector_path is not None:
            print(f"Loading pretrained projector from {self.args.pretrained_projector_path}")
            ckpt = torch.load(self.args.pretrained_projector_path, map_location="cpu")
            # Load projector weights
            components.transformer.projectors[0].load_state_dict(ckpt["projector"])
            # Load downsampler weights
            components.transformer.downsampler_cogvideo_output.load_state_dict(ckpt["downsampler"])
            
            if self.args.projector_lr_scale > 0:
                # Keep projector trainable with scaled learning rate
                print(f"Pretrained projector loaded. Finetuning with lr_scale={self.args.projector_lr_scale}")
            else:
                # Freeze projector and downsampler
                for param in components.transformer.projectors.parameters():
                    param.requires_grad = False
                for param in components.transformer.downsampler_cogvideo_output.parameters():
                    param.requires_grad = False
                print("Pretrained projector loaded and frozen.")

        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")

        components.scheduler = CogVideoXDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
        # components.scheduler = CogVideoXDDIMScheduler.from_pretrained(model_path, subfolder="scheduler")

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
        # This is used in the dataloader
        # shape of input video: [B, C, F, H, W]
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        latent = latent_dist.sample() * vae.config.scaling_factor
        return latent

    # def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
    #     latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    #     latents = 1 / self.vae_scaling_factor_image * latents

    #     frames = self.vae.decode(latents).sample
    #     return frames

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
        raw_frames = batch["raw_frames"]    # [B, C, F, H, W] whose value range from -1 to 1, e.g. torch.Size([Batch_size, 3, 49, 480, 720])
        
        # pre-process for vision encoder
        B, C, F, H, W = raw_frames.shape 
        raw_frames = raw_frames.transpose(1, 2).flatten(0, 1)   # B * F, C, H, W
        if self.args.align_models[0] in ['VideoMAEv2', 'VideoMAE', 'OminiMAE', "VJEPA", "VJEPA2"]:
            raw_frames = (raw_frames + 1.0) / 2.0
            raw_frames = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(raw_frames)    # should be NCHW
        elif self.args.align_models[0] == 'DINOv2':
            raw_frames = (raw_frames + 1.0) / 2.0
            raw_frames = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(raw_frames)
        else:
            raise NotImplementedError
        raw_frames = raw_frames.reshape(B, F, C, H, W).transpose(1, 2)
        
        
        # pre-process frames for Video Foundation Models
        assert len(self.args.align_models) == 1, "Support only align one model currently"
        if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
            repa_raw_frames = raw_frames[:, :, 1:]  # remove the first frames
            B, C, F, H, W = repa_raw_frames.shape 
            
            repa_raw_frames = repa_raw_frames.transpose(1, 2).flatten(0, 1)
            # 480x720 -> 160x240
            repa_raw_frames = torch.nn.functional.interpolate(repa_raw_frames, (H // 3, W // 3), mode='bicubic')    # hard coded
            repa_raw_frames = repa_raw_frames.reshape(B, F, C, H // 3, W // 3).transpose(1, 2)  # B, C, F, H, W
        elif self.args.align_models[0] == 'DINOv2':
            repa_raw_frames = raw_frames
            B, C, F, H, W = repa_raw_frames.shape 
            repa_raw_frames = repa_raw_frames.transpose(1, 2).flatten(0, 1) # B * F, C, H, W
            input_resolution = (420, 630)   # to fit the patch size 14 in DINOv2
            repa_raw_frames = torch.nn.functional.interpolate(repa_raw_frames, input_resolution, mode='bicubic')
            repa_raw_frames = repa_raw_frames.reshape(B, F, C, input_resolution[0], input_resolution[1]).transpose(1, 2)  # B, C, F, H, W
        
        
        # encode the frames with vision encoders
        with torch.no_grad():
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                B, C, F, H, W = repa_raw_frames.shape
                if self.args.teacher_pca_path is not None and self.args.align_models[0] == 'VideoMAEv2':
                    # Multi-layer teacher features + PCA projection to 1920d
                    layer_feats = self.vision_encoder(repa_raw_frames, return_multilayer=True, layer_indices=self.teacher_pca_layer_indices)
                    # Each: (B, N_tokens, 768), concatenate -> (B, N_tokens, 2304)
                    concat_feat = torch.cat(layer_feats, dim=-1)
                    # PCA projection: (B, N, 2304) -> (B, N, 1920)
                    align_target = (concat_feat.float() - self.teacher_pca_mean) @ self.teacher_pca_W
                    # Reshape to spatial: B, N, 1920 -> B, 1920, 24, 10, 15
                    align_target = align_target.transpose(1, 2).reshape(B, -1, F // self.vision_encoder.tubelet_size, H // self.vision_encoder.patch_size, W // self.vision_encoder.patch_size)
                else:
                    # Original single-layer encoding: B, 3, 48, 160, 240 -> B, 24x10x15, C
                    align_target = self.vision_encoder(repa_raw_frames)
                    # B, 24x10x15, D -> B, D, 24, 10, 15
                    align_target = align_target.transpose(1, 2).reshape(B, -1, F // self.vision_encoder.tubelet_size, H // self.vision_encoder.patch_size, W // self.vision_encoder.patch_size)
            elif self.args.align_models[0] == 'DINOv2':
                B, C, F, H, W = repa_raw_frames.shape
                repa_raw_frames = repa_raw_frames.transpose(1, 2).flatten(0, 1)
                group_size = 128  # 32 / 64 / 128 to avoid OOM
                chunked = repa_raw_frames.chunk((B * F) // group_size, dim=0)
                
                features = []
                for frames in chunked:
                    group, C, H, W = frames.shape
                    output = self.vision_encoder.forward_features(frames)['x_norm_patchtokens'].reshape(group, input_resolution[0] // self.vision_encoder.patch_size, input_resolution[1] // self.vision_encoder.patch_size, self.vision_encoder.embed_dim)
                    features.append(output)
                features = torch.cat(features, dim=0)
                features = features.reshape(B, F, input_resolution[0] // self.vision_encoder.patch_size, input_resolution[1] // self.vision_encoder.patch_size, self.vision_encoder.embed_dim)
        
        align_targets = []
        if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
            align_target = align_target.flatten(2).transpose(1, 2)  # B, 24x10x15, C
            align_targets.append(align_target)        
        elif self.args.align_models[0] == 'DINOv2':
            first_frame_feature = features[:, :1].permute(0, 4, 1, 2, 3)   # B, 1, H, W, C -> B, C, 1, H, W
            features = features[:, 1:]
            B, F, H, W, C = features.shape
            align_target = features.permute(0, 2, 3, 4, 1).flatten(0, 2)
            # To align with the features from CogVideoX, the encoded features are avg pooled to 1/4
            align_target = torch.nn.functional.avg_pool1d(align_target, kernel_size=4, stride=4)
            align_target = align_target.reshape(B, H, W, C, F // 4).permute(0, 3, 4, 1, 2)
            align_target = torch.cat([first_frame_feature, align_target], dim=2)
            align_target = align_target.flatten(2).transpose(1, 2)  # B, 13x30x45, C
            align_targets.append(align_target)  
            

        patch_size_t = self.state.transformer_config.patch_size_t
        if patch_size_t is not None:
            raise NotImplementedError("This is for CogVideoX1.5 but the 1.5 is not used in VideoREPA")
            ncopy = latent.shape[2] % patch_size_t
            # Copy the first frame ncopy times to match patch_size_t
            first_frame = latent[:, :, :1, :, :]
            latent = torch.cat([first_frame.repeat(1, 1, ncopy, 1, 1), latent], dim=2)
            assert latent.shape[2] % patch_size_t == 0

        batch_size, num_channels, num_frames, height, width = latent.shape

        # Get prompt embeddings
        _, seq_len, _ = prompt_embedding.shape
        prompt_embedding = prompt_embedding.view(batch_size, seq_len, -1).to(dtype=latent.dtype)

        # Sample a random timestep for each sample (standard uniform timestep)
        num_train_timesteps = self.components.scheduler.config.num_train_timesteps
        timesteps = torch.randint(
            0, num_train_timesteps, (batch_size,), device=self.accelerator.device
        ).long()

        # Add noise to latent (standard: same timestep for all frames)
        latent = latent.permute(0, 2, 1, 3, 4)  # from [B, C, F, H, W] to [B, F, C, H, W]
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
        
        # Predict noise
        predicted_noises, aligns = self.components.transformer(
            hidden_states=latent_added_noise,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )
        predicted_noise = predicted_noises[0]
        
        # Aligning features from CogVideoX to pre-trained frozen vision encoders
        # When dual-layer alignment is enabled, aligns contains features from both layers
        # Order depends on layer index: smaller layer index comes first
        if self.args.align_layer_secondary is not None:
            # Dual-layer mode: determine order based on layer indices
            if self.args.align_layer_secondary < self.args.align_layer:
                # Secondary layer is shallower, comes first in aligns
                align_secondary_raw = aligns[0]
                align_primary_raw = aligns[1]
            else:
                align_primary_raw = aligns[0]
                align_secondary_raw = aligns[1]
            
            # Process primary layer features
            align_primary = align_primary_raw.reshape(B, 13, 60 // 2, 90 // 2, -1)
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                align_primary = align_primary[:, 1:]
            
            # Process secondary layer features
            align_secondary = align_secondary_raw.reshape(B, 13, 60 // 2, 90 // 2, -1)
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                align_secondary = align_secondary[:, 1:]
            
            aligns = [align_primary]  # primary for main loss branch
            aligns_secondary = [align_secondary]  # secondary for secondary loss
        else:
            align = aligns[0]
            align = align.reshape(B, 13, 60 // 2, 90 // 2, -1)  # TODO: remove hard coded
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                # remove the first frame
                align = align[:, 1:]    
            aligns = [align]
            aligns_secondary = None
        if self.args.align_models[0] == 'DINOv2':
            # Only able to perform REPA loss when using DINOv2
            assert self.args.loss in ['cosine_similarity', 'cosine_similarity_dual_timestep']
        
        if self.args.loss == 'cosine_similarity' or self.args.loss == 'cosine_similarity_dual_timestep':
            # REPA loss - only align samples with high noise timesteps
            proj_loss = 0
            align = aligns[0].permute(0, 4, 1, 2, 3)    # B, C, F, H, W
            if self.args.align_models[0] != "DINOv2": 
                align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear') 
            
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                B, C, F, H, W = align.shape
                align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                if self.args.teacher_pca_path is not None:
                    # Teacher PCA mode: use avg_pool for spatial downsampling (1920d)
                    align = torch.nn.functional.avg_pool2d(align, kernel_size=3, stride=3)  # 30x45 -> 10x15
                else:
                    align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))   # 30x45 -> 10x15
                align = align.reshape(B, F, C, H // 3, W // 3).permute(0, 2, 1, 3, 4)   # B, C, F, H, W
            
            # Create per-sample mask: only align samples with timestep > threshold (high noise region)
            align_timestep_threshold = int(self.args.align_timestep_threshold * num_train_timesteps)
            high_noise_mask = (timesteps >= align_timestep_threshold)  # [B], boolean mask
            
            align = align.flatten(2).transpose(1, 2).flatten(0, 1)  # BFHW, C
            align_target = align_targets[0].flatten(0, 1)
            align = torch.nn.functional.normalize(align, dim=-1) 
            align_target = torch.nn.functional.normalize(align_target, dim=-1) 
            if self.args.teacher_pca_path is not None:
                inner_dim = self.state.transformer_config.num_attention_heads * self.state.transformer_config.attention_head_dim
                assert align_target.shape[-1] == align.shape[-1] == inner_dim  # 1920
            else:
                assert align_target.shape[-1] == align.shape[-1] == self.args.align_dims[0]  # 768

            # Compute per-token cosine loss
            per_token_loss = (-(align_target * align)).sum(dim=-1)  # (B*tokens_per_sample,)
            
            # Reshape to [B, tokens_per_sample] and apply per-sample mask
            tokens_per_sample = per_token_loss.shape[0] // batch_size
            per_token_loss = per_token_loss.reshape(batch_size, tokens_per_sample)
            
            # Apply mask: only compute loss for high-noise samples
            masked_loss = per_token_loss * high_noise_mask.unsqueeze(-1).float()
            num_valid = high_noise_mask.float().sum() * tokens_per_sample
            if num_valid > 0:
                proj_loss += masked_loss.sum() / num_valid
            else:
                proj_loss += torch.tensor(0.0, device=align.device, dtype=align.dtype)

            if self.args.loss == 'cosine_similarity_dual_timestep':
                # Sample a second set of per-frame timesteps
                timesteps_per_frame_2 = torch.randint(
                    0, num_train_timesteps, (batch_size, num_frames), device=self.accelerator.device
                ).long()
                timesteps_2 = timesteps_per_frame_2.float().mean(dim=1).long()

                # Add noise with the second per-frame timesteps
                noise_2 = torch.randn_like(latent)
                noise_2_flat = noise_2.reshape(batch_size * num_frames, num_channels, height, width)
                timesteps_flat_2 = timesteps_per_frame_2.reshape(batch_size * num_frames)
                latent_flat_for_noise_2 = latent.reshape(batch_size * num_frames, num_channels, height, width)
                latent_added_noise_2_flat = self.components.scheduler.add_noise(latent_flat_for_noise_2, noise_2_flat, timesteps_flat_2)
                latent_added_noise_2 = latent_added_noise_2_flat.reshape(batch_size, num_frames, num_channels, height, width)

                # Forward pass with the second timestep
                predicted_noises_2, aligns_2 = self.components.transformer(
                    hidden_states=latent_added_noise_2,
                    encoder_hidden_states=prompt_embedding,
                    timestep=timesteps_2,
                    image_rotary_emb=rotary_emb,
                    return_dict=False,
                )
                predicted_noise_2 = predicted_noises_2[0]

                # Process alignment features from the second forward pass
                align_2 = aligns_2[0]
                align_2 = align_2.reshape(B, 13, 60 // 2, 90 // 2, -1)  # TODO: remove hard coded
                if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                    align_2 = align_2[:, 1:]  # remove the first frame

                # Compute REPA loss for the second timestep
                align_2 = align_2.permute(0, 4, 1, 2, 3)  # B, C, F, H, W
                if self.args.align_models[0] != "DINOv2":
                    align_2 = torch.nn.functional.interpolate(align_2, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')

                if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                    B2, C2, F2, H2, W2 = align_2.shape
                    align_2 = align_2.permute(0, 2, 1, 3, 4).reshape(B2 * F2, C2, H2, W2)
                    align_2 = self.components.transformer.downsampler_cogvideo_output(align_2.to(torch.bfloat16))
                    align_2 = align_2.reshape(B2, F2, C2, H2 // 3, W2 // 3).permute(0, 2, 1, 3, 4)

                align_2 = align_2.flatten(2).transpose(1, 2).flatten(0, 1)  # BFHW, C
                align_target_2 = align_targets[0].flatten(0, 1)
                align_2 = torch.nn.functional.normalize(align_2, dim=-1)
                align_target_2 = torch.nn.functional.normalize(align_target_2, dim=-1)

                proj_loss_2 = (-(align_target_2 * align_2)).sum(dim=-1).mean(dim=0)
                proj_loss = (proj_loss + proj_loss_2) / 2.0  # Average the two timestep alignment losses

                # Compute diffusion loss for the second timestep as well (per-frame)
                predicted_noise_2_flat = predicted_noise_2.reshape(batch_size * num_frames, num_channels, height, width)
                latent_pred_2_flat = self.components.scheduler.get_velocity(predicted_noise_2_flat, latent_added_noise_2_flat, timesteps_flat_2)
                latent_pred_2 = latent_pred_2_flat.reshape(batch_size, num_frames, num_channels, height, width)
                alphas_cumprod_2 = self.components.scheduler.alphas_cumprod[timesteps_per_frame_2]  # [B, F]
                weights_2 = 1 / (1 - alphas_cumprod_2)
                weights_2 = weights_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, F, 1, 1, 1]
                loss_2 = torch.mean((weights_2 * (latent_pred_2 - latent) ** 2).reshape(batch_size, -1), dim=1)
                loss_2 = loss_2.mean()

        elif self.args.loss == 'cosine_similarity_temporal_diff':
            # REPA loss with temporal difference alignment
            # Key idea: align inter-frame motion direction (temporal difference) instead of absolute features
            # - First frame: align original features
            # - Subsequent frames: align frame differences (frame[t] - frame[t-1])
            proj_loss = 0
            align = aligns[0].permute(0, 4, 1, 2, 3)    # B, C, F, H, W
            if self.args.align_models[0] != "DINOv2": 
                align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear') 
            
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                B, C, F, H, W = align.shape
                align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))   # 30x45 -> 10x15
                align = align.reshape(B, F, C, H // 3, W // 3).permute(0, 2, 1, 3, 4)   # B, C, F, H, W
            
            # Convert to temporal difference representation
            # align shape: B, C, F, H, W
            # First frame stays as-is, subsequent frames become frame[t] - frame[t-1]
            align_first_frame = align[:, :, :1, :, :]  # B, C, 1, H, W
            align_diff = align[:, :, 1:, :, :] - align[:, :, :-1, :, :]  # B, C, F-1, H, W
            align_temporal_diff = torch.cat([align_first_frame, align_diff], dim=2)  # B, C, F, H, W
            
            # Same for align_target: B, D, F, H, W (from align_targets[0] which is B, F*H*W, D)
            align_target = align_targets[0]  # B, F*H*W, D
            # Reshape to spatial-temporal form
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                F_target = F  # same temporal dim after interpolation
                H_target = H // 3
                W_target = W // 3
            else:
                raise NotImplementedError("temporal_diff only supports VideoMAEv2/VJEPA/VJEPA2/VideoMAE/OminiMAE")
            
            align_target_reshaped = align_target.reshape(B, F_target, H_target * W_target, -1)  # B, F, HW, D
            align_target_reshaped = align_target_reshaped.permute(0, 3, 1, 2)  # B, D, F, HW
            # Compute temporal difference for target
            target_first_frame = align_target_reshaped[:, :, :1, :]  # B, D, 1, HW
            target_diff = align_target_reshaped[:, :, 1:, :] - align_target_reshaped[:, :, :-1, :]  # B, D, F-1, HW
            align_target_temporal_diff = torch.cat([target_first_frame, target_diff], dim=2)  # B, D, F, HW
            
            # Flatten for cosine similarity computation
            align_flat = align_temporal_diff.flatten(2).transpose(1, 2).flatten(0, 1)  # B*F*H*W, C
            align_target_flat = align_target_temporal_diff.flatten(2).transpose(1, 2).flatten(0, 1)  # B*F*HW, D
            
            align_flat = torch.nn.functional.normalize(align_flat, dim=-1) 
            align_target_flat = torch.nn.functional.normalize(align_target_flat, dim=-1) 
            assert align_target_flat.shape[-1] == align_flat.shape[-1] == self.args.align_dims[0]

            # Compute cosine similarity loss over all samples (no high-noise filtering)
            proj_loss += (-(align_target_flat * align_flat)).sum(dim=-1).mean()

        elif self.args.loss == 'cosine_similarity_temporal_diff_only':
            # REPA loss with pure temporal difference alignment
            # Key idea: ALL frames use inter-frame differences, including the first frame
            # - All frames: align frame[t] - frame[t-1] (first frame uses frame[0] - 0 = frame[0]... NO)
            # - Actually: only use frames 1..F, computing frame[t] - frame[t-1], discarding frame 0
            proj_loss = 0
            align = aligns[0].permute(0, 4, 1, 2, 3)    # B, C, F, H, W
            if self.args.align_models[0] != "DINOv2": 
                align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear') 
            
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                B, C, F, H, W = align.shape
                align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))   # 30x45 -> 10x15
                align = align.reshape(B, F, C, H // 3, W // 3).permute(0, 2, 1, 3, 4)   # B, C, F, H, W
            
            # Compute pure temporal differences (all frames are differences, no original features)
            # align shape: B, C, F, H, W -> differences: B, C, F-1, H, W
            align_temporal_diff = align[:, :, 1:, :, :] - align[:, :, :-1, :, :]  # B, C, F-1, H, W
            
            # Same for align_target
            align_target = align_targets[0]  # B, F*H*W, D
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                F_target = F
                H_target = H // 3
                W_target = W // 3
            else:
                raise NotImplementedError("temporal_diff_only only supports VideoMAEv2/VJEPA/VJEPA2/VideoMAE/OminiMAE")
            
            align_target_reshaped = align_target.reshape(B, F_target, H_target * W_target, -1)  # B, F, HW, D
            align_target_reshaped = align_target_reshaped.permute(0, 3, 1, 2)  # B, D, F, HW
            # Compute temporal difference for target (all frames are differences)
            align_target_temporal_diff = align_target_reshaped[:, :, 1:, :] - align_target_reshaped[:, :, :-1, :]  # B, D, F-1, HW
            
            # Flatten for cosine similarity computation
            align_flat = align_temporal_diff.flatten(2).transpose(1, 2).flatten(0, 1)  # B*(F-1)*H*W, C
            align_target_flat = align_target_temporal_diff.flatten(2).transpose(1, 2).flatten(0, 1)  # B*(F-1)*HW, D
            
            align_flat = torch.nn.functional.normalize(align_flat, dim=-1) 
            align_target_flat = torch.nn.functional.normalize(align_target_flat, dim=-1) 
            assert align_target_flat.shape[-1] == align_flat.shape[-1] == self.args.align_dims[0]

            # Compute cosine similarity loss over all samples (no high-noise filtering)
            proj_loss += (-(align_target_flat * align_flat)).sum(dim=-1).mean()

        elif self.args.loss == 'token_relation_distillation':
            # TRD loss in VideoREPA - only align samples with high noise timesteps
            assert len(aligns) == 1
            align = aligns[0].permute(0, 4, 1, 2, 3)   # B, F, H, W, C -> B, C, F, H, W  (e.g. B, 768, 12, 30, 45)
            # upsample the temporal dimension (from 12 to 24) to match the dimension in VideoMAEv2
            align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')
    
            # downsample the representation of VDM
            B, C, F, H, W = align.shape
            align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))
            align = align.reshape(B, F, C, H // 3, W // 3)
            
            align = align.permute(0, 1, 3, 4, 2)   # B, F, H, W, C          
            token_relation_distillation_loss = 0
            align = align.flatten(2, 3) # B, F, H*W, C
            align_target = align_targets[0].reshape(B, F, 10 * 15, -1)  # B, 12, 10 * 15, D
            
            # Create per-sample mask: only align samples with timestep > threshold (high noise region)
            align_timestep_threshold = int(self.args.align_timestep_threshold * num_train_timesteps)
            high_noise_mask = (timesteps >= align_timestep_threshold)  # [B], boolean mask
            
            if high_noise_mask.any():
                # normalize before calculate Gram matrix
                align = torch.nn.functional.normalize(align, dim=-1)
                align_target = torch.nn.functional.normalize(align_target, dim=-1)
                assert align.shape[-1] == align_target.shape[-1] == self.args.align_dims[0]

                # BF, HW, C @ BF, C, FHW -> BF, HW, FHW
                align_sim = torch.bmm(align.flatten(0, 1), align.flatten(1, 2).unsqueeze(1).expand(-1, F, -1, -1).flatten(0, 1).transpose(1, 2))
                align_target_sim = torch.bmm(align_target.flatten(0, 1), align_target.flatten(1, 2).unsqueeze(1).expand(-1, F, -1, -1).flatten(0, 1).transpose(1, 2))
                assert align_sim.shape == align_target_sim.shape
                
                # Compute per-sample TRD loss and apply mask
                # align_sim shape: [B*F, HW, F*HW], reshape to [B, F, HW, F*HW]
                trd_per_bf = nn.functional.relu((align_sim - align_target_sim).abs() - self.args.margin).mean(dim=(1, 2))  # [B*F]
                trd_per_sample = trd_per_bf.reshape(B, F).mean(dim=1)  # [B]
                
                # Only compute loss for high-noise samples
                token_relation_distillation_loss = (trd_per_sample * high_noise_mask.float()).sum() / high_noise_mask.float().sum()
            else:
                token_relation_distillation_loss = torch.tensor(0.0, device=align.device, dtype=align.dtype)

        elif self.args.loss == 'token_relation_distillation_only_spatial':
            # pre-process
            align = aligns[0].permute(0, 4, 1, 2, 3)
            align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear') 
            B, C, F, H, W = align.shape
            align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
            align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))
            
            align = align.reshape(B, F, C, H // 3, W // 3)
            align = align.permute(0, 1, 3, 4, 2)

            # calculate loss
            token_relation_distillation_loss = 0
            align = align.flatten(2, 3)
            align_target = align_targets[0].reshape(B, F, 10 * 15, -1)
            align = torch.nn.functional.normalize(align, dim=-1)
            align_target = torch.nn.functional.normalize(align_target, dim=-1)
            
            assert align.shape[-1] == align_target.shape[-1] == self.args.align_dims[0]
            align_sim = torch.bmm(align.flatten(0, 1), align.flatten(0, 1).transpose(1, 2))
            align_target_sim = torch.bmm(align_target.flatten(0, 1), align_target.flatten(0, 1).transpose(1, 2)) 
            assert align_sim.shape == align_target_sim.shape
            token_relation_distillation_loss = nn.functional.relu((align_sim - align_target_sim).abs() - self.args.margin_matrix).mean()
        
        elif self.args.loss == 'token_relation_distillation_only_temporal':
            # pre-process
            align = aligns[0].permute(0, 4, 1, 2, 3)
            align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')  
                
            B, C, F, H, W = align.shape
            align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)   
            align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))   
            
            align = align.reshape(B, F, C, H // 3, W // 3)
            align = align.permute(0, 1, 3, 4, 2)   

            token_relation_distillation_loss = 0
            align_temporal = align.flatten(2, 3) 
            align_target_temporal = align_targets[0].reshape(B, F, 10 * 15, -1)  
            align_temporal = torch.nn.functional.normalize(align_temporal, dim=-1)
            align_target_temporal = torch.nn.functional.normalize(align_target_temporal, dim=-1)
            
            assert align_temporal.shape[-1] == align_target_temporal.shape[-1] == self.args.align_dims[0]  
            align_sim = torch.bmm(align_temporal.flatten(1, 2), align_temporal.flatten(1, 2).transpose(1, 2))  
            align_target_temporal_sim = torch.bmm(align_target_temporal.flatten(1, 2), align_target_temporal.flatten(1, 2).transpose(1, 2)) 

            assert align_sim.shape == align_target_temporal_sim.shape
            token_relation_distillation_loss = nn.functional.relu((align_sim - align_target_temporal_sim).abs() - self.args.margin_matrix)
            
            token_relation_distillation_loss = token_relation_distillation_loss.clone()   # To prevent the following inplace operation which will raise gradient backward error

            token_relation_distillation_loss = token_relation_distillation_loss.reshape(B, 24, 10 * 15, 24, 10 * 15)
            for iddx in range(24):
                token_relation_distillation_loss[:, iddx, :, iddx, :] = torch.tensor(0.0)  
            token_relation_distillation_loss = token_relation_distillation_loss.mean() * (B * 24. * 10 * 15 * 24 * 10 * 15) / (B * 24. * 10 * 15 * (24 - 1) * 10 * 15)          

        elif self.args.loss == 'gram_matrix':
            # Gram Matrix MSE alignment WITH learnable projector:
            # 1. Project student features via MLP projector (1920 -> 768)
            # 2. Spatial downsample via learnable Conv2d (30x45 -> 10x15)
            # 3. Compute cosine Gram matrices for both student (projected) and teacher
            # 4. Loss = MSE(Gram_student, Gram_teacher)
            # Gradients flow through projector + downsampler + transformer backbone.
            proj_loss = 0
            align = aligns[0].permute(0, 4, 1, 2, 3)    # B, C, F, H, W
            if self.args.align_models[0] != "DINOv2":
                align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                B, C, F, H, W = align.shape
                align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))
                align = align.reshape(B, F, C, H // 3, W // 3).permute(0, 2, 1, 3, 4)   # B, C, F, 10, 15

            align_timestep_threshold = int(self.args.align_timestep_threshold * num_train_timesteps)
            high_noise_mask = (timesteps >= align_timestep_threshold)

            # student: (B, N=3600, D=768) after projector; teacher: (B, N=3600, D=768)
            student = align.flatten(2).transpose(1, 2)                  # B, F*10*15, C
            teacher = align_targets[0]                                  # B, 3600, D
            teacher = teacher.to(student.dtype)
            assert student.shape[-1] == teacher.shape[-1] == self.args.align_dims[0]

            # L2-normalize for cosine Gram matrix
            student = torch.nn.functional.normalize(student, dim=-1)
            teacher = torch.nn.functional.normalize(teacher, dim=-1)

            # Cosine Gram matrices (N x N)
            S = torch.bmm(student, student.transpose(1, 2))             # [B, N, N]
            T_gram = torch.bmm(teacher, teacher.transpose(1, 2))        # [B, N, N]

            if high_noise_mask.any():
                per_sample = ((S - T_gram) ** 2).mean(dim=(1, 2))       # [B]
                proj_loss = (per_sample * high_noise_mask.float()).sum() / high_noise_mask.float().sum()
            else:
                proj_loss = torch.tensor(0.0, device=align.device, dtype=align.dtype)

        elif self.args.loss == 'token_similarity_matrix':
            # Dimension-independent alignment: match token-token COSINE similarity matrices
            # computed in each model's NATIVE feature space. No projector; C_s may differ from C_t.
            proj_loss = 0
            # --- replicate REPA preprocessing to obtain align: [B, C_s, F, 10, 15] ---
            align = aligns[0].permute(0, 4, 1, 2, 3)    # B*F, C, H, W
            if self.args.align_models[0] != "DINOv2":
                align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                B, C, F, H, W = align.shape
                align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                if getattr(self.components.transformer, 'skip_projector', False) or self.args.teacher_pca_path is not None:
                    align = torch.nn.functional.avg_pool2d(align, kernel_size=3, stride=3)  # 30x45 -> 10x15
                else:
                    align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))
                align = align.reshape(B, F, C, H // 3, W // 3).permute(0, 2, 1, 3, 4)   # B, C, F, 10, 15

            align_timestep_threshold = int(self.args.align_timestep_threshold * num_train_timesteps)
            high_noise_mask = (timesteps >= align_timestep_threshold)

            # [B, N=3600, C_s] and [B, N=3600, C_t]; C is irrelevant for the similarity matrix
            student = align.flatten(2).transpose(1, 2)                  # B, F*10*15, C_s
            teacher = align_targets[0]                                  # B, 3600, C_t
            teacher = teacher.to(student.dtype)
            student = torch.nn.functional.normalize(student, dim=-1)
            teacher = torch.nn.functional.normalize(teacher, dim=-1)

            # N x N cosine similarity matrices (the C dimension never enters the comparison)
            S = torch.bmm(student, student.transpose(1, 2))             # [B, N, N]
            T = torch.bmm(teacher, teacher.transpose(1, 2))             # [B, N, N]

            if high_noise_mask.any():
                per_sample = ((S - T) ** 2).mean(dim=(1, 2))            # [B]
                proj_loss = (per_sample * high_noise_mask.float()).sum() / high_noise_mask.float().sum()
            else:
                proj_loss = torch.tensor(0.0, device=align.device, dtype=align.dtype)

        elif self.args.loss == 'cka_alignment':
            # Linear CKA: dimension-independent, scale & orthogonal invariant, bounded in [0,1].
            proj_loss = 0
            align = aligns[0].permute(0, 4, 1, 2, 3)    # B*F, C, H, W
            if self.args.align_models[0] != "DINOv2":
                align = torch.nn.functional.interpolate(align, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                B, C, F, H, W = align.shape
                align = align.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W)
                if getattr(self.components.transformer, 'skip_projector', False) or self.args.teacher_pca_path is not None:
                    align = torch.nn.functional.avg_pool2d(align, kernel_size=3, stride=3)
                else:
                    align = self.components.transformer.downsampler_cogvideo_output(align.to(torch.bfloat16))
                align = align.reshape(B, F, C, H // 3, W // 3).permute(0, 2, 1, 3, 4)   # B, C, F, 10, 15

            align_timestep_threshold = int(self.args.align_timestep_threshold * num_train_timesteps)
            high_noise_mask = (timesteps >= align_timestep_threshold)

            student = align.flatten(2).transpose(1, 2)                  # B, N, C_s
            teacher = align_targets[0]                                  # B, N, C_t
            teacher = teacher.to(student.dtype)
            # center tokens (over the token dimension) for CKA
            student = student - student.mean(dim=1, keepdim=True)
            teacher = teacher - teacher.mean(dim=1, keepdim=True)
            Gs = torch.bmm(student, student.transpose(1, 2))            # [B, N, N]
            Gt = torch.bmm(teacher, teacher.transpose(1, 2))            # [B, N, N]
            num = (Gs * Gt).sum(dim=(1, 2))
            den = torch.sqrt((Gs * Gs).sum(dim=(1, 2)) * (Gt * Gt).sum(dim=(1, 2)) + 1e-8)
            cka_per = num / den                                         # [B], in [0,1]
            if high_noise_mask.any():
                proj_loss = ((1.0 - cka_per) * high_noise_mask.float()).sum() / high_noise_mask.float().sum()
            else:
                proj_loss = torch.tensor(0.0, device=align.device, dtype=align.dtype)

        else:
            raise NotImplementedError
        
        # Denoise (standard: uniform timestep for all frames)
        latent_pred = self.components.scheduler.get_velocity(predicted_noise, latent_added_noise, timesteps)

        alphas_cumprod = self.components.scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        loss = torch.mean((weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1)
        loss = loss.mean()

        # For dual timestep mode, average the diffusion loss from both timesteps
        if self.args.loss == 'cosine_similarity_dual_timestep':
            loss = (loss + loss_2) / 2.0

        # Compute secondary layer alignment loss (dual-layer mode)
        if aligns_secondary is not None and self.args.loss_secondary is not None:
            proj_loss_secondary = 0
            align_sec = aligns_secondary[0].permute(0, 4, 1, 2, 3)  # B, C, F, H, W
            if self.args.align_models[0] != "DINOv2":
                align_sec = torch.nn.functional.interpolate(align_sec, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')
            
            if self.args.align_models[0] in ['VideoMAEv2', 'VJEPA', 'VJEPA2', 'VideoMAE', 'OminiMAE']:
                B_sec, C_sec, F_sec, H_sec, W_sec = align_sec.shape
                align_sec = align_sec.permute(0, 2, 1, 3, 4).reshape(B_sec * F_sec, C_sec, H_sec, W_sec)
                align_sec = self.components.transformer.downsampler_cogvideo_output_secondary(align_sec.to(torch.bfloat16))
                align_sec = align_sec.reshape(B_sec, F_sec, C_sec, H_sec // 3, W_sec // 3).permute(0, 2, 1, 3, 4)  # B, C, F, H, W
            
            if self.args.loss_secondary == 'cosine_similarity':
                # Standard REPA loss for secondary layer
                # align_sec shape: B, C, F, H, W (H, W already downsampled to 10, 15)
                align_sec_flat = align_sec.flatten(2).transpose(1, 2).flatten(0, 1)  # B*F*H*W, C
                align_target_sec = align_targets[0].flatten(0, 1)  # B*F*H*W, D
                align_sec_flat = torch.nn.functional.normalize(align_sec_flat, dim=-1)
                align_target_sec = torch.nn.functional.normalize(align_target_sec, dim=-1)
                proj_loss_secondary = (-(align_target_sec * align_sec_flat)).sum(dim=-1).mean()
                
            elif self.args.loss_secondary == 'cosine_similarity_temporal_diff_only':
                # Pure temporal difference alignment for secondary layer
                B_sec, C_sec, F_sec, H_sec, W_sec = align_sec.shape
                align_sec_diff = align_sec[:, :, 1:, :, :] - align_sec[:, :, :-1, :, :]  # B, C, F-1, H, W
                
                # Target temporal differences
                # Note: H_sec and W_sec are already downsampled (10, 15), no need to divide by 3 again
                align_target_sec = align_targets[0]  # B, F*H*W, D
                F_t = F_sec
                H_t = H_sec  # already downsampled
                W_t = W_sec  # already downsampled
                align_target_reshaped_sec = align_target_sec.reshape(B_sec, F_t, H_t * W_t, -1).permute(0, 3, 1, 2)  # B, D, F, HW
                align_target_diff_sec = align_target_reshaped_sec[:, :, 1:, :] - align_target_reshaped_sec[:, :, :-1, :]  # B, D, F-1, HW
                
                align_sec_flat = align_sec_diff.flatten(2).transpose(1, 2).flatten(0, 1)  # B*(F-1)*H*W, C
                align_target_sec_flat = align_target_diff_sec.flatten(2).transpose(1, 2).flatten(0, 1)  # B*(F-1)*HW, D
                
                align_sec_flat = torch.nn.functional.normalize(align_sec_flat, dim=-1)
                align_target_sec_flat = torch.nn.functional.normalize(align_target_sec_flat, dim=-1)
                proj_loss_secondary = (-(align_target_sec_flat * align_sec_flat)).sum(dim=-1).mean()
                
            elif self.args.loss_secondary == 'cosine_similarity_temporal_diff':
                # Temporal diff alignment (first frame original + subsequent frames diff)
                B_sec, C_sec, F_sec, H_sec, W_sec = align_sec.shape
                align_first = align_sec[:, :, :1, :, :]
                align_diff = align_sec[:, :, 1:, :, :] - align_sec[:, :, :-1, :, :]
                align_sec_td = torch.cat([align_first, align_diff], dim=2)  # B, C, F, H, W
                
                # Note: H_sec and W_sec are already downsampled (10, 15), no need to divide by 3 again
                align_target_sec = align_targets[0]
                F_t = F_sec
                H_t = H_sec  # already downsampled
                W_t = W_sec  # already downsampled
                align_target_reshaped_sec = align_target_sec.reshape(B_sec, F_t, H_t * W_t, -1).permute(0, 3, 1, 2)
                target_first = align_target_reshaped_sec[:, :, :1, :]
                target_diff = align_target_reshaped_sec[:, :, 1:, :] - align_target_reshaped_sec[:, :, :-1, :]
                align_target_td_sec = torch.cat([target_first, target_diff], dim=2)
                
                align_sec_flat = align_sec_td.flatten(2).transpose(1, 2).flatten(0, 1)
                align_target_sec_flat = align_target_td_sec.flatten(2).transpose(1, 2).flatten(0, 1)
                align_sec_flat = torch.nn.functional.normalize(align_sec_flat, dim=-1)
                align_target_sec_flat = torch.nn.functional.normalize(align_target_sec_flat, dim=-1)
                proj_loss_secondary = (-(align_target_sec_flat * align_sec_flat)).sum(dim=-1).mean()
            
            # Combine primary and secondary alignment losses
            coeff_secondary = self.args.proj_coeff_secondary if self.args.proj_coeff_secondary is not None else self.args.proj_coeff
            proj_loss = proj_loss + coeff_secondary / self.args.proj_coeff * proj_loss_secondary

        if self.args.loss == 'token_relation_distillation' or self.args.loss == 'token_relation_distillation_only_spatial' or self.args.loss == 'token_relation_distillation_only_temporal':
            return [loss, None, token_relation_distillation_loss]
        return [loss, proj_loss]

    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipelineAlign
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        """
        Return the data that needs to be saved. For videos, the data format is List[PIL],
        and for images, the data format is PIL
            video_generate, list_of_frames_feature_maps = pipe(
                height=height,
                width=width,
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
                feature_maps=True,
            )        
        """
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


register("cogvideox-t2v-align", "lora", CogVideoXT2VAlignLoraTrainer)
