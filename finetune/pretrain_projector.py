"""
Projector Pretraining Script (Decoupled from Main Training)

This script pre-trains the projector MLP that maps student (CogVideoX) intermediate
features to teacher (VideoMAEv2) feature space. The projector is trained independently
so that it provides a stable mapping during the actual alignment training.

Workflow:
1. Load CogVideoX transformer (frozen) and VideoMAEv2 (frozen)
2. For each video sample:
   - Encode with VAE -> get latent
   - Add random noise at random timestep (noisy student state)
   - Forward through transformer with noisy latent -> get intermediate features at align_layer
   - Encode raw frames with VideoMAEv2 -> get teacher features
3. Train projector to minimize cosine distance between projected student features and teacher features
4. Save the trained projector weights

Usage:
    python pretrain_projector.py \
        --model_path <CKPT>/cogvideox-2b \
        --data_csv <REPO>/finetune/openvid/openvid_subset_3000.csv \
        --data_root <REPO>/finetune \
        --output_path <REPO>/finetune/pretrained_projector.pth \
        --align_layer 18 \
        --align_dims 768 \
        --projector_dim 2048 \
        --epochs 10 \
        --lr 1e-3 \
        --batch_size 1
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as TF
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import pandas as pd
import decord
from torchvision.transforms import Normalize, Resize, CenterCrop, Compose
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from diffusers import AutoencoderKLCogVideoX, DDPMScheduler
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from transformers import AutoTokenizer, T5EncoderModel
from finetune.models.cogvideox_t2v_align.models.cogvideox_align import (
    CogVideoXTransformer3DModelAlign,
    build_mlp,
)
from finetune.models.cogvideox_t2v_align.models.ssl.VideoMAEv2 import vit_base_patch16_224
from finetune.paths import ckpt, repo


class ProjectorPretrainDataset(Dataset):
    """Dataset that loads videos and returns raw frames for both student and teacher."""

    def __init__(self, data_csv, data_root, max_num_frames=49, resolution=(480, 720)):
        self.data_root = Path(data_root)
        self.base_path = self.data_root / "openvid" / "videos"
        self.max_num_frames = max_num_frames
        self.resolution = resolution  # (H, W)

        df = pd.read_csv(data_csv)
        self.videos = [self.base_path / row["video"] for _, row in df.iterrows()]
        self.prompts = [row["caption"] for _, row in df.iterrows()]

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        video_path = self.videos[idx]
        prompt = self.prompts[idx]

        # Load video frames
        vr = decord.VideoReader(str(video_path))
        total_frames = len(vr)

        # Sample frames uniformly
        if total_frames >= self.max_num_frames:
            frame_indices = np.linspace(0, total_frames - 1, self.max_num_frames, dtype=int).tolist()
        else:
            frame_indices = list(range(total_frames))
            # Pad by repeating last frame
            while len(frame_indices) < self.max_num_frames:
                frame_indices.append(frame_indices[-1])

        frames = vr.get_batch(frame_indices)  # F, H, W, C
        if hasattr(frames, 'asnumpy'):
            frames = torch.from_numpy(frames.asnumpy()).float()
        else:
            frames = frames.float()
        frames = frames.permute(0, 3, 1, 2)  # F, C, H, W
        frames = frames / 127.5 - 1.0  # Normalize to [-1, 1]

        # Resize and center crop
        H, W = self.resolution
        frames = TF.interpolate(frames, size=(H, W), mode='bilinear', align_corners=False)

        return {
            "frames": frames,  # F, C, H, W
            "prompt": prompt,
            "video_path": str(video_path),
        }


def get_rotary_embeddings(height, width, num_frames, transformer_config, vae_scale_factor_spatial, device):
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


@torch.no_grad()
def extract_student_features(transformer, vae, tokenizer, text_encoder, batch, device, align_layer):
    """Extract intermediate features from CogVideoX with zero noise (clean latent)."""
    frames = batch["frames"].to(device)  # B, F, C, H, W
    prompts = batch["prompt"]

    B, F_total, C, H, W = frames.shape
    frames_for_vae = frames.permute(0, 2, 1, 3, 4)  # B, C, F, H, W

    # Encode video with VAE
    vae_input = frames_for_vae.to(dtype=vae.dtype)
    latent_dist = vae.encode(vae_input).latent_dist
    latent = latent_dist.sample() * vae.config.scaling_factor  # B, C, F', H', W'

    # Encode text
    prompt_token_ids = tokenizer(
        prompts,
        padding="max_length",
        max_length=transformer.config.max_text_seq_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids.to(device)
    prompt_embedding = text_encoder(prompt_token_ids)[0]  # B, seq_len, hidden

    # Prepare for transformer forward (no noise, timestep=0)
    batch_size, num_channels, num_frames, height, width = latent.shape
    latent = latent.permute(0, 2, 1, 3, 4)  # B, F, C, H, W

    # Zero noise: use clean latent directly
    timesteps = torch.zeros(batch_size, device=device).long()

    # Rotary embeddings
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)
    rotary_emb = get_rotary_embeddings(
        height=height * vae_scale_factor_spatial,
        width=width * vae_scale_factor_spatial,
        num_frames=num_frames,
        transformer_config=transformer.config,
        vae_scale_factor_spatial=vae_scale_factor_spatial,
        device=device,
    ) if transformer.config.use_rotary_positional_embeddings else None

    # Forward pass - get intermediate features
    _, aligns = transformer(
        hidden_states=latent,
        encoder_hidden_states=prompt_embedding.to(dtype=latent.dtype),
        timestep=timesteps,
        image_rotary_emb=rotary_emb,
        return_dict=False,
    )

    # aligns[0] shape: B*13*30*45, align_dim (after projector)
    # But we need the RAW features before projector
    # We need to hook into the transformer to get pre-projector features
    return aligns


@torch.no_grad()
def extract_teacher_features(vision_encoder, frames, device):
    """Extract VideoMAEv2 features from raw frames."""
    # frames: B, C, F, H, W (value range [-1, 1])
    B, C, F, H, W = frames.shape

    # Pre-process for VideoMAEv2
    raw_frames = frames.transpose(1, 2).flatten(0, 1)  # B*F, C, H, W
    raw_frames = (raw_frames + 1.0) / 2.0
    raw_frames = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(raw_frames)
    raw_frames = raw_frames.reshape(B, F, C, H, W).transpose(1, 2)  # B, C, F, H, W

    # Remove first frame (VideoMAEv2 processes frames 1..48)
    repa_raw_frames = raw_frames[:, :, 1:]
    B, C, F_new, H, W = repa_raw_frames.shape

    # Resize to 160x240 (as in original code)
    repa_raw_frames = repa_raw_frames.transpose(1, 2).flatten(0, 1)  # B*F, C, H, W
    repa_raw_frames = TF.interpolate(repa_raw_frames, (H // 3, W // 3), mode='bicubic')
    repa_raw_frames = repa_raw_frames.reshape(B, F_new, C, H // 3, W // 3).transpose(1, 2)  # B, C, F, H/3, W/3

    # Forward through VideoMAEv2
    align_target = vision_encoder(repa_raw_frames.to(dtype=next(vision_encoder.parameters()).dtype))  # B, 24*10*15, D
    align_target = align_target.transpose(1, 2).reshape(
        B, -1, F_new // vision_encoder.tubelet_size, (H // 3) // vision_encoder.patch_size, (W // 3) // vision_encoder.patch_size
    )
    # B, D, 24, 10, 15 -> B, 24*10*15, D
    align_target = align_target.flatten(2).transpose(1, 2)

    return align_target


class ProjectorPretrainer:
    def __init__(self, args):
        self.args = args
        # DDP setup
        self.distributed = dist.is_initialized()
        if self.distributed:
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.local_rank = 0
            self.rank = 0
            self.world_size = 1
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_models()
        self.setup_data()
        self.setup_projector()

    def setup_models(self):
        """Load all models (frozen)."""
        model_path = str(self.args.model_path)
        print(f"Loading models from {model_path}...")

        # Tokenizer and text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder", torch_dtype=torch.bfloat16
        ).to(self.device).eval()

        # VAE
        self.vae = AutoencoderKLCogVideoX.from_pretrained(
            model_path, subfolder="vae", torch_dtype=torch.bfloat16
        ).to(self.device).eval()

        # Noise scheduler for adding noise to latents
        self.noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        # Transformer (we need intermediate features WITHOUT projector)
        # Load in float32 to avoid dtype mismatch during forward
        self.transformer = CogVideoXTransformer3DModelAlign.from_pretrained(
            model_path, subfolder="transformer",
            align_layer=self.args.align_layer,
            align_dims=self.args.align_dims,
            projector_dim=self.args.projector_dim,
        ).to(self.device).eval()

        # VideoMAEv2
        self.vision_encoder = vit_base_patch16_224().to(self.device)
        self.vision_encoder.from_pretrained(ckpt("VideoMAEv2", "vit_b_k710_dl_from_giant.pth"))
        self.vision_encoder.eval()

        # Freeze all
        for model in [self.text_encoder, self.vae, self.transformer, self.vision_encoder]:
            for param in model.parameters():
                param.requires_grad = False

        print("All models loaded and frozen.")

    def setup_data(self):
        """Setup dataset and dataloader."""
        self.dataset = ProjectorPretrainDataset(
            data_csv=self.args.data_csv,
            data_root=self.args.data_root,
            max_num_frames=49,
            resolution=(480, 720),
        )
        if self.distributed:
            self.sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        else:
            self.sampler = None
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.args.batch_size,
            shuffle=(self.sampler is None),
            sampler=self.sampler,
            num_workers=0,  # Avoid CUDA fork issues
            pin_memory=True,
        )
        if self.rank == 0:
            print(f"Dataset: {len(self.dataset)} samples, {self.world_size} GPUs")

    def setup_projector(self):
        """Initialize the projector MLP (trainable)."""
        inner_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.projector = build_mlp(inner_dim, self.args.projector_dim, self.args.align_dims[0]).to(self.device)
        # Also need the spatial downsampler
        self.downsampler = nn.Conv2d(
            in_channels=self.args.align_dims[0],
            out_channels=self.args.align_dims[0],
            kernel_size=(3, 3),
            stride=(3, 3),
        ).to(self.device)

        # Wrap with DDP if distributed
        if self.distributed:
            self.projector = DDP(self.projector, device_ids=[self.local_rank])
            self.downsampler = DDP(self.downsampler, device_ids=[self.local_rank])

        self.optimizer = torch.optim.AdamW(
            list(self.projector.parameters()) + list(self.downsampler.parameters()),
            lr=self.args.lr,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.args.epochs * len(self.dataloader)
        )

        total_params = sum(p.numel() for p in self.projector.parameters()) + sum(p.numel() for p in self.downsampler.parameters())
        if self.rank == 0:
            print(f"Projector + Downsampler parameters: {total_params / 1e6:.2f}M")

    @torch.no_grad()
    def extract_raw_student_features(self, frames, prompts):
        """
        Extract raw (pre-projector) intermediate features from CogVideoX transformer.
        Uses a hook to capture features at align_layer before the projector is applied.
        """
        B, F_total, C, H, W = frames.shape
        frames_for_vae = frames.permute(0, 2, 1, 3, 4).to(device=self.device, dtype=self.vae.dtype)

        # VAE encode
        latent_dist = self.vae.encode(frames_for_vae).latent_dist
        latent = latent_dist.sample() * self.vae.config.scaling_factor

        # Text encode
        prompt_token_ids = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.transformer.config.max_text_seq_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        prompt_embedding = self.text_encoder(prompt_token_ids)[0]

        batch_size, num_channels, num_frames, height, width = latent.shape
        latent = latent.permute(0, 2, 1, 3, 4).float()  # B, F, C, H, W, ensure float32

        # Random timestep and add noise to latent (noisy student state)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        noise = torch.randn_like(latent)
        # Reshape latent for scheduler: B, F, C, H, W -> B, C, F, H, W
        latent_for_noise = latent.permute(0, 2, 1, 3, 4)
        noisy_latent = self.noise_scheduler.add_noise(latent_for_noise, noise.permute(0, 2, 1, 3, 4), timesteps)
        latent = noisy_latent.permute(0, 2, 1, 3, 4)  # Back to B, F, C, H, W

        # Rotary embeddings
        vae_scale_factor_spatial = 2 ** (len(self.vae.config.block_out_channels) - 1)
        rotary_emb = get_rotary_embeddings(
            height=height * vae_scale_factor_spatial,
            width=width * vae_scale_factor_spatial,
            num_frames=num_frames,
            transformer_config=self.transformer.config,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            device=self.device,
        ) if self.transformer.config.use_rotary_positional_embeddings else None

        # Hook to capture raw features at align_layer (before projector)
        raw_features = {}

        def hook_fn(module, input, output):
            # CogVideoX block output is (hidden_states, encoder_hidden_states)
            if isinstance(output, tuple):
                raw_features["align"] = output[0]
            else:
                raw_features["align"] = output

        # Register hook on the align_layer-th block
        align_layer_idx = self.args.align_layer - 1  # 0-indexed
        hook = self.transformer.transformer_blocks[align_layer_idx].register_forward_hook(hook_fn)

        # Forward pass (all inputs must be float32 to match transformer)
        _ = self.transformer(
            hidden_states=latent.float(),
            encoder_hidden_states=prompt_embedding.float(),
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )

        hook.remove()

        # raw_features["align"] is the output of the transformer block
        # Shape: (B, text_seq_len + video_seq_len, inner_dim)
        hidden_states = raw_features["align"]

        # CogVideoX block output is (hidden_states, encoder_hidden_states)
        # hidden_states contains video tokens, encoder_hidden_states contains text tokens
        # So hook output[0] = video hidden_states with shape [B, 13*30*45, 1920]
        return hidden_states.float()  # B, 13*30*45, inner_dim (1920)

    def process_student_features(self, raw_student_features):
        """
        Apply projector and spatial downsampling to raw student features.
        This mimics what the original code does but with OUR projector.
        raw_student_features: B, 13*30*45, 1920 (float32)
        """
        B = raw_student_features.shape[0]
        # CogVideoX-2B with 49x480x720: latent=13 frames, patch=30x45, inner_dim=1920
        # 13 * 30 * 45 = 17550
        feat = raw_student_features.float().reshape(B, 13, 30, 45, -1)

        # Remove first frame to match VideoMAEv2
        feat = feat[:, 1:]  # B, 12, 30, 45, 1920

        # Apply projector: 1920 -> 768 (projector is float32)
        proj = self.projector.module if self.distributed else self.projector
        feat = proj(feat)  # B, 12, 30, 45, 768

        # Spatial-temporal processing to match teacher
        # B, 12, 30, 45, 768 -> B, 768, 12, 30, 45
        feat = feat.permute(0, 4, 1, 2, 3).contiguous()

        # Temporal upsample: 12 -> 24 to match VideoMAEv2
        feat = TF.interpolate(feat, scale_factor=(2.0, 1.0, 1.0), mode='trilinear')

        # Spatial downsample: 30x45 -> 10x15
        B, C, F, H, W = feat.shape
        feat = feat.permute(0, 2, 1, 3, 4).reshape(B * F, C, H, W).contiguous()
        ds = self.downsampler.module if self.distributed else self.downsampler
        feat = ds(feat)  # B*F, 768, 10, 15
        feat = feat.reshape(B, F, C, H // 3, W // 3).permute(0, 2, 1, 3, 4)  # B, 768, 24, 10, 15

        # Flatten: B, 24*10*15, 768
        feat = feat.flatten(2).transpose(1, 2)

        return feat

    def train(self):
        """Main training loop."""
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"Starting Projector Pretraining")
            print(f"  Epochs: {self.args.epochs}")
            print(f"  Samples: {len(self.dataset)}")
            print(f"  Batch size: {self.args.batch_size}")
            print(f"  Learning rate: {self.args.lr}")
            print(f"  Align layer: {self.args.align_layer}")
            print(f"  Student dim: {self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim}")
            print(f"  Teacher dim: {self.args.align_dims[0]}")
            print(f"  Projector dim: {self.args.projector_dim}")
            print(f"  World size: {self.world_size}")
            print(f"{'='*60}\n")

        best_loss = float('inf')

        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            num_batches = 0
            if self.sampler is not None:
                self.sampler.set_epoch(epoch)

            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.args.epochs}", disable=(self.rank != 0))
            for batch in pbar:
                frames = batch["frames"].to(self.device)  # B, F, C, H, W
                prompts = list(batch["prompt"])

                try:
                    # Extract raw student features (no projector, no noise)
                    raw_student = self.extract_raw_student_features(frames, prompts)
                    raw_student = raw_student.float()  # Ensure float32

                    # Extract teacher features
                    teacher_features = extract_teacher_features(
                        self.vision_encoder,
                        frames.permute(0, 2, 1, 3, 4),  # B, C, F, H, W
                        self.device,
                    )  # B, 24*10*15, 768
                    teacher_features = teacher_features.float()  # Ensure float32

                    # Apply projector + downsampler (trainable, float32)
                    student_projected = self.process_student_features(raw_student)  # B, 24*10*15, 768

                    # Compute cosine similarity loss
                    student_norm = TF.normalize(student_projected, dim=-1)
                    teacher_norm = TF.normalize(teacher_features, dim=-1)

                    loss = (-(student_norm * teacher_norm)).sum(dim=-1).mean()

                    # Backward
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.projector.parameters()) + list(self.downsampler.parameters()),
                        max_norm=1.0,
                    )
                    self.optimizer.step()
                    self.scheduler.step()

                    epoch_loss += loss.item()
                    num_batches += 1
                    pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"})

                except Exception as e:
                    import traceback
                    print(f"  Skipping batch due to error: {e}")
                    traceback.print_exc()
                    if num_batches == 0:
                        # Print full traceback for first error to debug
                        raise
                    continue

            avg_loss = epoch_loss / max(num_batches, 1)
            if self.rank == 0:
                print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

                # Save best
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self.save(self.args.output_path)
                    print(f"  Saved best projector (loss={best_loss:.4f})")

            if self.distributed:
                dist.barrier()

        # Final save
        if self.rank == 0:
            self.save(self.args.output_path)
            print(f"\nTraining complete! Projector saved to {self.args.output_path}")

    def save(self, path):
        """Save projector and downsampler weights."""
        proj = self.projector.module if self.distributed else self.projector
        ds = self.downsampler.module if self.distributed else self.downsampler
        torch.save({
            "projector": proj.state_dict(),
            "downsampler": ds.state_dict(),
            "args": vars(self.args),
        }, path)


def main():
    parser = argparse.ArgumentParser(description="Pretrain projector for VideoREPA")
    parser.add_argument("--model_path", type=str, default=ckpt("cogvideox-2b"))
    parser.add_argument("--data_csv", type=str, default=repo("finetune", "openvid", "openvid_1w6.csv"))
    parser.add_argument("--data_root", type=str, default=repo("finetune"))
    parser.add_argument("--output_path", type=str, default=repo("finetune", "pretrained_projector.pth"))
    parser.add_argument("--align_layer", type=int, default=18)
    parser.add_argument("--align_dims", type=int, nargs='+', default=[768])
    parser.add_argument("--projector_dim", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # Initialize DDP if launched with torchrun
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    trainer = ProjectorPretrainer(args)
    trainer.train()

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
