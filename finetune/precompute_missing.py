"""Simple single-GPU script to precompute missing video latents and prompt embeddings."""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import pandas as pd
import decord
import hashlib
import numpy as np
from safetensors.torch import save_file, load_file
from diffusers import AutoencoderKLCogVideoX
from transformers import AutoTokenizer, T5EncoderModel
from torchvision.transforms import Compose, Lambda, Resize, CenterCrop

MODEL_PATH = "/efs/zixianhuang/ckpt/cogvideox-2b"
DATA_ROOT = Path("/efs/zixianhuang/VideoREPA/finetune")
CACHE_DIR = DATA_ROOT / "cache" / "openvid"
VIDEO_DIR = DATA_ROOT / "openvid" / "videos"
RESOLUTION = (480, 720)  # H, W
MAX_FRAMES = 49
DEVICE = "cuda:0"


def load_models():
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        MODEL_PATH, subfolder="text_encoder", torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    vae = AutoencoderKLCogVideoX.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(DEVICE).eval()
    print("Models loaded.")
    return tokenizer, text_encoder, vae


def preprocess_video(video_path):
    """Load and preprocess video frames."""
    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)

    if total_frames >= MAX_FRAMES:
        indices = np.linspace(0, total_frames - 1, MAX_FRAMES, dtype=int).tolist()
    else:
        indices = list(range(total_frames))
        while len(indices) < MAX_FRAMES:
            indices.append(indices[-1])

    frames = vr.get_batch(indices)
    if hasattr(frames, 'asnumpy'):
        frames = torch.from_numpy(frames.asnumpy()).float()
    else:
        frames = frames.float()

    # F, H, W, C -> F, C, H, W
    frames = frames.permute(0, 3, 1, 2)
    frames = frames / 127.5 - 1.0

    # Resize to target resolution
    H, W = RESOLUTION
    frames = torch.nn.functional.interpolate(frames, size=(H, W), mode='bilinear', align_corners=False)

    return frames, indices


@torch.no_grad()
def encode_video(vae, frames):
    """Encode video frames with VAE."""
    # frames: F, C, H, W -> 1, C, F, H, W
    video = frames.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device=DEVICE, dtype=vae.dtype)
    latent_dist = vae.encode(video).latent_dist
    latent = latent_dist.sample() * vae.config.scaling_factor
    return latent[0]  # C, F', H', W'


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt):
    """Encode text prompt."""
    inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=226,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    prompt_embedding = text_encoder(inputs.input_ids.to(DEVICE))[0]
    return prompt_embedding[0].cpu()  # seq_len, hidden


def main():
    # Find missing
    missing_file = "/tmp/still_missing.txt"
    with open(missing_file) as f:
        missing_stems = [line.strip() for line in f if line.strip()]
    print(f"Missing videos: {len(missing_stems)}")

    # Load CSV for prompts
    csv_path = DATA_ROOT / "openvid" / "openvid_3w2.csv"
    df = pd.read_csv(csv_path)
    video_to_caption = {row["video"].replace(".mp4", ""): row["caption"] for _, row in df.iterrows()}

    # Setup dirs
    video_latent_dir = CACHE_DIR / "video_latent" / "cogvideox-t2v" / "49x480x720"
    prompt_dir = CACHE_DIR / "prompt_embeddings"
    frame_idx_dir = CACHE_DIR / "frame_idx" / "cogvideox-t2v" / "49x480x720"
    video_latent_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    frame_idx_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, text_encoder, vae = load_models()

    done = 0
    errors = 0
    for i, stem in enumerate(missing_stems):
        video_path = VIDEO_DIR / f"{stem}.mp4"
        latent_path = video_latent_dir / f"{stem}.safetensors"
        frame_idx_path = frame_idx_dir / f"{stem}.safetensors"

        if latent_path.exists():
            done += 1
            continue

        caption = video_to_caption.get(stem, "")
        prompt_hash = hashlib.sha256(caption.encode()).hexdigest()
        prompt_path = prompt_dir / f"{prompt_hash}.safetensors"

        try:
            # Encode prompt if not cached
            if not prompt_path.exists():
                prompt_emb = encode_prompt(tokenizer, text_encoder, caption)
                save_file({"prompt_embedding": prompt_emb}, str(prompt_path))

            # Encode video
            frames, indices = preprocess_video(video_path)
            latent = encode_video(vae, frames)
            save_file({"encoded_video": latent.cpu()}, str(latent_path))
            save_file({"frame_idx_list": torch.tensor(indices)}, str(frame_idx_path))

            done += 1
            if (done) % 10 == 0:
                print(f"  [{done}/{len(missing_stems)}] done")

        except Exception as e:
            errors += 1
            print(f"  ERROR [{stem}]: {e}")
            continue

    print(f"\nFinished! Done: {done}, Errors: {errors}")


if __name__ == "__main__":
    main()
