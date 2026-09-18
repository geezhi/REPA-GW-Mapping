"""
Fit PCA on multi-layer VideoMAEv2 teacher features to produce a fixed 
projection matrix that maps concatenated multi-layer features (768*3=2304) 
to the student's inner dimension (1920).

This creates a fixed projection matrix W_pca (2304 x 1920) that:
- Preserves teacher's multi-layer knowledge in 1920 dimensions
- Each dimension has semantic meaning from teacher's understanding
- Is completely fixed during training (no learnable parameters)

Usage:
    python fit_teacher_pca.py \
        --data_csv <REPO>/finetune/openvid/openvid_subset_3000.csv \
        --data_root <REPO>/finetune \
        --output_path <REPO>/finetune/teacher_pca_matrix.pth \
        --num_samples 3000 \
        --layer_indices 9 10 11
"""

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import decord
from tqdm import tqdm
from torchvision.transforms import Normalize
import torch.nn.functional as F

from finetune.models.cogvideox_t2v_align.models.ssl.VideoMAEv2 import vit_base_patch16_224
from finetune.paths import ckpt, repo


DEVICE = "cuda:0"


def load_video(video_path, max_frames=49, resolution=(480, 720)):
    """Load and preprocess video."""
    vr = decord.VideoReader(str(video_path))
    total_frames = len(vr)

    if total_frames >= max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()
    else:
        indices = list(range(total_frames))
        while len(indices) < max_frames:
            indices.append(indices[-1])

    frames = vr.get_batch(indices)
    if hasattr(frames, 'asnumpy'):
        frames = torch.from_numpy(frames.asnumpy()).float()
    else:
        frames = frames.float()

    frames = frames.permute(0, 3, 1, 2)  # F, C, H, W
    frames = frames / 127.5 - 1.0
    H, W = resolution
    frames = F.interpolate(frames, size=(H, W), mode='bilinear', align_corners=False)

    return frames  # F, C, H, W


@torch.no_grad()
def extract_multilayer_teacher_features(vision_encoder, frames, layer_indices):
    """
    Extract multi-layer features from VideoMAEv2.
    
    Args:
        frames: (F, C, H, W) in range [-1, 1]
        layer_indices: list of layer indices to extract
    
    Returns:
        concatenated features: (N_tokens, D*num_layers)
    """
    # Preprocess for VideoMAEv2
    B = 1
    F_total = frames.shape[0]
    C, H, W = frames.shape[1], frames.shape[2], frames.shape[3]
    
    raw_frames = frames.clone()
    raw_frames = (raw_frames + 1.0) / 2.0
    raw_frames = Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(raw_frames)
    
    # Remove first frame, reshape for VideoMAEv2
    raw_frames = raw_frames[1:]  # F-1, C, H, W
    F_new = raw_frames.shape[0]
    
    # Resize to 160x240
    raw_frames = F.interpolate(raw_frames, (H // 3, W // 3), mode='bicubic')
    
    # Reshape: (1, C, F, H/3, W/3)
    raw_frames = raw_frames.unsqueeze(0).permute(0, 2, 1, 3, 4).to(DEVICE)  # 1, C, F, H/3, W/3
    
    # Get multi-layer features
    layer_feats = vision_encoder(raw_frames, return_multilayer=True, layer_indices=layer_indices)
    # Each is (1, N_tokens, 768)
    
    # Concatenate along feature dimension
    concat_feat = torch.cat(layer_feats, dim=-1)  # (1, N_tokens, 768*num_layers)
    
    return concat_feat[0]  # (N_tokens, 768*num_layers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default=repo("finetune", "openvid", "openvid_subset_3000.csv"))
    parser.add_argument("--data_root", type=str, default=repo("finetune"))
    parser.add_argument("--output_path", type=str, default=repo("finetune", "teacher_pca_matrix.pth"))
    parser.add_argument("--num_samples", type=int, default=3000)
    parser.add_argument("--layer_indices", type=int, nargs='+', default=[9, 10, 11])
    parser.add_argument("--target_dim", type=int, default=1920)
    parser.add_argument("--max_tokens_for_pca", type=int, default=500000, help="Max tokens to use for PCA fitting")
    args = parser.parse_args()

    print(f"Layer indices: {args.layer_indices}")
    print(f"Source dim: 768 * {len(args.layer_indices)} = {768 * len(args.layer_indices)}")
    print(f"Target dim: {args.target_dim}")

    # Load VideoMAEv2
    print("Loading VideoMAEv2...")
    vision_encoder = vit_base_patch16_224().to(DEVICE)
    vision_encoder.from_pretrained(ckpt("VideoMAEv2", "vit_b_k710_dl_from_giant.pth"))
    vision_encoder.eval()
    for param in vision_encoder.parameters():
        param.requires_grad = False
    print(f"VideoMAEv2 loaded. Depth: {len(vision_encoder.blocks)} blocks")

    # Load data
    data_root = Path(args.data_root)
    video_dir = data_root / "openvid" / "videos"
    df = pd.read_csv(args.data_csv)
    videos = [video_dir / row["video"] for _, row in df.iterrows()]
    
    if args.num_samples < len(videos):
        videos = videos[:args.num_samples]
    print(f"Processing {len(videos)} videos...")

    # Collect teacher features
    all_features = []
    total_tokens = 0

    for i, video_path in enumerate(tqdm(videos, desc="Extracting features")):
        try:
            frames = load_video(video_path)
            feat = extract_multilayer_teacher_features(vision_encoder, frames, args.layer_indices)
            # feat: (N_tokens, concat_dim)
            all_features.append(feat.cpu())
            total_tokens += feat.shape[0]

            # Limit total tokens for memory
            if total_tokens >= args.max_tokens_for_pca:
                print(f"Reached {total_tokens} tokens, stopping collection.")
                break
        except Exception as e:
            print(f"  Error on {video_path.name}: {e}")
            continue

    # Concatenate all features
    print(f"Collected {total_tokens} tokens from {len(all_features)} videos")
    all_features = torch.cat(all_features, dim=0)  # (total_tokens, concat_dim)
    print(f"Feature matrix shape: {all_features.shape}")

    concat_dim = all_features.shape[1]
    assert concat_dim == 768 * len(args.layer_indices)

    # Fit PCA: compute top-k eigenvectors of covariance matrix
    print(f"Fitting PCA: {concat_dim} -> {args.target_dim}...")
    
    # Center the data
    mean = all_features.mean(dim=0)  # (concat_dim,)
    centered = all_features - mean

    # SVD on centered data (more numerically stable than eigendecomposition)
    # For large matrices, use randomized SVD
    print("Computing SVD (this may take a minute)...")
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)
    # Vt: (min(N, D), D) — rows are principal components
    
    # Take top target_dim components
    W_pca = Vt[:args.target_dim, :].T  # (concat_dim, target_dim)
    explained_variance = (S[:args.target_dim] ** 2).sum() / (S ** 2).sum()
    print(f"PCA explained variance ratio: {explained_variance:.4f}")

    # Save
    torch.save({
        "W_pca": W_pca,  # (2304, 1920)
        "mean": mean,     # (2304,) for centering
        "layer_indices": args.layer_indices,
        "source_dim": concat_dim,
        "target_dim": args.target_dim,
        "explained_variance": explained_variance.item(),
        "num_samples": len(all_features),
    }, args.output_path)

    print(f"\nSaved PCA matrix to {args.output_path}")
    print(f"  W_pca shape: {W_pca.shape}")
    print(f"  Explained variance: {explained_variance:.4f}")
    print(f"  Usage: teacher_1920 = (teacher_2304 - mean) @ W_pca")


if __name__ == "__main__":
    main()
