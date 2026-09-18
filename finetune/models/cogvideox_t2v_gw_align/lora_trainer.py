"""GW-aligned LoRA / SFT trainer for CogVideoX.

Replaces the learnable projector of REPA with a Gromov-Wasserstein transport plan
between *feature dimensions*, so the alignment is parameter-free and all of the
gradient signal reaches the diffusion backbone.

The heavy lifting lives in :mod:`finetune.gw_relational`; this file only wires the
loss into the training loop.

Registered as ``cogvideox-t2v-gw-align`` for both ``lora`` and ``sft``.
"""

import os
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
)
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from PIL import Image
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision.transforms import Normalize
from transformers import AutoTokenizer, T5EncoderModel
from typing_extensions import override

from finetune.gw_relational import (
    ZeroParamFeatureAdapter,
    build_alignment_loss,
)
from finetune.models.cogvideox_t2v_gw_align.models.cogvideox_align import (
    CogVideoXPipelineAlign,
    CogVideoXTransformer3DModelAlign,
)
from finetune.models.cogvideox_t2v_gw_align.models.ssl.VideoMAE import (
    vit_base_patch16_224 as VideoMAE_vit_base_patch16_224,
)
from finetune.models.cogvideox_t2v_gw_align.models.ssl.VideoMAEv2 import vit_base_patch16_224
from finetune.schemas import Components
from finetune.trainer import Trainer
from finetune.utils import unwrap_model

from ..utils import register

# Encoders whose pre-processing is ImageNet normalisation on ``[0, 1]`` pixels.
IMAGENET_NORMALIZED_ENCODERS = ("VideoMAEv2", "VideoMAE", "OminiMAE", "VJEPA", "VJEPA2")


class CogVideoXT2VGWAlignLoraTrainer(Trainer):
    """Trainer using Gromov-Wasserstein alignment instead of projector-based REPA.

    Differences from the projector-based trainer:

    - No MLP projector and no learnable downsampler: the student features are
      resampled onto the teacher grid with parameter-free operations only.
    - A GW transport plan ``T in R^{D1 x D2}`` is computed **per sample** under
      ``no_grad`` and used to project the teacher into the student's space.
    - The alignment loss (``gw_relational`` or ``gw_gram``) is then a plain cosine
      / Gram objective in the student's own space.
    """

    UNLOAD_LIST = ["text_encoder", "vae"]

    #: Root that holds the frozen video-encoder weights. Override with ``VFM_CKPT_DIR``.
    VFM_CKPT_DIR = os.environ.get("VFM_CKPT_DIR", "/efs/zixianhuang/ckpt")

    def __init__(self, args: Any) -> None:
        super().__init__(args)
        self._adapters: Dict[Tuple[int, int, int], ZeroParamFeatureAdapter] = {}

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def initialize_vision_encoder(self) -> None:
        if len(self.args.align_models) != 1:
            raise NotImplementedError("Currently only one alignment model is supported")

        name = self.args.align_models[0]
        device = self.accelerator.device

        if name == "VideoMAEv2":
            self.vision_encoder = vit_base_patch16_224().to(device)
            self.vision_encoder.from_pretrained(
                os.path.join(self.VFM_CKPT_DIR, "VideoMAEv2", "vit_b_k710_dl_from_giant.pth")
            )
        elif name == "VideoMAE":
            self.vision_encoder = VideoMAE_vit_base_patch16_224().to(device)
            self.vision_encoder.from_pretrained(
                os.path.join(
                    self.VFM_CKPT_DIR,
                    "VideoMAE",
                    "k400_videomae_pretrain_base_patch16_224_frame_16x4_tube_mask_ratio_0_9_e1600.pth",
                )
            )
        elif name == "OminiMAE":
            from finetune.models.cogvideox_t2v_gw_align.models.ssl.omini_mae import (
                vit_base_mae_pretraining,
            )

            self.vision_encoder = vit_base_mae_pretraining().to(device)
            self.vision_encoder.tubelet_size = 2
            self.vision_encoder.patch_size = 16
            self.vision_encoder.embed_dim = 768
        elif name == "VJEPA":
            from finetune.models.cogvideox_t2v_gw_align.models.ssl.JEPA import load_VJEPA

            self.vision_encoder = load_VJEPA(
                device=device,
                pretrained_path=os.path.join(self.VFM_CKPT_DIR, "vjepa_l", "vitl16.pth.tar"),
            )
        elif name == "VJEPA2":
            encoder, _ = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_large")
            self.vision_encoder = encoder.to(device)
            # ``norm`` is the VJEPA2 predictor head; drop it for feature extraction.
            del self.vision_encoder.norm
            self.vision_encoder.norm = torch.nn.Identity()
        else:
            raise NotImplementedError(f"Unsupported alignment model: {name}")

        self.vision_encoder.eval()
        self.vision_encoder.requires_grad_(False)

        self.initialize_alignment_loss()

    def initialize_alignment_loss(self) -> None:
        """Instantiate the GW alignment loss from ``--loss`` and the ``--gw_*`` flags."""
        self.align_loss = build_alignment_loss(self.args.loss, self.args)

    def _get_adapter(self, grid: Tuple[int, int, int]) -> ZeroParamFeatureAdapter:
        """Return (and cache) the parameter-free student -> teacher grid adapter."""
        if grid not in self._adapters:
            self._adapters[grid] = ZeroParamFeatureAdapter(grid=grid)
        return self._adapters[grid]

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------
    @override
    def load_components(self) -> Components:
        components = Components()
        model_path = str(self.args.model_path)

        components.pipeline_cls = CogVideoXPipelineAlign
        components.tokenizer = AutoTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        components.text_encoder = T5EncoderModel.from_pretrained(
            model_path, subfolder="text_encoder"
        )

        # GW mode needs no projector, but the layer indices still drive feature extraction.
        load_kwargs = dict(
            align_layer=self.args.align_layer,
            align_dims=self.args.align_dims,
            projector_dim=getattr(self.args, "projector_dim", 2048),
            align_residual=self.args.align_residual,
            align_attn_residual=self.args.align_attn_residual,
        )
        if self.args.align_layer_secondary is not None:
            load_kwargs["align_layer_secondary"] = self.args.align_layer_secondary

        components.transformer = CogVideoXTransformer3DModelAlign.from_pretrained(
            model_path, subfolder="transformer", **load_kwargs
        )
        components.vae = AutoencoderKLCogVideoX.from_pretrained(model_path, subfolder="vae")
        components.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_path, subfolder="scheduler"
        )
        return components

    @override
    def initialize_pipeline(self) -> CogVideoXPipelineAlign:
        return CogVideoXPipelineAlign(
            tokenizer=self.components.tokenizer,
            text_encoder=self.components.text_encoder,
            vae=self.components.vae,
            transformer=unwrap_model(self.accelerator, self.components.transformer),
            scheduler=self.components.scheduler,
        )

    @override
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self.components.vae
        video = video.to(vae.device, dtype=vae.dtype)
        latent_dist = vae.encode(video).latent_dist
        return latent_dist.sample() * vae.config.scaling_factor

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
        return self.components.text_encoder(
            prompt_token_ids.input_ids.to(self.accelerator.device)
        )[0]

    @override
    def collate_fn(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        ret = {"encoded_videos": [], "prompt_embedding": [], "raw_frames": []}
        for sample in samples:
            ret["encoded_videos"].append(sample["encoded_video"])
            ret["prompt_embedding"].append(sample["prompt_embedding"])
            ret["raw_frames"].append(sample["raw_frames"])

        return {key: torch.stack(value) for key, value in ret.items()}

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def encode_teacher(self, raw_frames: torch.Tensor) -> torch.Tensor:
        """Run the frozen video encoder on the pixel frames.

        Args:
            raw_frames: ``(B, C, F, H, W)`` pixels in ``[-1, 1]``.

        Returns:
            ``(B, N, D2)`` teacher tokens on the teacher's spatio-temporal grid.
        """
        if self.args.align_models[0] not in IMAGENET_NORMALIZED_ENCODERS:
            raise NotImplementedError(
                "GW alignment currently only supports "
                f"{', '.join(IMAGENET_NORMALIZED_ENCODERS)}"
            )

        batch, channels, frames, height, width = raw_frames.shape

        # Normalize to the encoder's expected input range.
        frames_flat = raw_frames.transpose(1, 2).flatten(0, 1)  # (B*F, C, H, W)
        frames_flat = (frames_flat + 1.0) / 2.0
        frames_flat = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(frames_flat)
        video = frames_flat.reshape(batch, frames, channels, height, width).transpose(1, 2)

        # The teacher encodes frames 1..F, so drop the first frame.
        video = video[:, :, 1:]
        frames = video.shape[2]

        # 480x720 -> 160x240: match the encoder's pre-training resolution.
        video_flat = video.transpose(1, 2).flatten(0, 1)
        video_flat = F.interpolate(
            video_flat, (height // 3, width // 3), mode="bicubic", align_corners=False
        )
        video = video_flat.reshape(
            batch, frames, channels, height // 3, width // 3
        ).transpose(1, 2)

        with torch.no_grad():
            encoder = self.vision_encoder
            features = encoder(video)  # (B, N, D2)
            batch, _, dim = features.shape
            features = features.transpose(1, 2).reshape(
                batch,
                dim,
                frames // encoder.tubelet_size,
                (height // 3) // encoder.patch_size,
                (width // 3) // encoder.patch_size,
            )

        return features.flatten(2).transpose(1, 2)  # (B, N, D2)

    def diffusion_loss(
        self, latent: torch.Tensor, predicted_noise: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Velocity-prediction diffusion loss with SNR weighting."""
        scheduler = self.components.scheduler
        latent_pred = scheduler.get_velocity(predicted_noise, latent, timesteps)

        alphas_cumprod = scheduler.alphas_cumprod[timesteps]
        weights = 1 / (1 - alphas_cumprod)
        while len(weights.shape) < len(latent_pred.shape):
            weights = weights.unsqueeze(-1)

        batch_size = latent.shape[0]
        loss = torch.mean(
            (weights * (latent_pred - latent) ** 2).reshape(batch_size, -1), dim=1
        )
        return loss.mean()

    @override
    def compute_loss(self, batch: Dict[str, Any]) -> List[torch.Tensor]:
        prompt_embedding = batch["prompt_embedding"]
        latent = batch["encoded_videos"]  # (B, C, F, H, W)
        raw_frames = batch["raw_frames"]  # (B, C, F, H, W) in [-1, 1]

        batch_size, _, num_frames, height, width = latent.shape

        # --- 1. Frozen teacher features ---------------------------------
        teacher_features = self.encode_teacher(raw_frames)  # (B, N, D2)

        # Student token grid implied by the VAE latent and the patch embedding.
        patch_size = self.state.transformer_config.patch_size
        adapter = self._get_adapter((num_frames, height // patch_size, width // patch_size))

        # --- 2. Diffusion forward pass ----------------------------------
        prompt_embedding = prompt_embedding.view(batch_size, -1, prompt_embedding.shape[-1])
        prompt_embedding = prompt_embedding.to(dtype=latent.dtype)

        timesteps = torch.randint(
            0,
            self.components.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=self.accelerator.device,
        ).long()

        latent = latent.permute(0, 2, 1, 3, 4)  # (B, F, C, H, W) for the transformer
        noise = torch.randn_like(latent)
        noisy_latent = self.components.scheduler.add_noise(latent, noise, timesteps)

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

        predicted_noises, aligns = self.components.transformer(
            hidden_states=noisy_latent,
            encoder_hidden_states=prompt_embedding,
            timestep=timesteps,
            image_rotary_emb=rotary_emb,
            return_dict=False,
        )
        predicted_noise = predicted_noises[0]

        # --- 3. Student features on the teacher's grid (zero parameters) --
        hidden = aligns[0].view(batch_size, -1, aligns[0].shape[-1])
        student_features = adapter(hidden)  # (B, N, D1)

        # --- 4. Losses ----------------------------------------------------
        align_loss = self.align_loss(student_features, teacher_features)
        diffusion_loss = self.diffusion_loss(latent, predicted_noise, timesteps)

        return [diffusion_loss, align_loss]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @override
    def validation_step(
        self, eval_data: Dict[str, Any], pipe: CogVideoXPipelineAlign
    ) -> List[Tuple[str, Image.Image | List[Image.Image]]]:
        prompt = eval_data["prompt"]
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
            base_num_frames = (
                num_frames + transformer_config.patch_size_t - 1
            ) // transformer_config.patch_size_t

        return get_3d_rotary_pos_embed(
            embed_dim=transformer_config.attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(grid_height, grid_width),
            device=device,
        )


register("cogvideox-t2v-gw-align", "lora", CogVideoXT2VGWAlignLoraTrainer)
