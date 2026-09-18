import argparse
import datetime
import logging
from pathlib import Path
from typing import Any, List, Literal, Tuple

from pydantic import BaseModel, ValidationInfo, field_validator

# 你不仅要在 parse_args 函数里添加命令行参数，还需要在 Args 类中定义相应的字段
class Args(BaseModel):
    ########## Model ##########
    model_path: Path
    model_name: str
    model_type: Literal["i2v", "t2v"]
    training_type: Literal["lora", "sft"] = "lora"

    ########## Output ##########
    output_dir: Path = Path("train_results/{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now()))
    report_to: Literal["tensorboard", "wandb", "all", "none"] | None = None
    tracker_name: str = "finetrainer-cogvideo"

    ########## Data ###########
    data_root: Path
    caption_column: Path
    image_column: Path | None = None
    video_column: Path
    train_data_path: List[str] = None
    precomputing: bool = None

    ########## VideoREPA ###########
    align_models: List[str] = ["VideoMAEv2"]
    align_layer: int = 18,          
    align_dims: List[int] =[768]    
    projector_dim: int = 2048       
    proj_coeff: float = None
    loss: str = "cosine"            
    margin: float = 0.1
    align_timestep_threshold: float = 0.5  # Only align when timestep > threshold * num_train_timesteps
    comment: str = ""
    pretrained_projector_path: str | None = None  # Path to pretrained projector weights (freeze projector if set)
    projector_lr_scale: float = 0.0  # LR scale for projector (0=freeze, >0=finetune with scale*lr)
    teacher_pca_path: str | None = None  # Path to teacher PCA matrix (multi-layer teacher -> 1920d, no projector needed)
    # Dual-layer alignment: secondary layer with a different loss
    align_layer_secondary: int | None = None  # e.g. 5 for shallow layer temporal_diff_only alignment
    loss_secondary: str | None = None  # e.g. "cosine_similarity_temporal_diff_only" for the secondary layer
    proj_coeff_secondary: float | None = None  # coefficient for secondary alignment loss, defaults to proj_coeff if None
    align_residual: bool = False  # if True, align block residual (x_out - x_in) instead of x_out
    align_attn_residual: bool = False  # if True, align attention residual (gate_msa * attn_out) instead of x_out
    # Gromov-Wasserstein alignment parameters
    gw_reg: float = 0.1  # entropic regularization for GW
    gw_outer_iters: int = 20  # outer iterations for GW linearization
    gw_sinkhorn_iters: int = 50  # inner Sinkhorn iterations
    gw_sample_size: int = 0  # 0 = use ALL tokens (no subsampling) for GW computation
    gw_outer_tol: float = 1e-4  # convergence tolerance for outer GW loop (0 to disable)
    gw_sinkhorn_tol: float = 1e-4  # convergence tolerance for inner Sinkhorn (0 to disable)
    gw_margin: float = 0.0  # hinge on the per-token cosine distance (0 disables)
    # Deprecated: kept only for backward compatibility with older run scripts.
    gw_update_interval: int = 1  # unused (the plan is recomputed every sample)
    gw_distance_type: str = 'euclidean'  # unused (dimension distances are always cosine)
    # Direct relational alignment parameters (no GW)
    direct_align_sample_size: int = 0  # subsample tokens for direct relational loss (0 = use all)
    # Dimension-level GW alignment parameters
    dim_gw_reg: float = 0.1  # entropic regularization for dimension-level GW
    dim_gw_outer_iters: int = 30  # outer iterations for dimension-level GW
    dim_gw_sinkhorn_iters: int = 100  # inner Sinkhorn iterations for dimension-level GW
    dim_gw_dim_sample_size: int = 0  # subsample dimensions (0 = use all D1, D2)
    dim_gw_token_sample_size: int = 512  # subsample tokens for computing dim distance matrices
    dim_gw_update_interval: int = 50  # recompute dimension transport plan every N steps
    dim_gw_outer_tol: float = 1e-4  # convergence tolerance for outer loop
    dim_gw_sinkhorn_tol: float = 1e-4  # convergence tolerance for Sinkhorn
    # Dimension-guided token alignment (Plan A) parameters
    dim_token_sample_size_for_loss: int = 0  # subsample tokens for per-token loss (0 = use all)
    # OT Dimension Alignment parameters
    ot_reg: float = 0.1  # entropic regularization for OT
    ot_sinkhorn_iters: int = 100  # Sinkhorn iterations for OT
    ot_tol: float = 1e-4  # convergence tolerance for OT Sinkhorn
    # Local Gram Flow alignment parameters
    lgf_patch_size_h: int = 5  # patch height for local Gram (H=10 -> 2 patches)
    lgf_patch_size_w: int = 5  # patch width for local Gram (W=15 -> 3 patches)
    lgf_alpha: float = 1.0  # weight for temporal flow loss (delta_G alignment)
    lgf_beta: float = 0.0  # weight for static Gram loss (0 = pure temporal flow)
    lgf_multiscale: bool = False  # if True, use multi-scale local Gram flow [(2,3),(5,5),(10,15)]
    lgf_temporal_diff_weight: float = 0.0  # weight for temporal diff cosine loss (0 = disabled)
    lgf_gram_weight: float = 0.0  # weight for global Gram loss in native space (0 = disabled)

    ########## Training #########
    resume_from_checkpoint: Path | None = None

    seed: int | None = None
    train_epochs: int
    train_steps: int | None = None
    checkpointing_steps: int = 200
    checkpointing_limit: int = 10

    batch_size: int
    gradient_accumulation_steps: int = 1

    train_resolution: Tuple[int, int, int]  # shape: (frames, height, width)

    #### deprecated args: video_resolution_buckets
    # if use bucket for training, should not be None
    # Note1: At least one frame rate in the bucket must be less than or equal to the frame rate of any video in the dataset
    # Note2:  For cogvideox, cogvideox1.5
    #   The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #   The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # video_resolution_buckets: List[Tuple[int, int, int]] | None = None

    mixed_precision: Literal["no", "fp16", "bf16"]

    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    beta3: float = 0.98
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 100
    lr_num_cycles: int = 1
    lr_power: float = 1.0

    num_workers: int = 8
    pin_memory: bool = True

    gradient_checkpointing: bool = True
    enable_slicing: bool = True
    enable_tiling: bool = True
    nccl_timeout: int = 1800

    ########## Lora ##########
    rank: int = 128
    lora_alpha: int = 64
    target_modules: List[str] = ["to_q", "to_k", "to_v", "to_out.0"]

    ########## Validation ##########
    do_validation: bool = False
    validation_steps: int | None  # if set, should be a multiple of checkpointing_steps
    validation_dir: Path | None  # if set do_validation, should not be None
    validation_prompts: str | None  # if set do_validation, should not be None
    validation_images: str | None  # if set do_validation and model_type == i2v, should not be None
    validation_videos: str | None  # if set do_validation and model_type == v2v, should not be None
    gen_fps: int = 15

    #### deprecated args: gen_video_resolution
    # 1. If set do_validation, should not be None
    # 2. Suggest selecting the bucket from `video_resolution_buckets` that is closest to the resolution you have chosen for fine-tuning
    #        or the resolution recommended by the model
    # 3. Note:  For cogvideox, cogvideox1.5
    #        The frame rate set in the bucket must be an integer multiple of 8 (spatial_compression_rate[4] * path_t[2] = 8)
    #        The height and width set in the bucket must be an integer multiple of 8 (temporal_compression_rate[8])
    # gen_video_resolution: Tuple[int, int, int] | None  # shape: (frames, height, width)

    @field_validator("image_column")
    def validate_image_column(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("model_type") == "i2v" and not v:
            logging.warning(
                "No `image_column` specified for i2v model. Will automatically extract first frames from videos as conditioning images."
            )
        return v

    @field_validator("validation_dir", "validation_prompts")
    def validate_validation_required_fields(cls, v: Any, info: ValidationInfo) -> Any:
        values = info.data
        if values.get("do_validation") and not v:
            field_name = info.field_name
            raise ValueError(f"{field_name} must be specified when do_validation is True")
        return v

    @field_validator("validation_images")
    def validate_validation_images(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "i2v" and not v:
            raise ValueError("validation_images must be specified when do_validation is True and model_type is i2v")
        return v

    @field_validator("validation_videos")
    def validate_validation_videos(cls, v: str | None, info: ValidationInfo) -> str | None:
        values = info.data
        if values.get("do_validation") and values.get("model_type") == "v2v" and not v:
            raise ValueError("validation_videos must be specified when do_validation is True and model_type is v2v")
        return v

    @field_validator("validation_steps")
    def validate_validation_steps(cls, v: int | None, info: ValidationInfo) -> int | None:
        values = info.data
        if values.get("do_validation"):
            if v is None:
                raise ValueError("validation_steps must be specified when do_validation is True")
            if values.get("checkpointing_steps") and v % values["checkpointing_steps"] != 0:
                raise ValueError("validation_steps must be a multiple of checkpointing_steps")
        return v

    @field_validator("train_resolution")
    def validate_train_resolution(cls, v: Tuple[int, int, int], info: ValidationInfo) -> str:
        try:
            frames, height, width = v

            # Check if (frames - 1) is multiple of 8
            if (frames - 1) % 8 != 0:
                raise ValueError("Number of frames - 1 must be a multiple of 8")

            # Check resolution for cogvideox-5b models
            model_name = info.data.get("model_name", "")
            if model_name in ["cogvideox-5b-i2v", "cogvideox-5b-t2v"]:
                if (height, width) != (480, 720):
                    raise ValueError("For cogvideox-5b models, height must be 480 and width must be 720")

            return v

        except ValueError as e:
            if (
                str(e) == "not enough values to unpack (expected 3, got 0)"
                or str(e) == "invalid literal for int() with base 10"
            ):
                raise ValueError("train_resolution must be in format 'frames x height x width'")
            raise e

    @field_validator("mixed_precision")
    def validate_mixed_precision(cls, v: str, info: ValidationInfo) -> str:
        if v == "fp16" and "cogvideox-2b" not in str(info.data.get("model_path", "")).lower():
            logging.warning(
                "All CogVideoX models except cogvideox-2b were trained with bfloat16. "
                "Using fp16 precision may lead to training instability."
            )
        return v

    @classmethod
    def parse_args(cls):
        """Parse command line arguments and return Args instance"""
        parser = argparse.ArgumentParser()
        # Required arguments
        parser.add_argument("--model_path", type=str, required=True)
        parser.add_argument("--model_name", type=str, required=True)
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--training_type", type=str, required=True)
        parser.add_argument("--output_dir", type=str, required=True)
        parser.add_argument("--data_root", type=str, required=True)
        parser.add_argument("--caption_column", type=str, required=True)
        parser.add_argument("--video_column", type=str, required=True)
        parser.add_argument("--train_resolution", type=str, required=True)
        parser.add_argument("--report_to", type=str, required=True)
        
        
        # for VideoREPA
        parser.add_argument("--train_data_path", type=str, nargs='+', required=True, help="List of train_data_path")
        parser.add_argument("--align_models", type=str, nargs='+', help="List of alignment models", choices=["VideoMAEv2", "DINOv2", "VJEPA", "VideoMAE", "OminiMAE", "VJEPA2"])
        parser.add_argument("--align_dims", type=int, nargs='+', help="List of alignment dims")
        parser.add_argument("--align_layer", type=int, help="The target layers in the transformer layer")
        parser.add_argument("--projector_dim", type=int, default=2048, help="The intermidiate dimension for the projector MLPs")
        parser.add_argument("--proj_coeff", type=float, help="The coefficient of the projection loss term")
        parser.add_argument("--loss", type=str, default="cosine", choices=["token_relation_distillation", "token_relation_distillation_only_spatial", "token_relation_distillation_only_temporal", "cosine_similarity", "cosine_similarity_dual_timestep", "cosine_similarity_temporal_diff", "cosine_similarity_temporal_diff_only", "gw_relational", "gw_gram", "token_similarity_matrix", "cka_alignment", "merger_align", "fixed_proj_align", "fixed_proj_temporal_diff", "local_gram_flow", "gram_matrix"])
        parser.add_argument("--margin", type=float, default=0.1, help="Margin of the TRD loss")
        parser.add_argument("--align_timestep_threshold", type=float, default=0.5, help="Only align when timestep > threshold * num_train_timesteps (high noise region)")
        parser.add_argument("--comment", type=str, default="", help="Comment for wandb run_name and the output path")
        parser.add_argument("--pretrained_projector_path", type=str, default=None, help="Path to pretrained projector weights. If set, projector will be loaded and frozen (unless projector_lr_scale > 0).")
        parser.add_argument("--projector_lr_scale", type=float, default=0.0, help="LR scale for projector. 0=freeze, >0=finetune with this scale * base lr.")
        parser.add_argument("--teacher_pca_path", type=str, default=None, help="Path to teacher PCA matrix. If set, uses multi-layer teacher features projected to 1920d without projector.")
        # Dual-layer alignment
        parser.add_argument("--align_layer_secondary", type=int, default=None, help="Secondary alignment layer (e.g. 5 for shallow layer)")
        parser.add_argument("--loss_secondary", type=str, default=None, choices=["cosine_similarity", "cosine_similarity_temporal_diff", "cosine_similarity_temporal_diff_only"], help="Loss type for secondary alignment layer")
        parser.add_argument("--proj_coeff_secondary", type=float, default=None, help="Coefficient for secondary alignment loss")
        parser.add_argument("--align_residual", action="store_true", default=False, help="If set, align block residual (x_out - x_in) instead of x_out")
        parser.add_argument("--align_attn_residual", action="store_true", default=False, help="If set, align attention residual (gate_msa * attn_out) instead of x_out")
        # Gromov-Wasserstein alignment parameters
        parser.add_argument("--gw_reg", type=float, default=0.1, help="Entropic regularization for GW")
        parser.add_argument("--gw_outer_iters", type=int, default=20, help="Outer iterations for GW linearization")
        parser.add_argument("--gw_sinkhorn_iters", type=int, default=50, help="Inner Sinkhorn iterations")
        parser.add_argument("--gw_sample_size", type=int, default=0, help="Subsample tokens for GW computation (0 = use ALL tokens)")
        parser.add_argument("--gw_outer_tol", type=float, default=1e-4, help="Convergence tolerance for outer GW loop (0 to disable)")
        parser.add_argument("--gw_sinkhorn_tol", type=float, default=1e-4, help="Convergence tolerance for inner Sinkhorn (0 to disable)")
        parser.add_argument("--gw_margin", type=float, default=0.0, help="Hinge margin on the per-token cosine distance (0 disables)")
        # Deprecated: accepted so that older run scripts keep working, but ignored.
        parser.add_argument("--gw_update_interval", type=int, default=1, help=argparse.SUPPRESS)
        parser.add_argument("--gw_distance_type", type=str, default='euclidean', choices=['cosine', 'euclidean'], help=argparse.SUPPRESS)
        # Direct relational alignment parameters
        parser.add_argument("--direct_align_sample_size", type=int, default=0, help="Subsample tokens for direct relational loss (0 = use all 3600 tokens)")
        # Dimension-level GW alignment parameters
        parser.add_argument("--dim_gw_reg", type=float, default=0.1, help="Entropic regularization for dimension-level GW")
        parser.add_argument("--dim_gw_outer_iters", type=int, default=30, help="Outer iterations for dimension-level GW")
        parser.add_argument("--dim_gw_sinkhorn_iters", type=int, default=100, help="Inner Sinkhorn iterations for dimension-level GW")
        parser.add_argument("--dim_gw_dim_sample_size", type=int, default=0, help="Subsample dimensions (0 = use all D1, D2)")
        parser.add_argument("--dim_gw_token_sample_size", type=int, default=512, help="Subsample tokens for computing dimension distance matrices")
        parser.add_argument("--dim_gw_update_interval", type=int, default=50, help="Recompute dimension transport plan every N steps")
        parser.add_argument("--dim_gw_outer_tol", type=float, default=1e-4, help="Convergence tolerance for outer dimension GW loop")
        parser.add_argument("--dim_gw_sinkhorn_tol", type=float, default=1e-4, help="Convergence tolerance for dimension Sinkhorn")
        # Dimension-guided token alignment (Plan A) parameters
        parser.add_argument("--dim_token_sample_size_for_loss", type=int, default=0, help="Subsample tokens for per-token cosine loss (0 = use all 3600 tokens)")
        # OT Dimension Alignment parameters
        parser.add_argument("--ot_reg", type=float, default=0.1, help="Entropic regularization for OT dimension alignment")
        parser.add_argument("--ot_sinkhorn_iters", type=int, default=100, help="Sinkhorn iterations for OT dimension alignment")
        parser.add_argument("--ot_tol", type=float, default=1e-4, help="Convergence tolerance for OT Sinkhorn")
        # Local Gram Flow alignment parameters
        parser.add_argument("--lgf_patch_size_h", type=int, default=5, help="Patch height for local Gram (H=10 -> 2 patches with p=5)")
        parser.add_argument("--lgf_patch_size_w", type=int, default=5, help="Patch width for local Gram (W=15 -> 3 patches with p=5)")
        parser.add_argument("--lgf_alpha", type=float, default=1.0, help="Weight for temporal flow loss (delta_G alignment)")
        parser.add_argument("--lgf_beta", type=float, default=0.0, help="Weight for static Gram loss (0 = pure temporal flow)")
        parser.add_argument("--lgf_multiscale", action="store_true", default=False, help="Use multi-scale local Gram flow [(2,3),(5,5),(10,15)]")
        parser.add_argument("--lgf_temporal_diff_weight", type=float, default=0.0, help="Weight for temporal diff cosine loss (0 = disabled)")
        parser.add_argument("--lgf_gram_weight", type=float, default=0.0, help="Weight for global Gram loss in native space before projector (0 = disabled)")
    
        # Training hyperparameters
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--train_epochs", type=int, default=10)
        parser.add_argument("--train_steps", type=int, default=None)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
        parser.add_argument("--batch_size", type=int, default=1)
        parser.add_argument("--learning_rate", type=float, default=2e-5)
        parser.add_argument("--optimizer", type=str, default="adamw")
        parser.add_argument("--beta1", type=float, default=0.9)
        parser.add_argument("--beta2", type=float, default=0.95)
        parser.add_argument("--beta3", type=float, default=0.98)
        parser.add_argument("--epsilon", type=float, default=1e-8)
        parser.add_argument("--weight_decay", type=float, default=1e-4)
        parser.add_argument("--max_grad_norm", type=float, default=1.0)

        # Learning rate scheduler
        parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
        parser.add_argument("--lr_warmup_steps", type=int, default=100)
        parser.add_argument("--lr_num_cycles", type=int, default=1)
        parser.add_argument("--lr_power", type=float, default=1.0)

        # Data loading
        parser.add_argument("--num_workers", type=int, default=8)
        parser.add_argument("--pin_memory", type=bool, default=True)
        parser.add_argument("--image_column", type=str, default=None)
        parser.add_argument("--precomputing", action="store_true", help="Enable data precomputing. This should be enabled when performing training the first time")

        # Model configuration
        parser.add_argument("--mixed_precision", type=str, default="no")
        parser.add_argument("--gradient_checkpointing", type=bool, default=True)
        parser.add_argument("--enable_slicing", type=bool, default=True)
        parser.add_argument("--enable_tiling", type=bool, default=True)
        parser.add_argument("--nccl_timeout", type=int, default=1800)

        # LoRA parameters
        parser.add_argument("--rank", type=int, default=128)
        parser.add_argument("--lora_alpha", type=int, default=64)
        parser.add_argument("--target_modules", type=str, nargs="+", default=["to_q", "to_k", "to_v", "to_out.0"])

        # Checkpointing
        parser.add_argument("--checkpointing_steps", type=int, default=200)
        parser.add_argument("--checkpointing_limit", type=int, default=10)
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)

        # Validation
        parser.add_argument("--do_validation", type=lambda x: x.lower() == 'true', default=False)
        parser.add_argument("--validation_steps", type=int, default=None)
        parser.add_argument("--validation_dir", type=str, default=None)
        parser.add_argument("--validation_prompts", type=str, default=None)
        parser.add_argument("--validation_images", type=str, default=None)
        parser.add_argument("--validation_videos", type=str, default=None)
        parser.add_argument("--gen_fps", type=int, default=15)

        args = parser.parse_args()
        # Convert video_resolution_buckets string to list of tuples
        frames, height, width = args.train_resolution.split("x")
        args.train_resolution = (int(frames), int(height), int(width))

        return cls(**vars(args))
