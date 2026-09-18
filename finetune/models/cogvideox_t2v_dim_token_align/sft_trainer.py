from finetune.models.cogvideox_t2v_dim_token_align.lora_trainer import CogVideoXT2VDimTokenAlignLoraTrainer
from ..utils import register

register("cogvideox-t2v-dim-token-align", "sft", CogVideoXT2VDimTokenAlignLoraTrainer)
