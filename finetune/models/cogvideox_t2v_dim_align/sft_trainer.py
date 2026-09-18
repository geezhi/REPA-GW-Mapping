from finetune.models.cogvideox_t2v_dim_align.lora_trainer import CogVideoXT2VDimAlignLoraTrainer
from ..utils import register

register("cogvideox-t2v-dim-align", "sft", CogVideoXT2VDimAlignLoraTrainer)
