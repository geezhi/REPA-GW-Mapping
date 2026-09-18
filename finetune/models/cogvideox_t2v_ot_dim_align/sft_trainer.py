from finetune.models.cogvideox_t2v_ot_dim_align.lora_trainer import CogVideoXT2VOTDimAlignLoraTrainer
from ..utils import register

register("cogvideox-t2v-ot-dim-align", "sft", CogVideoXT2VOTDimAlignLoraTrainer)
register("cogvideox-t2v-ot-dim-align", "lora", CogVideoXT2VOTDimAlignLoraTrainer)
