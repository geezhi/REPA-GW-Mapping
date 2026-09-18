from finetune.models.cogvideox_t2v_direct_align.lora_trainer import CogVideoXT2VDirectAlignLoraTrainer
from ..utils import register

register("cogvideox-t2v-direct-align", "sft", CogVideoXT2VDirectAlignLoraTrainer)
