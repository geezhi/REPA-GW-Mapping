from ..cogvideox_t2v_gw_align.lora_trainer import CogVideoXT2VGWAlignLoraTrainer
from ..utils import register


class CogVideoXT2VGWAlignSftTrainer(CogVideoXT2VGWAlignLoraTrainer):
    pass


register("cogvideox-t2v-gw-align", "sft", CogVideoXT2VGWAlignSftTrainer)
