from ..cogvideox_t2v_merger_align.lora_trainer import CogVideoXT2VMergerAlignLoraTrainer
from ..utils import register


class CogVideoXT2VMergerAlignSftTrainer(CogVideoXT2VMergerAlignLoraTrainer):
    pass


register("cogvideox-t2v-merger-align", "sft", CogVideoXT2VMergerAlignSftTrainer)
