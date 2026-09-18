from ..cogvideox_t2v_local_gram_flow.lora_trainer import CogVideoXT2VLocalGramFlowLoraTrainer
from ..utils import register


class CogVideoXT2VLocalGramFlowSftTrainer(CogVideoXT2VLocalGramFlowLoraTrainer):
    pass


register("cogvideox-t2v-local-gram-flow", "sft", CogVideoXT2VLocalGramFlowSftTrainer)
register("cogvideox-t2v-local-gram-flow-align", "sft", CogVideoXT2VLocalGramFlowSftTrainer)
