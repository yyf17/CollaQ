REGISTRY = {}

# for project CollaQ
from .rnn_agent_CollaQ import RNNAgent as RNNAgent_CollaQ
from .rnn_agent_CollaQ import RNNAttnAgent
from .rnn_interactive_agent import RNNInteractiveAgent, RNNInteractiveAttnAgentV1, RNNInteractiveAttnAgentV2, RNNInteractiveRegAgent, RNNInteractiveAttnAgent
REGISTRY["rnn_CollaQ"] = RNNAgent_CollaQ
REGISTRY["rnn_attn"] = RNNAttnAgent
REGISTRY["rnn_interactive"] = RNNInteractiveAgent
REGISTRY["rnn_interactive_reg"] = RNNInteractiveRegAgent
REGISTRY["rnn_interactive_attnv1"] = RNNInteractiveAttnAgentV1
REGISTRY["rnn_interactive_attnv2"] = RNNInteractiveAttnAgentV2
REGISTRY["rnn_interactive_attn"] = RNNInteractiveAttnAgent

# for project QPLEX,    NDQ,  WQMIX
from .rnn_agent import RNNAgent    #标准的   rnn
REGISTRY["rnn"] = RNNAgent

# for project RODE
from .rode_agent import RODEAgent
REGISTRY["rode"] = RODEAgent

# for project ROMA
from .latent_ce_dis_rnn_agent import LatentCEDisRNNAgent
REGISTRY["latent_ce_dis_rnn"] = LatentCEDisRNNAgent

# for project AIQMIX
from .rnn_agent_AIQMIX import RNNAgent as RNNAgent_AIQMIX
from .ff_agent import FFAgent
from .entity_rnn_agent_AIQMIX import ImagineEntityAttentionRNNAgent, EntityAttentionRNNAgent

REGISTRY["rnn_AIQMIX"] = RNNAgent_AIQMIX
REGISTRY["ff"] = FFAgent
REGISTRY["entity_attend_rnn"] = EntityAttentionRNNAgent
REGISTRY["imagine_entity_attend_rnn"] = ImagineEntityAttentionRNNAgent

# for project WQMIX
from .central_rnn_agent import CentralRNNAgent
REGISTRY["central_rnn"] = CentralRNNAgent

# for project MAVEN
from .noise_rnn_agent import RNNAgent as NoiseRNNAgent
REGISTRY["noise_rnn"] = NoiseRNNAgent

# for project ASN
from .asn_agent import AsnAgent
from .asn_rnn_agent import AsnRNNAgent
from .asn_diff_type_agent import AsnDiffAgent
from .asn_diff_type_rnn_agent import AsnDiffRnnAgent
from .asn_wo_share_diff_type_agent import AsnDiffWoShareAgent

from .dense_agent import DenseAgent
from .dense_rnn_agent import DenseRNNAgent
from .dense_rnn_dueling_agent import DenseRNNDuelingAgent
from .dense_rnn_attention_agent import DenseRNNAttentionAgent
from .dense_rnn_entity_attention_agent import DenseRNNEntityAttentionAgent

# modify add agent type
REGISTRY["asn"] = AsnAgent
REGISTRY['asn_rnn'] = AsnRNNAgent
REGISTRY['asn_diff'] = AsnDiffAgent
REGISTRY['asn_wo_share_diff'] = AsnDiffWoShareAgent
REGISTRY['asn_diff_rnn'] = AsnDiffRnnAgent

REGISTRY['dense'] = DenseAgent
REGISTRY['dense_rnn'] = DenseRNNAgent
REGISTRY['dense_rnn_dueling'] = DenseRNNDuelingAgent
REGISTRY['dense_rnn_attention'] = DenseRNNAttentionAgent
REGISTRY['dense_rnn_entity_attention'] = DenseRNNEntityAttentionAgent


