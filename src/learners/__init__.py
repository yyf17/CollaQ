from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .q_interactive_learner import QInteractiveLearner
from .q_influence_learner import QInfluenceLearner
from .q_explore_learner import QExploreLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["q_interactive_learner"] = QInteractiveLearner
REGISTRY["q_influence_learner"] = QInfluenceLearner
REGISTRY["q_explore_learner"] = QExploreLearner

# for QPLEX
from .dmaq_qatten_learner import DMAQ_qattenLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner

# for RODE
from .rode_learner import RODELearner
REGISTRY["rode_learner"] = RODELearner

# for ROMA
from .latent_q_learner import LatentQLearner
REGISTRY['latent_q_learner'] =LatentQLearner

# for NDQ
from .categorical_q_learner import CateQLearner
REGISTRY["cate_q_learner"] = CateQLearner

# for AIQMIX
# from .q_learner_AIQMIX import QLearner as q_learner_AIQMIX
# REGISTRY["q_learner_AIQMIX"] = q_learner_AIQMIX


# for WQMIX
from .max_q_learner import MAXQLearner
from .max_q_learner_ddpg import DDPGQLearner
from .max_q_learner_sac import SACQLearner
from .q_learner_w import QLearner as WeightedQLearner
from .qatten_learner import QattenLearner

REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["ddpg"] = DDPGQLearner
REGISTRY["sac"] = SACQLearner
REGISTRY["w_q_learner"] = WeightedQLearner
REGISTRY["qatten_learner"] = QattenLearner

# for MAVEN
from .noise_q_learner import QLearner as NoiseQLearner  # mixer 的名字需要注意 “qmix”
from .actor_critic_learner import ActorCriticLearner
from .qtran_learner_MAVEN import QLearner as QTranLearner_MAVEN

REGISTRY["noise_q_learner"] = NoiseQLearner
REGISTRY["actor_critic_learner"] = ActorCriticLearner
REGISTRY["qtran_learner_MAVEN"] = QTranLearner_MAVEN

# for LICA
from .lica_learner import LICALearner
REGISTRY["lica_learner"] = LICALearner