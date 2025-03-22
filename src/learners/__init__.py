from .discrete.q_learner import QLearner as discrete_QLearner
from .discrete.coma_learner import COMALearner as discrete_COMALearner
from .discrete.qtran_learner import QLearner as discrete_QTranLearner
from .discrete.actor_critic_learner import ActorCriticLearner as discrete_ActorCriticLearner
from .discrete.actor_critic_pac_learner import PACActorCriticLearner as discrete_PACActorCriticLearner
from .discrete.actor_critic_pac_dcg_learner import PACDCGLearner as discrete_PACDCGLearner
from .discrete.maddpg_learner import MADDPGLearner as discrete_MADDPGLearner
from .discrete.ppo_learner import PPOLearner as discrete_PPOLearner
from .discrete.dcg_learner import DCGLearner as discrete_DCGLearner
from .discrete.gacg_learner import GroupQLearner as discrete_GroupQLearner
from .discrete.vast_learner import VastQLearner as discrete_VastQLearner
from .discrete.hgcn_learner import HGCNLearner as discrete_HGCNLearner
from .discrete.ppo_learner_hg import PPOLearner_hg as discrete_PPOLearner_hg
REGISTRY = {}
REGISTRY["discrete_q_learner"] = discrete_QLearner
REGISTRY["discrete_coma_learner"] = discrete_COMALearner
REGISTRY["discrete_qtran_learner"] = discrete_QTranLearner
REGISTRY["discrete_actor_critic_learner"] = discrete_ActorCriticLearner
REGISTRY["discrete_maddpg_learner"] = discrete_MADDPGLearner
REGISTRY["discrete_ppo_learner"] = discrete_PPOLearner
REGISTRY["discrete_pac_learner"] = discrete_PACActorCriticLearner
REGISTRY["discrete_pac_dcg_learner"] = discrete_PACDCGLearner
REGISTRY["discrete_dcg_learner"] = discrete_DCGLearner
REGISTRY["discrete_gacg_learner"] = discrete_GroupQLearner
REGISTRY["discrete_vast_learner"] = discrete_VastQLearner
REGISTRY["discrete_hgcn_learner"] = discrete_HGCNLearner
REGISTRY["discrete_ppo_learner_hg"] = discrete_PPOLearner_hg


from .continuous.cq_learner import CQLearner as continuous_QLearner
REGISTRY["continuous_cq_learner"] = continuous_QLearner
from .continuous.facmac_learner import FACMACLearner
REGISTRY["continuous_facmac_learner"] = FACMACLearner
from .continuous.ppo_learner import PPOLearner as continuous_PPOLearner
REGISTRY["continuous_ppo_learner"] = continuous_PPOLearner
