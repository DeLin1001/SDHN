REGISTRY = {}

from .basic_controller import BasicMAC
from .discrete.non_shared_controller import NonSharedMAC as discrete_NonSharedMAC
from .discrete.maddpg_controller import MADDPGMAC as  discrete_NonSharedMAC
from .discrete.dcg_controller import DeepCoordinationGraphMAC as  discrete_DeepCoordinationGraphMAC
from .discrete.dicg_controller import DICGraphMAC as  discrete_DICGraphMAC
from .discrete.gacg_controller import GroupMessageMAC as  discrete_GroupMessageMAC
from .discrete.HGCN_controller import HGCNMAC as discrete_HGCNMAC
from .discrete.HGCN_controller_sample import HGCNMAC_sample as discrete_HGCNMAC_sample

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["discrete_non_shared_mac"] = discrete_NonSharedMAC
REGISTRY["discrete_maddpg_mac"] = discrete_NonSharedMAC
REGISTRY["discrete_dcg_mac"] = discrete_DeepCoordinationGraphMAC
REGISTRY["discrete_dicg_mac"] = discrete_DICGraphMAC
REGISTRY["discrete_gacg_mac"] = discrete_GroupMessageMAC
REGISTRY["discrete_HGCNMAC"] = discrete_HGCNMAC
REGISTRY["discrete_HGCNMAC_sample"] = discrete_HGCNMAC_sample
  

from .continuous.cqmix_controller import CQMixMAC as continuous_QMixMAC
REGISTRY["continuous_qmix_mac"] = continuous_QMixMAC

from .continuous.ppo_controller import PPOMAC as continuous_PPOMAC
REGISTRY["continuous_ppo_mac"] = continuous_PPOMAC
