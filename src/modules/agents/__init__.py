from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent

from .mlp_agent import MLPAgent
from .comix_agent import CEMAgent, CEMRecurrentAgent
from .qmix_agent import QMIXRNNAgent, FFAgent

REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent

REGISTRY["mlp"] = MLPAgent
REGISTRY["cem"] = CEMAgent
REGISTRY["cemrnn"] = CEMRecurrentAgent
REGISTRY["qmixrnn"] = QMIXRNNAgent
REGISTRY["ff"] = FFAgent

from .gauss_agent import GaussianAgent
REGISTRY["gauss"] = GaussianAgent
