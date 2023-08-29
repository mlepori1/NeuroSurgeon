from typing import List

from transformers import PretrainedConfig

from ..Models.model_configs import CircuitConfig


class ResidualUpdateModelConfig(PretrainedConfig):
    """A config object defining the behavior of the ResidualUpdateModel

    :param model_type: The type of model that you are hooking. One of ["bert", "gpt", "gptneox", "vit"]
    :type model_type: str
    :param target_layers: A list of layers to hook
    :type target_layers: List[int]
    :param mlp: Whether to hook the mlp layers
    :type mlp: bool
    :param attn: Whether to hook the attention layers
    :type attn: bool
    :param updates: Whether to store updates to the residual stream, defaults to True
    :type updates: bool, optional
    :param stream: Whether to store the intermediate residual stream values, defaults to False
    :type stream: bool, optional
    """

    def __init__(
        self,
        model_type: str,
        target_layers: List[int],
        mlp: bool,
        attn: bool,
        updates: bool = True,
        stream: bool = False,
    ):
        super().__init__()
        self.model_type = model_type
        self.target_layers = target_layers
        self.mlp = mlp
        self.attn = attn
        self.updates = updates
        self.stream = stream


class SubnetworkProbeConfig(PretrainedConfig):
    """A config object defining the behavior of the SubnetworkProbe

    :param probe_vectors: The entries in a ResidualUpdateModel's vector_cache that one will train a probe on. Ex. attn_update_1, mlp_stream_5
    :type probe_vectors: str
    :param n_classes: The number of classes in the probe task
    :type n_classes: int
    :param circuit_config: A CircuitConfig object defining the masking behavior of the model
    :type circuit_config: CircuitConfig
    :param resid_config: A ResidualUpdateModelConfig defining the behavior of the residual update model. Make sure that the probe_vectors argument aligns with this config!
    :type resid_config: ResidualUpdateModelConfig
    :param intermediate_size: If an MLP probe is required, the dimensionality of the hidden layer, defaults to -1
    :type intermediate_size: int, optional
    :param labeling: Either "sequence" or "token" - whether to expect one label per input or many, defaults to "sequence"
    :type labeling: str, optional
    """

    def __init__(
        self,
        probe_vectors: str,
        n_classes: int,
        circuit_config: CircuitConfig,
        resid_config: ResidualUpdateModelConfig,
        intermediate_size: int = -1,
        labeling: str = "sequence",
    ):
        super().__init__()
        self.probe_vectors = probe_vectors
        self.circuit_config = circuit_config
        self.resid_config = resid_config
        self.intermediate_size = intermediate_size
        self.n_classes = n_classes
        self.labeling = labeling
