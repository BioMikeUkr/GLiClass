from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
logger = logging.get_logger(__name__)


class CrossEncoderHeadConfig(PretrainedConfig):
    def __init__(self, active_layers=None, z_steps=1, inner_batch_size: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.active_layers = active_layers or [-2, -1]
        self.z_steps = z_steps
        self.inner_batch_size = inner_batch_size

