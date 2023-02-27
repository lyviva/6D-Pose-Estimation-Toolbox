from typing import Union, List, Tuple

import torch
import torch.nn as nn
from mmyolo.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from mmyolo.models import BaseBackbone

@MODELS.register_module()
class YOLOv2Darknet(BaseBackbone):
    arch_settings = {
        'P5': [[64, 128, 3, ], [128, 256, 3, ],
               [256, 512, 3, ], [512, 1024, 5, ]]
    }
    
    def __init__(self,
                 arch: str = 'P5',
                 last_stage_out_channels: int = 1024,
                 plugins: Union[dict, List[dict]] = None,
                 deepen_factor: float = 1.0,
                 widen_factor: float = 1.0,
                 input_channels: int = 3,
                 out_indices: Tuple[int] = (4),
                 frozen_stages: int = -1,
                 norm_cfg: ConfigType = dict(
                     type='BN', momentum=0.03, eps=0.001),
                 act_cfg: ConfigType = dict(type='SiLU', inplace=True),
                 norm_eval: bool = False,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            self.arch_setting[arch],
        )