from typing import Union, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, MaxPool2d
from mmyolo.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from .base_backbone import BaseBackbone

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
            deepen_factor,
            widen_factor,
            input_channels=input_channels,
            out_indices=out_indices,
            plugins=plugins,
            frozen_stages=frozen_stages,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
            init_cfg=init_cfg)
    
    def build_stem_layer(self) -> nn.Module:
        """Build a stem layer"""
        stage = []
        conv_layer1 = ConvModule(in_channels=self.input_channels,
                                 out_channels=32,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg)
        stage.append(conv_layer1)
        maxpool1 = MaxPool2d(kernel_size=2, stride=2)
        stage.append(maxpool1)
        
        conv_layer2 = ConvModule(in_channels=32,
                                 out_channels=64,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1,
                                 norm_cfg=self.norm_cfg,
                                 act_cfg=self.act_cfg)
        stage.append(conv_layer2)
        maxpool2 = MaxPool2d(kernel_size=2, stride=2)
        stage.append(maxpool2)
        
        return stage

        
        
        