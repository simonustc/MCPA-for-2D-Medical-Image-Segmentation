# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from functools import partial
from networks.MCPA_Net import MCPANET as MCPANET

logger = logging.getLogger(__name__)


class MCPA(nn.Module):
    def __init__(self, config, num_classes):
        super(MCPA, self).__init__()
        self.vanunet = MCPANET(config=config, num_classes=num_classes, num_heads=config.MODEL.SWIN.NUM_HEADS,
                           embed_dims=config.MODEL.SWIN.EMBED_DIM, depths=config.MODEL.SWIN.DEPTHS)

    def forward(self, x):
        logits = self.vanunet(x)
        return logits

    def load_from(self, config):
        # 加载预训练模型
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            # if "model" not in pretrained_dict:
            #     print("---start load pretrained modle by splitting---")
            #     pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
            #     for k in list(pretrained_dict.keys()):
            #         if "output" in k:
            #             print("delete key:{}".format(k))
            #             del pretrained_dict[k]
            #     msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
            #     # print(msg)
            #     return
            # pretrained_dict = pretrained_dict['state_dict']
            # print(list(pretrained_dict.keys()))
            print("---start load pretrained modle of van encoder---")

            model_dict = self.vanunet.state_dict()
            # print(list(model_dict.keys()))
            k_head_weight = model_dict['head.weight']
            k_head_bias = model_dict['head.bias']

            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if 'head.weight' in k:
                    v = k_head_weight
                    current_k = k
                    full_dict.update({current_k: v})
                if 'head.bias' in k:
                    v = k_head_bias
                    current_k = k
                    full_dict.update({current_k: v})
                if "block1" in k:
                    current_layer_num = 4
                    current_k = "up_block" + str(current_layer_num) + k[6:]
                    full_dict.update({current_k: v})
                if "block2" in k:
                    current_layer_num = 3
                    current_k = "up_block" + str(current_layer_num) + k[6:]
                    full_dict.update({current_k: v})
                if "block3" in k:
                    current_layer_num = 2
                    current_k = "up_block" + str(current_layer_num) + k[6:]
                    full_dict.update({current_k: v})
                if "norm1" in k and 'block' not in k and 'patch' not in k:
                    current_layer_num = 4
                    current_k = "up_norm" + str(current_layer_num)
                    full_dict.update({current_k: v})
                if "norm2" in k and 'block' not in k and 'patch' not in k:
                    current_layer_num = 3
                    current_k = "up_norm" + str(current_layer_num)
                    full_dict.update({current_k: v})
                if "norm3" in k and 'block' not in k and 'patch' not in k:
                    current_layer_num = 2
                    current_k = "up_norm" + str(current_layer_num)
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.vanunet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
