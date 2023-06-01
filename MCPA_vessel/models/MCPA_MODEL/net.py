# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import torch
import torch.nn as nn
from models.MAIN_FINE import MCPANET
logger = logging.getLogger(__name__)


def cal_foreground_ratio(label):
    """
    calculate the ratio of the foreground.
    """
    N, _, H, W = label.shape
    ratio = []
    assert label.max() == 1
    for n in range(N):
        mol = label[n, ...].sum()
        den = H * W
        ratio.append(mol / den)
    ratio = torch.tensor(ratio)
    # pdb.set_trace()
    assert ratio.max() <= 1, "Please check label ratio!"
    return ratio


class ResultsErasing(object):
    # default=0.1
    def __init__(self, top_n=0.10):
        self.top_n = top_n

    def __call__(self, input, x, ratio):
        # input: shape [N, 2, H, W]
        # x:     shape [N, C, H, W] input image
        # ratio: shape[N,]
        # f_p: number of foreground pixels (f_p)

        # x_u: shape [N, 1, H, W]

        # N, C, H, W = x_u.shape
        x2 = x.clone()
        N, C, H, W = input.shape
        for i_c in range(C):

            for n in range(N):  # N-dimension index one by one
                f_p = int(ratio[n] * H * W)

                max_list = input[n, i_c, ...]
                max_list = max_list.flatten()
                max_list = max_list.topk(int(f_p * self.top_n))[0]

                if len(max_list)==0:
                    max_list_min=1
                else:
                    max_list_min = max_list[-1]

                for c in range(x.shape[1]):  # C-dimension index one by one
                    x2[n, c, ...][input[n, i_c, ...] > max_list_min] = 0  # input 肯定单channel，因为二分类

        return x2


class MCPA(nn.Module):
    def __init__(self, num_classes):
        super(MCPA, self).__init__()
        self.net1 = MCPANET(inplanes=1, num_classes=num_classes, layers=3, filters=16)
        self.net2 = MCPANET(inplanes=1, num_classes=num_classes, layers=4, filters=16)

        self.resultsErasing = ResultsErasing(top_n=0.15)
        self.ratio = 0.15

    def transform(self, img, x, ratio):
        img = self.resultsErasing(img, x, ratio)
        return img

    def forward(self, x, ratio):
        x1 = self.net1(x)

        if ratio == [0]:
            x2 = None
        else:
            x2 = self.transform(x1, x, ratio)

            # x2 = self.unet(x2)
            x2 = self.net2(x2)

        return x1, x2

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
