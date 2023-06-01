import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

np.set_printoptions(threshold=1000)
import cv2
import random
from torchvision.ops import roi_align, nms


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNorm3p(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm5 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x5, x4, x3, **kwargs):
        return self.fn(self.norm5(x5), self.norm4(x4), self.norm3(x3), **kwargs)


class Attention_global1(nn.Module):
    def __init__(self, dim, heads=2, dim_head=64, dropout=0.):
        super().__init__()

        inner_dim = dim  # 128

        self.dim = dim
        self.heads = heads

        head_dim = dim // heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, c1, c2):
        B, _, C = c1.shape  # b 3136 64

        # print('start')
        q = self.to_q(c2)  # .reshape(B, N, 2, 64).permute(0, 2, 1, 3)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)  # b 2 784 64

        k = self.to_k(c1)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = self.to_k(c1)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = self.to_out(out)
        return out


class Attention_global3(nn.Module):
    def __init__(self, dim, heads=2, dim_head=128, dropout=0.):
        super().__init__()

        # 不提高维度，保持原型
        inner_dim = dim
        project_out = not (heads == 1 and dim_head == dim)

        self.dim = dim
        self.heads = heads

        head_dim = dim // heads

        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q1 = nn.Linear(dim, inner_dim, bias=False)  # 256
        self.to_k1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_q2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_k2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v2 = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim * 2, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, c1, c2, c3):
        q1 = self.to_q1(c3)  # 196 512
        q1 = rearrange(q1, 'b n (h d) -> b h n d', h=self.heads)  # b 2 196 256
        q2 = self.to_q2(c3)
        q2 = rearrange(q2, 'b n (h d) -> b h n d', h=self.heads)  # b 2 196 256

        k1 = self.to_k1(c2)
        k1 = rearrange(k1, 'b n (h d) -> b h n d', h=self.heads)  # b 2 784 256
        v1 = self.to_v1(c2)
        v1 = rearrange(v1, 'b n (h d) -> b h n d', h=self.heads)  # b 2 784 256

        k2 = self.to_k1(c1)
        k2 = rearrange(k2, 'b n (h d) -> b h n d', h=self.heads)  # b 2 784 256
        v2 = self.to_v1(c1)
        v2 = rearrange(v2, 'b n (h d) -> b h n d', h=self.heads)  # b 2 784 256

        dots1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale
        attn1 = self.attend(dots1)
        out1 = torch.matmul(attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        dots2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
        attn2 = self.attend(dots2)
        out2 = torch.matmul(attn2, v2)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')

        out = torch.cat((out1, out2), dim=-1)  # 196, 1024

        return self.to_out(out)


class Attention_global2(nn.Module):
    def __init__(self, dim, heads=2, dim_head=64, dropout=0.):
        super().__init__()
        # dim=512
        # 不提高维度，保持原型
        inner_dim = dim
        project_out = not (heads == 1 and dim_head == dim)

        self.dim = dim
        self.heads = heads

        head_dim = dim // heads

        self.scale = head_dim ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_q1 = nn.Linear(dim, inner_dim, bias=False)  # 512
        self.to_k1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v1 = nn.Linear(dim, inner_dim, bias=False)
        self.to_q2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_k2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_q3 = nn.Linear(dim, inner_dim, bias=False)
        self.to_k3 = nn.Linear(dim, inner_dim, bias=False)
        self.to_v3 = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim * 3, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, c1, c2, c3, c4):
        q1 = self.to_q1(c4)  # 49 1024
        q1 = rearrange(q1, 'b n (h d) -> b h n d', h=self.heads)  # b 2 49 256
        q2 = self.to_q2(c4)
        q2 = rearrange(q2, 'b n (h d) -> b h n d', h=self.heads)  # b 2 49 256
        q3 = self.to_q3(c4)
        q3 = rearrange(q3, 'b n (h d) -> b h n d', h=self.heads)  # b 2 49 256

        k1 = self.to_k1(c1)  # 49 1024
        k1 = rearrange(k1, 'b n (h d) -> b h n d', h=self.heads)  # b 2 49 256
        v1 = self.to_v1(c1)
        v1 = rearrange(v1, 'b n (h d) -> b h n d', h=self.heads)

        k2 = self.to_k2(c2)
        k2 = rearrange(k2, 'b n (h d) -> b h n d', h=self.heads)
        v2 = self.to_v2(c2)
        v2 = rearrange(v2, 'b n (h d) -> b h n d', h=self.heads)

        k3 = self.to_k3(c3)
        k3 = rearrange(k3, 'b n (h d) -> b h n d', h=self.heads)
        v3 = self.to_v3(c3)
        v3 = rearrange(v3, 'b n (h d) -> b h n d', h=self.heads)

        dots1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scale
        attn1 = self.attend(dots1)
        out1 = torch.matmul(attn1, v1)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')

        dots2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scale
        attn2 = self.attend(dots2)
        out2 = torch.matmul(attn2, v2)
        out2 = rearrange(out2, 'b h n d -> b n (h d)')

        dots3 = torch.matmul(q3, k3.transpose(-1, -2)) * self.scale
        attn3 = self.attend(dots3)
        out3 = torch.matmul(attn3, v3)
        out3 = rearrange(out3, 'b h n d -> b n (h d)')

        out = torch.cat((out1, out2, out3), dim=-1)

        return self.to_out(out)


class Attention_global4(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64, dropout=0.):
        super().__init__()
        # dim=512
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(64, 64, bias=False)  #
        self.to_k = nn.Linear(64, 64, bias=False)
        self.to_v = nn.Linear(64, 64, bias=False)

        #
        self.to_out = nn.Sequential(
            nn.Linear(64, 64),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, c2, c1):
        q = self.to_q(c1)  # 64 3136
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)  # b 4 3136 16

        k = self.to_k(c2)  # 64 784
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)  # b 4 49 16
        v = self.to_v(c2)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        #

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class Attention_global5(nn.Module):
    def __init__(self, dim, heads=16, dim_head=64, dropout=0.):
        super().__init__()
        # dim=512
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(256, 256, bias=False)  # 49 1024
        self.to_k = nn.Linear(256, 256, bias=False)
        self.to_v = nn.Linear(256, 256, bias=False)

        #
        self.to_out = nn.Sequential(
            nn.Linear(256, 256),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, c2, c1):
        q = self.to_q(c1)  # 196 512
        q = rearrange(q, 'b n (h d) -> b h n d', h=14)  # 14 14 512

        k = self.to_k(c2)  # 49 512
        k = rearrange(k, 'b n (h d) -> b h n d', h=7)  # 7 7 512
        v = self.to_v(c2)
        v = rearrange(v, 'b n (h d) -> b h n d', h=7)

        #

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class CPA1(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        # dim=128 depth=1 heads=4 dim_head=128 mlp_dim=512 num_patch=3136
        # 判断dim是否是64
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.Attention = Attention_global1(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.feedforward = FeedForward(dim, mlp_dim, dropout=dropout)
        self.drop_path = nn.Dropout(dropout)

    def forward(self, c1, x):
        # attn + mlp
        x = x + self.drop_path(self.Attention(self.norm1(c1), self.norm2(x)))
        x = x + self.drop_path(self.feedforward(self.norm3(x)))
        return x


class CPA3(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()
        self.layers = nn.ModuleList([])

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)
        self.norm5 = nn.LayerNorm(dim)

        self.Attention = Attention_global2(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.drop_path = nn.Dropout(dropout)
        self.feedforward = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, c1, c2, c3, x):
        x = x + self.drop_path(self.Attention(self.norm1(c1), self.norm2(c2), self.norm3(c3), self.norm4(x)))
        x = x + self.drop_path(self.feedforward(self.norm5(x)))
        return x


class CPA2(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        self.Attention = Attention_global3(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.feedforward = FeedForward(dim, mlp_dim, dropout=dropout)
        self.drop_path = nn.Dropout(dropout)

    def forward(self, x5, x4, x):
        x = x + self.drop_path(self.Attention(self.norm1(x5), self.norm2(x4), self.norm3(x)))
        x = x + self.drop_path(self.feedforward(self.norm4(x)))
        return x


class Transformer_global4(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.Attention = Attention_global4(dim, heads=heads, dim_head=dim_head, dropout=dropout)

        self.norm4 = nn.LayerNorm(dim)
        self.feedforward = FeedForward(dim, mlp_dim, dropout=dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm3p(dim, Attention_global4(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, c2, x):
        x = self.norm1(x)
        x = self.norm2(x)
        x = self.norm3(x)
        x = self.Attention(c2, x) + x
        x = self.norm4(x)
        x = self.feedforward(x) + x
        return x


class Transformer_global5(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim=1024, dropout=0., num_patches=128):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.Attention = Attention_global5(dim, heads=heads, dim_head=dim_head, dropout=dropout)

        self.norm4 = nn.LayerNorm(dim)
        self.feedforward = FeedForward(dim, mlp_dim, dropout=dropout)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm3p(dim, Attention_global4(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, c2, x):
        x = self.norm1(x)
        x = self.norm2(x)
        x = self.norm3(x)
        x = self.Attention(c2, x) + x
        x = self.norm4(x)
        x = self.feedforward(x) + x
        return x


class cross_perceptron1(nn.Module):

    def __init__(self, in_channels, out_channels, imagesize, depth, patch_size, heads, dim_head=128, dropout=0.1,
                 emb_dropout=0.1):
        # in_chan=128 out_chan=128 img=28 patch=1
        super().__init__()
        image_height, image_width = pair(imagesize)  # 28 28
        patch_height, patch_width = pair(patch_size)  # 1 1
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 28 28
        self.patch_dim = in_channels * patch_height * patch_width  # 64
        self.dmodel = out_channels  # 128
        self.mlp_dim = self.dmodel * 4  # 512

        self.to_patch_embedding_c1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2),  # b 784 256
            nn.Linear(in_channels * 2, self.dmodel),  # [ b 784 128] 降维
        )

        self.to_patch_embedding_c2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),  # b 784 128
            nn.Linear(in_channels, self.dmodel),  # [ b 784 128] 不变
        )

        #self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, self.dmodel))
        #self.dropout = nn.Dropout(0.1)

        # dmodel=128 depth=1 heads=4 dim_head=128 mlp_dim=512 num_patch=3136 ? 784
        self.transformer = CPA1(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout,
                                num_patches)

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),  # image_height // patch_height
        )



    def forward(self, c1, c2):
        c1 = self.to_patch_embedding_c1(c1)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        c2 = self.to_patch_embedding_c2(c2)

        #_, n1, _ = c1.shape
        #_, n2, _ = c2.shape

        #c1 += self.pos_embedding[:, :n1]
        #c2 += self.pos_embedding[:, :n2]

        #c1 = self.dropout(c1)
        #c2 = self.dropout(c2)

        # transformer layer
        ax = self.transformer(c1, c2)
        out = self.recover_patch_embedding(ax)
        return out


class cross_perceptron2(nn.Module):

    def __init__(self, in_channels, out_channels, imagesize, depth=1, patch_size=1, heads=2, dim_head=128, dropout=0,
                 emb_dropout=0):
        super().__init__()

        # in=256 out=256 img=14 depth=1 heads=4 patchsize=1
        image_height, image_width = pair(imagesize)  # 14 14
        patch_height, patch_width = pair(patch_size)  # 1 1
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 14 14
        self.patch_dim = in_channels * patch_height * patch_width  # 256
        self.dmodel = out_channels  # 256
        self.mlp_dim = self.dmodel * 4  # 1024

        self.to_patch_embedding_c1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2),  # 784 256 ->256 降维 56 56 64 #3136 64
            nn.Linear(in_channels, self.dmodel),
        )
        self.to_patch_embedding_c2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),  # 784 128 ->256 升维 28 28 128
            nn.Linear(in_channels // 2, self.dmodel),
        )
        self.to_patch_embedding_c3 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            # 196 256 不变 14 14 256
            nn.Linear(in_channels, self.dmodel),
        )
        #self.pos_embedding = nn.Parameter(torch.randn(1, 784, self.dmodel))
        #self.dropout = nn.Dropout(0.1)
        self.transformer = CPA2(self.dmodel, depth, heads, dim_head, self.mlp_dim, dropout,
                                num_patches)  # dim=256

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=image_height // patch_height),  # image_height // patch_height),
        )

    def forward(self, c1, c2, c3):  # 16,8,4
        c1 = self.to_patch_embedding_c1(c1)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        c2 = self.to_patch_embedding_c2(c2)
        c3 = self.to_patch_embedding_c3(c3)

        #_, n1, _ = c1.shape
        #_, n2, _ = c2.shape
        #_, n3, _ = c3.shape

        #c1 += self.pos_embedding[:, :n1]
        #c2 += self.pos_embedding[:, :n2]
        #c3 += self.pos_embedding[:, :n3]

        #c1 = self.dropout(c1)
        #c2 = self.dropout(c2)
        #c3 = self.dropout(c3)

        # transformer layer
        ax = self.transformer(c1, c2, c3)
        out = self.recover_patch_embedding(ax)
        return out


class cross_perceptron3(nn.Module):

    def __init__(self, in_channels, out_channels, imagesize, depth=1, patch_size=1, heads=2, dim_head=128, dropout=0.1,
                 emb_dropout=0.1):
        super().__init__()
        # in= 512 out=512 img=7
        image_height, image_width = pair(imagesize)  # 7 7
        patch_height, patch_width = pair(patch_size)  # 1 1
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)  # 49
        self.patch_dim = in_channels * patch_height * patch_width  # 512
        self.dmodel = out_channels  # 512
        self.mlp_dim = self.dmodel * 4  # ?

        self.to_patch_embedding_c1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=4, p2=4),  # 56 64  -> 196 1024
            nn.Linear(in_channels * 2, self.dmodel),  # 196 512
        )
        self.to_patch_embedding_c2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=2, p2=2),  # 28 128 -> 196 512
            nn.Linear(in_channels, self.dmodel),
        )
        self.to_patch_embedding_c3 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),  # 14 256 -> 196 256
            nn.Linear(in_channels // 2, self.dmodel),
        )

        self.to_patch_embedding_c4 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),  # 49 512
            nn.Linear(in_channels, self.dmodel),
        )

        self.transformer = CPA3(512, depth, heads, dim_head, self.mlp_dim, dropout,
                                num_patches)  # dim=256

        #self.pos_embedding = nn.Parameter(torch.randn(1, 196, self.dmodel))
        #self.dropout = nn.Dropout(0.1)
        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=7),
        )

    def forward(self, c1, c2, c3, c4):  # 16,8,4
        c1 = self.to_patch_embedding_c1(c1)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        c2 = self.to_patch_embedding_c2(c2)
        c3 = self.to_patch_embedding_c3(c3)
        c4 = self.to_patch_embedding_c4(c4)

        #_, n1, _ = c1.shape
        #_, n2, _ = c2.shape
        #_, n3, _ = c3.shape
        #_, n4, _ = c4.shape

        #c1 += self.pos_embedding[:, :n1]
        #c2 += self.pos_embedding[:, :n2]
        #c3 += self.pos_embedding[:, :n3]
        #c4 += self.pos_embedding[:, :n4]

        #c1 = self.dropout(c1)
        #c2 = self.dropout(c2)
        #c3 = self.dropout(c3)
        #c4 = self.dropout(c4)


        # transformer layer
        ax = self.transformer(c1, c2, c3, c4)
        out = self.recover_patch_embedding(ax)
        return out


class Transformer_block_global4(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, imagesize, depth=2, patch_size=2, heads=6, dim_head=128, dropout=0.1,
                 emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(imagesize)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel * 4

        self.to_patch_embedding_c1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),  # 64 56 56-> 3136 64
            nn.Linear(64, 64),
        )
        self.to_patch_embedding_c2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),  # 128 28 28-> 784 128
            nn.Linear(128, 64),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 3136, 64))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_global4(64, depth, heads, dim_head, self.mlp_dim, dropout,
                                               num_patches)  # dim=256

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=56),
        )

    def forward(self, c2, c1):  # 16,8,4
        c1 = self.to_patch_embedding_c1(c1)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        c2 = self.to_patch_embedding_c2(c2)

        #_, n1, _ = c1.shape
        #_, n2, _ = c2.shape

        #c1 += self.pos_embedding[:, :n1]
        #c2 += self.pos_embedding[:, :n2]

        #c1 = self.dropout(c1)
        #c2 = self.dropout(c2)

        # transformer layer
        ax = self.transformer(c2, c1)
        out = self.recover_patch_embedding(ax)
        return out


class Transformer_block_global5(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, imagesize, depth=2, patch_size=2, heads=6, dim_head=128, dropout=0.1,
                 emb_dropout=0.1):
        super().__init__()
        image_height, image_width = pair(imagesize)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = in_channels * patch_height * patch_width
        self.dmodel = out_channels
        self.mlp_dim = self.dmodel * 4

        self.to_patch_embedding_c1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),  # 196 256
            nn.Linear(256, 256),
        )
        self.to_patch_embedding_c2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),  # 49 512 -> 49 256
            nn.Linear(512, 256),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, 196, 64))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer_global5(256, depth, heads, dim_head, self.mlp_dim, dropout,
                                               num_patches)  # dim=256

        self.recover_patch_embedding = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=14),
        )

    def forward(self, c2, c1):  # 16,8,4
        c1 = self.to_patch_embedding_c3(c1)  # (b, n, h, w) -> (b, num_patches, dim_patches)
        c2 = self.to_patch_embedding_c4(c2)

        _, n1, _ = c1.shape
        _, n2, _ = c2.shape

        c1 += self.pos_embedding[:, :n1]
        c2 += self.pos_embedding[:, :n2]

        #c1 = self.dropout(c1)
        #c2 = self.dropout(c2)

        # transformer layer
        ax = self.transformer(c2, c1)
        out = self.recover_patch_embedding(ax)
        return out


class cross_perceptron4(nn.Module):
    def __init__(self, dim, heads=2, dropout=0., imagesize=224):
        super().__init__()
        # dim=64

        self.img = imagesize
        inner_dim = dim
        self.dim = dim
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5

        self.attn_drop = nn.Dropout(0)
        self.act = nn.GELU()

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)  # 64
        self.kv1 = nn.Linear(dim, inner_dim, bias=False)
        self.kv2 = nn.Linear(dim, inner_dim, bias=False)

        # conv
        self.sr1 = nn.Conv2d(dim, inner_dim, kernel_size=8, stride=8)
        self.norm1 = nn.LayerNorm(dim)
        self.sr2 = nn.Conv2d(dim, inner_dim, kernel_size=4, stride=4)
        self.norm2 = nn.LayerNorm(dim)

        self.sr3 = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=4)
        self.norm3 = nn.LayerNorm(dim)
        self.sr4 = nn.Conv2d(dim * 2, dim * 2, kernel_size=2, stride=2)
        self.norm4 = nn.LayerNorm(dim)

        self.drop_path=nn.Dropout(0)
        self.feedforward = FeedForward(dim, hidden_dim=1024, dropout=dropout)

        self.apply(self._init_weights)
        #
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, inputs):
        if type(inputs) == list:
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape  # B,64

            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 784 64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392 64
            c34 = torch.cat([c3f, c4f], -2)  # 1176 64 如果保证原来尺度 c4特征图过小 可以改为 C=128 or 256

            q = self.to_q(c34)  # 1176 64
            q = q.reshape(B, -1, 2, 32).permute(0, 2, 1, 3)  # B 32 1176 2

            x_1 = self.sr1(c1).reshape(B, C, -1).permute(0, 2, 1)  # 49 64
            x_2 = self.sr2(c1).reshape(B, C, -1).permute(0, 2, 1)  # 196 64

            x_3 = self.sr3(c2).reshape(B, C, -1).permute(0, 2, 1)  # 98 64
            x_4 = self.sr4(c2).reshape(B, C, -1).permute(0, 2, 1)  # 392 64

            x1_ = torch.cat([x_1, x_3], dim=-2)  # 147 64
            x1_ = self.act(self.norm1(x1_))

            x2_ = torch.cat([x_2, x_4], dim=-2)  # 588 64
            x2_ = self.act(self.norm1(x2_))

            kv1 = self.kv1(x1_).reshape(B, -1, 2, 1, 32).permute(2, 0, 3, 1, 4)  # 2 b 1 147 32
            kv2 = self.kv2(x2_).reshape(B, -1, 2, 1, 32).permute(2, 0, 3, 1, 4)  # 2 b 1 588 32
            k1, v1 = kv1[0], kv1[1]
            k2, v2 = kv2[0], kv2[1]

            dots1 = (q[:,:1] @ k1.transpose(-2, -1)) * self.scale
            attn1 = self.attend(dots1)
            attn1 = self.attn_drop(attn1)
            attn1 = (attn1 @ v1).transpose(1, 2).reshape(B, -1, 32)

            dots2 = (q[:,1:] @ k2.transpose(-2, -1)) * self.scale
            attn2 = self.attend(dots2)
            attn2 = self.attn_drop(attn2)
            attn2 = (attn2 @ v2).transpose(1, 2).reshape(B, -1, 32)

            out = torch.cat([attn1, attn2], dim=-1)
            out = self.to_out(out)  # b 1176 64


            # x = x + self.drop_path(self.feedforward(self.norm3(x)))
            out = out + self.drop_path(self.feedforward(self.norm3(out)))



            x3_ = out[:, :784, :].reshape(B, 14, 14, C * 4).permute(0, 3, 1, 2)
            x4_ = out[:, 784:1176, :].reshape(B, 7, 7, C * 8).permute(0, 3, 1, 2)
            return x3_,x4_



        else:
            assert type(inputs) != list, 'input must be list!'
