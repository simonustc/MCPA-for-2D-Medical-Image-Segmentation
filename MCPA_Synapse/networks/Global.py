import torch.nn as nn
import torch
import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange
from .Former import *
import math


class M_EfficientSelfAtten_3(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.num_heads = head
        #self.reduction_ratio = reduction_ratio  # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)

        self.attn_drop=nn.Dropout(0)
        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(0)

        self.act=nn.GELU()

        self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
        self.norm1 = nn.LayerNorm(dim)
        self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
        self.norm2 = nn.LayerNorm(dim)

        self.sr3 = nn.Conv2d(128, 128, kernel_size=4, stride=4)
        self.sr4 = nn.Conv2d(128, 128, kernel_size=2, stride=2)

        self.sr5 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.sr6 = nn.Conv2d(256, 256, kernel_size=1, stride=1)


        self.kv1 = nn.Linear(dim, dim*2, bias=True)
        self.kv2 = nn.Linear(dim, dim, bias=True)
        self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
        self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)


        # self.kv = nn.Linear(dim, dim * 2, bias=True)
        # self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

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


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        tem0 = x[:, :392, :]  # 56
        tem1 = x[:, 392:3528, :].reshape(B, 56, 56, C ).permute(0, 3, 1, 2)  # 28
        tem2 = x[:, 3528:5096, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)  # 14
        tem3 = x[:, 5096:5880, :].reshape(B, 14, 14, C*4).permute(0, 3, 1, 2)  # 7

        #sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
        #x_1 = self.act(self.norm1(self.sr1(tem0).reshape(B, C, -1).permute(0, 2, 1)))

        x_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
        #x_2 = self.sr2(tem0).reshape(B, C, -1).permute(0, 2, 1)

        x_3 = self.sr3(tem2).reshape(B, C, -1).permute(0, 2, 1)
        #x_4 = self.sr4(tem1).reshape(B, C, -1).permute(0, 2, 1)

        x_5 = self.sr5(tem3).reshape(B, C, -1).permute(0, 2, 1)
        #x_6 = self.sr6(tem2).reshape(B, C, -1).permute(0, 2, 1)

        x1_=torch.cat([tem0,x_1,x_3,x_5],dim=1)
        x1_=self.act(self.norm1(x1_))
        #print(x1_.shape)

        # x2_=torch.cat([x_2,x_4,x_6,tem3],dim=1)
        # x2_=self.act(self.norm2(x2_))

        kv1 = self.kv1(x1_).reshape(B, -1, 2, 1 , 64).permute(2, 0, 3, 1, 4)
        #kv2 = self.kv2(x2_).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]  # B head N C
        #k2, v2 = kv2[0], kv2[1]
        attn1 = (q @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        #v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C )
        # attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
        # attn2 = attn2.softmax(dim=-1)
        # attn2 = self.attn_drop(attn2)
        #v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        #x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)
        #x1 = torch.cat([x1, x1], dim=-1)

        x = self.proj(x1)
        out = self.proj_drop(x)

        return out


class BridgeLayer_3(nn.Module):
    def __init__(self, dims, head):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten_3(dims, head)
        self.norm2 = nn.LayerNorm(dims)
        self.drop_path=nn.Identity()
        # self.mixffn1 = MixFFN_skip(dims, dims * 4)
        # self.mixffn2 = MixFFN_skip(dims * 2, dims * 8)
        # self.mixffn3 = MixFFN_skip(dims * 5, dims * 20)
        # self.mixffn4 = MixFFN_skip(dims * 8, dims * 32)
        self.mlp1=Mlp(in_features=dims, hidden_features=dims*4, act_layer=nn.GELU(), drop=nn.Dropout())
        self.mlp2 = Mlp(in_features=2*dims, hidden_features=dims * 8, act_layer=nn.GELU(), drop=nn.Dropout())
        self.mlp3 = Mlp(in_features=4 * dims, hidden_features=dims * 16, act_layer=nn.GELU(), drop=nn.Dropout())
        self.mlp4 = Mlp(in_features=8 * dims, hidden_features=dims * 32, act_layer=nn.GELU(), drop=nn.Dropout())
        self.apply(self._init_weights)

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
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        tx = self.norm2(tx1)

        tem1 = tx[:, :392, :].reshape(B, -1, C*8) #7
        tem2 = tx[:, 392:3528, :].reshape(B, -1, C ) #56
        tem3 = tx[:, 3528:5096, :].reshape(B, -1, C * 2) #28
        tem4 = tx[:, 5096:5880, :].reshape(B, -1, C * 4) #14

        m1f = self.mlp1(tem2, 56, 56).reshape(B, -1, C)
        m2f = self.mlp2(tem3, 28, 28).reshape(B, -1, C)
        m3f = self.mlp3(tem4, 14, 14).reshape(B, -1, C)
        m4f = self.mlp4(tem1, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m3f, m4f, m1f, m2f], -2)
        t1=self.drop_path(t1)
        tx2 = tx1 + t1
        return tx2


class M_EfficientSelfAtten_2(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.num_heads = head
        # self.reduction_ratio = reduction_ratio  # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0)

        self.act = nn.GELU()

        self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
        self.norm1 = nn.LayerNorm(dim)
        self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
        self.norm2 = nn.LayerNorm(dim)

        self.sr3 = nn.Conv2d(128, 128, kernel_size=4, stride=4)
        self.sr4 = nn.Conv2d(128, 128, kernel_size=2, stride=2)

        self.sr5 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.sr6 = nn.Conv2d(256, 256, kernel_size=1, stride=1)

        self.kv1 = nn.Linear(dim, dim * 2, bias=True)
        self.kv2 = nn.Linear(dim, dim, bias=True)
        self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
        self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)

        # self.kv = nn.Linear(dim, dim * 2, bias=True)
        # self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        tem0 = x[:, :784, :].reshape(B, 14, 14, C * 4).permute(0, 3, 1, 2)  # 14
        tem1 = x[:, 784:1176, :]  # 7
        tem2 = x[:, 1176:4312, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2) # 56
        tem3 = x[:, 4312:5880, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)  # 28

        # sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
        # x_1 = self.act(self.norm1(self.sr1(tem0).reshape(B, C, -1).permute(0, 2, 1)))

        x_1 = self.sr1(tem2).reshape(B, C, -1).permute(0, 2, 1)
        # x_2 = self.sr2(tem0).reshape(B, C, -1).permute(0, 2, 1)

        x_3 = self.sr3(tem3).reshape(B, C, -1).permute(0, 2, 1)
        # x_4 = self.sr4(tem1).reshape(B, C, -1).permute(0, 2, 1)

        x_5 = self.sr5(tem0).reshape(B, C, -1).permute(0, 2, 1)
        # x_6 = self.sr6(tem2).reshape(B, C, -1).permute(0, 2, 1)

        x1_ = torch.cat([x_5, tem1, x_1, x_3], dim=1)
        x1_ = self.act(self.norm1(x1_))
        # print(x1_.shape)

        # x2_=torch.cat([x_2,x_4,x_6,tem3],dim=1)
        # x2_=self.act(self.norm2(x2_))

        kv1 = self.kv1(x1_).reshape(B, -1, 2, 1, 64).permute(2, 0, 3, 1, 4)
        # kv2 = self.kv2(x2_).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]  # B head N C
        # k2, v2 = kv2[0], kv2[1]
        attn1 = (q @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
        # attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
        # attn2 = attn2.softmax(dim=-1)
        # attn2 = self.attn_drop(attn2)
        # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        # x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)
        # x1 = torch.cat([x1, x1], dim=-1)

        x = self.proj(x1)
        out = self.proj_drop(x)

        return out


class BridgeLayer_2(nn.Module):
    def __init__(self, dims, head):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten_2(dims, head)
        self.norm2 = nn.LayerNorm(dims)
        self.drop_path = nn.Identity()
        # self.mixffn1 = MixFFN_skip(dims, dims * 4)
        # self.mixffn2 = MixFFN_skip(dims * 2, dims * 8)
        # self.mixffn3 = MixFFN_skip(dims * 5, dims * 20)
        # self.mixffn4 = MixFFN_skip(dims * 8, dims * 32)
        self.mlp1 = Mlp(in_features=dims, hidden_features=dims * 4, act_layer=nn.GELU(), drop=nn.Dropout())
        self.mlp2 = Mlp(in_features=2 * dims, hidden_features=dims * 8, act_layer=nn.GELU(), drop=nn.Dropout())
        self.mlp3 = Mlp(in_features=4 * dims, hidden_features=dims * 16, act_layer=nn.GELU(), drop=nn.Dropout())
        self.mlp4 = Mlp(in_features=8 * dims, hidden_features=dims * 32, act_layer=nn.GELU(), drop=nn.Dropout())
        self.apply(self._init_weights)

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
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        tx = self.norm2(tx1)

        tem1 = tx[:, :784, :].reshape(B, -1, C * 4)  # 14
        tem2 = tx[:, 784:1176, :].reshape(B, -1, C*8)  # 7
        tem3 = tx[:, 1176:4312, :].reshape(B, -1, C )  # 56
        tem4 = tx[:, 4312:5880, :].reshape(B, -1, C * 2)  # 28

        m1f = self.mlp1(tem3, 56, 56).reshape(B, -1, C)
        m2f = self.mlp2(tem4, 28, 28).reshape(B, -1, C)
        m3f = self.mlp3(tem1, 14, 14).reshape(B, -1, C)
        m4f = self.mlp4(tem2, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m2f, m3f, m4f, m1f], -2)
        t1 = self.drop_path(t1)
        tx2 = tx1 + t1
        return tx2


class M_EfficientSelfAtten_1(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.num_heads = head
        # self.reduction_ratio = reduction_ratio  # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0)

        self.act = nn.GELU()

        self.sr1 = nn.Conv2d(dim, dim, kernel_size=8, stride=8)
        self.norm1 = nn.LayerNorm(dim)
        self.sr2 = nn.Conv2d(dim, dim, kernel_size=4, stride=4)
        self.norm2 = nn.LayerNorm(dim)

        self.sr3 = nn.Conv2d(128, 128, kernel_size=4, stride=4)
        self.sr4 = nn.Conv2d(128, 128, kernel_size=2, stride=2)

        self.sr5 = nn.Conv2d(256, 256, kernel_size=2, stride=2)
        self.sr6 = nn.Conv2d(256, 256, kernel_size=1, stride=1)

        self.kv1 = nn.Linear(dim, dim * 2, bias=True)
        self.kv2 = nn.Linear(dim, dim, bias=True)
        self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
        self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)

        # self.kv = nn.Linear(dim, dim * 2, bias=True)
        # self.proj = nn.Linear(dim, dim)
        self.apply(self._init_weights)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        tem0 = x[:, :1568, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)  # 28
        tem1 = x[:, 1568:2352, :].reshape(B, 14, 14, C * 4).permute(0, 3, 1, 2)  # 14
        tem2 = x[:, 2352:2744, :] # 7
        tem3 = x[:, 2744:5880, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)  # 56

        # sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
        # x_1 = self.act(self.norm1(self.sr1(tem0).reshape(B, C, -1).permute(0, 2, 1)))

        x_1 = self.sr1(tem3).reshape(B, C, -1).permute(0, 2, 1)
        # x_2 = self.sr2(tem0).reshape(B, C, -1).permute(0, 2, 1)

        x_3 = self.sr3(tem0).reshape(B, C, -1).permute(0, 2, 1)
        # x_4 = self.sr4(tem1).reshape(B, C, -1).permute(0, 2, 1)

        x_5 = self.sr5(tem1).reshape(B, C, -1).permute(0, 2, 1)
        # x_6 = self.sr6(tem2).reshape(B, C, -1).permute(0, 2, 1)

        x1_ = torch.cat([x_3, x_5, tem2, x_1], dim=1)
        x1_ = self.act(self.norm1(x1_))
        # print(x1_.shape)

        # x2_=torch.cat([x_2,x_4,x_6,tem3],dim=1)
        # x2_=self.act(self.norm2(x2_))

        kv1 = self.kv1(x1_).reshape(B, -1, 2, 1, 64).permute(2, 0, 3, 1, 4)
        # kv2 = self.kv2(x2_).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]  # B head N C
        # k2, v2 = kv2[0], kv2[1]
        attn1 = (q @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
        # attn2 = (q[:, self.num_heads // 2:] @ k2.transpose(-2, -1)) * self.scale
        # attn2 = attn2.softmax(dim=-1)
        # attn2 = self.attn_drop(attn2)
        # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        # x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, C // 2)
        # x1 = torch.cat([x1, x1], dim=-1)

        x = self.proj(x1)
        out = self.proj_drop(x)

        return out



class BridgeLayer_1(nn.Module):
    def __init__(self, dims, head):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten_2(dims, head)
        self.norm2 = nn.LayerNorm(dims)
        self.drop_path = nn.Identity()
        # self.mixffn1 = MixFFN_skip(dims, dims * 4)
        # self.mixffn2 = MixFFN_skip(dims * 2, dims * 8)
        # self.mixffn3 = MixFFN_skip(dims * 5, dims * 20)
        # self.mixffn4 = MixFFN_skip(dims * 8, dims * 32)
        self.mlp1 = Mlp(in_features=dims, hidden_features=dims * 4, act_layer=nn.GELU(), drop=nn.Dropout())
        self.mlp2 = Mlp(in_features=2 * dims, hidden_features=dims * 8, act_layer=nn.GELU(), drop=nn.Dropout())
        self.mlp3 = Mlp(in_features=4 * dims, hidden_features=dims * 16, act_layer=nn.GELU(), drop=nn.Dropout())
        self.mlp4 = Mlp(in_features=8 * dims, hidden_features=dims * 32, act_layer=nn.GELU(), drop=nn.Dropout())
        self.apply(self._init_weights)

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
        B = inputs[0].shape[0]
        C = 64
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4 = inputs
            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            # print(c1f.shape, c2f.shape, c3f.shape, c4f.shape)
            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)
        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        tx = self.norm2(tx1)

        tem1 = tx[:, :1568, :].reshape(B, -1, C * 2)  # 28
        tem2 = tx[:, 1568:2352, :].reshape(B, -1, C*4)  # 14
        tem3 = tx[:, 2352:2744, :].reshape(B, -1, C*8)  # 7
        tem4 = tx[:, 2744:5880, :].reshape(B, -1, C)  # 56

        m1f = self.mlp1(tem4, 56, 56).reshape(B, -1, C)
        m2f = self.mlp2(tem1, 28, 28).reshape(B, -1, C)
        m3f = self.mlp3(tem2, 14, 14).reshape(B, -1, C)
        m4f = self.mlp4(tem3, 7, 7).reshape(B, -1, C)

        t1 = torch.cat([m1f, m2f, m3f, m4f], -2)
        t1 = self.drop_path(t1)
        tx2 = tx1 + t1
        return tx2