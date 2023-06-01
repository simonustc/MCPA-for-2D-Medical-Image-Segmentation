import torch.nn as nn
import torch
import torch
import torch.nn as nn
from typing import Tuple
from einops import rearrange
from .Former import *
import math
from .Global import *
from .CP_attention import *


class M_EfficientSelfAtten(nn.Module):
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

        self.kv1 = nn.Linear(dim, dim, bias=True)
        self.kv2 = nn.Linear(dim, dim, bias=True)
        # self.local_conv1 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
        # self.local_conv2 = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, stride=1, groups=dim // 2)
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
        q = self.q(x).reshape(B, N, 2, 32).permute(0, 2, 1, 3)

        tem0 = x[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
        tem1 = x[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
        tem2 = x[:, 4704:5488, :].reshape(B, 14, 14, C * 4).permute(0, 3, 1, 2)
        tem3 = x[:, 5488:5880, :]

        x_1 = self.sr1(tem0).reshape(B, C, -1).permute(0, 2, 1)  # 49 64
        x_2 = self.sr2(tem0).reshape(B, C, -1).permute(0, 2, 1)  # 196 64

        x_3 = self.sr3(tem1).reshape(B, C, -1).permute(0, 2, 1)
        x_4 = self.sr4(tem1).reshape(B, C, -1).permute(0, 2, 1)

        x_5 = self.sr5(tem2).reshape(B, C, -1).permute(0, 2, 1)
        x_6 = self.sr6(tem2).reshape(B, C, -1).permute(0, 2, 1)

        x1_ = torch.cat([x_1, x_3, x_5, tem3], dim=1)
        x1_ = self.act(self.norm1(x1_))
        # print(x1_.shape)

        x2_ = torch.cat([x_2, x_4, x_6, tem3], dim=1)
        x2_ = self.act(self.norm2(x2_))

        kv1 = self.kv1(x1_).reshape(B, -1, 2, 1, 32).permute(2, 0, 3, 1, 4)
        kv2 = self.kv2(x2_).reshape(B, -1, 2, 1, 32).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]  # B head N C
        k2, v2 = kv2[0], kv2[1]
        attn1 = (q[:, :1] @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, 32)
        attn2 = (q[:, 1:] @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, 32)
        x1 = torch.cat([x1, x2], dim=-1)
        x = self.proj(x1)
        out = self.proj_drop(x)

        return out


class MS_Perceptron(nn.Module):
    def __init__(self, dims, head, img_size):
        super(MS_Perceptron, self).__init__()
        self.scale = 2
        factor = 2

        # c2: 28 28 128
        self.trans_local1 = cross_perceptron1(in_channels=256 // factor // self.scale * 2,
                                                      out_channels=256 // factor // self.scale * 2,
                                                      imagesize=img_size // 8, depth=1, heads=4, patch_size=1)

        # c3: 14 14 256
        self.trans_local2 = cross_perceptron2(in_channels=256 // factor // self.scale * 4,
                                                      out_channels=256 // factor // self.scale * 4,
                                                      imagesize=img_size // 16, depth=1, heads=8, patch_size=1)

        # c4 7 7 512
        self.trans_local3 = cross_perceptron3(in_channels=256 // factor // self.scale * 8,
                                                      out_channels=256 // factor // self.scale * 8,
                                                      imagesize=img_size // 32, depth=1, heads=16, patch_size=1)

        # c1 56 56 64
        self.trans_local4 = Transformer_block_global4(in_channels=256 // factor // self.scale,
                                                      out_channels=256 // factor // self.scale * 2,
                                                      imagesize=img_size // 4, depth=1, heads=4, patch_size=1)

        # c3: 14 14 256
        self.trans_local5 = Transformer_block_global5(in_channels=256 // factor // self.scale,
                                                      out_channels=256 // factor // self.scale * 2,
                                                      imagesize=img_size // 4, depth=1, heads=4, patch_size=1)

        # 需要修改
        self.global_aware2 = cross_perceptron4(dim=64, heads=2, dropout=0, imagesize=img_size)

    def forward(self, x: list):
        # [x1,x2,x3,x4]
        outs = []
        x1, x2, x3, x4 = x

        # 将输入改成list型 9.14

        x2_ = self.trans_local1(x1, x2)  # c1 kv c2 q
        x3_ = self.trans_local2(x1, x2_, x3)  # c1 c2 kv c3 q
        x4_ = self.trans_local3(x1, x2_, x3_, x4)  # c1 c2 c3 kv c4 q

        # x1 x2_ x3_ x4_  2 bridges 可以去掉
        # x1_ = self.trans_local4(x2_, x1)  # c1 q c2 kv
        # x3_ = self.trans_local5(x4_, x3_) # c3 q c4 kv

        # global aware 1
        x3_, x4_ = self.global_aware2([x1, x2_, x3_, x4_])

        outs.append(x1)
        outs.append(x2_)
        outs.append(x3_)
        outs.append(x4_)

        return outs


class CAT(nn.Module):
    def __init__(self, dims, resolution):
        super().__init__()

        self.fc1 = nn.Linear(dims // 2, dims // resolution)
        self.fc2 = nn.Linear(dims // resolution, dims)
        self.fc3 = nn.Linear(dims // 2, dims // resolution)
        self.fc4 = nn.Linear(dims // resolution, dims)
        self.gelu = nn.GELU()
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

    def forward(self, inputs1, inputs2):
        B = inputs1.size(0)
        # inputs1_1 = F.avg_pool2d(inputs1, (inputs1.size(2),inputs1.size(3)),stride=(inputs1.size(2),inputs1.size(3)))
        # inputs1_2=F.max_pool2d(inputs1, (inputs1.size(2),inputs1.size(3)),stride=(inputs1.size(2),inputs1.size(3)))
        # inputs1_1=inputs1_1.view(B,-1)
        # inputs1_2 = inputs1_2.view(B, -1)

        inputs1 = inputs1.transpose(1, 2)
        inputs1_1 = torch.max(inputs1, 2)[0].unsqueeze(2).transpose(1, 2)
        inputs1_2 = torch.mean(inputs1, 2).unsqueeze(2).transpose(1, 2)

        inputs1_1 = self.fc1(inputs1_1)
        inputs1_1 = self.fc2(inputs1_1)

        inputs1_2 = self.fc3(inputs1_2)
        inputs1_2 = self.fc4(inputs1_2)

        inputs = inputs1_1 + inputs1_2
        inputs = self.gelu(inputs)
        out = inputs2 * inputs

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class BridgeLayer_4(nn.Module):
    def __init__(self, dims, head):
        super().__init__()

        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten(dims, head)
        self.norm2 = nn.LayerNorm(dims)
        self.drop_path = nn.Identity()

        # self.mlp1 = Mlp(in_features=dims, hidden_features=dims * 4, act_layer=nn.GELU(), drop=nn.Dropout())
        # self.mlp2 = Mlp(in_features=2 * dims, hidden_features=dims * 8, act_layer=nn.GELU(), drop=nn.Dropout())
        # self.mlp3 = Mlp(in_features=4 * dims, hidden_features=dims * 16, act_layer=nn.GELU(), drop=nn.Dropout())
        # self.mlp4 = Mlp(in_features=8 * dims, hidden_features=dims * 32, act_layer=nn.GELU(), drop=nn.Dropout())

        # self.channel_attention1 = CAT(128, 16)
        # self.channel_attention2 = CAT(256, 16)
        # self.channel_attention3 = CAT(512, 16)

        self.feedforward=FeedForward(64, 64*4, dropout=0)

        self.conv1 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))

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

            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)

        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        tx = self.norm2(tx1)

        tx= tx + self.drop_path(self.feedforward(self.norm2(tx)))
        return tx


class Globol_Perceptron(nn.Module):
    def __init__(self, dims, head):
        super().__init__()
        self.bridge_layer4 = BridgeLayer_4(dims, head)
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
        bridge4 = self.bridge_layer4(x)
        # bridge4 = self.bridge_layer4(bridge1)
        # bridge3 = self.bridge_layer4(bridge2)
        # bridge4 = self.bridge_layer4(bridge3)

        B, _, C = bridge4.shape
        outs = []

        sk1 = bridge4[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
        sk2 = bridge4[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
        sk3 = bridge4[:, 4704:5488, :].reshape(B, 14, 14, C * 4).permute(0, 3, 1, 2)
        sk4 = bridge4[:, 5488:5880, :].reshape(B, 7, 7, C * 8).permute(0, 3, 1, 2)

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim0, dim1, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim0 = dim0

        self.expand = nn.ConvTranspose2d(dim0, dim1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                         output_padding=(1, 1))

        self.norm = norm_layer(dim0 // dim_scale)
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
        elif isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        x: B, H*W, C
        """
        # print("x_shape-----",x.shape)
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.expand(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        # B, L, C = x.shape
        # print(x.shape)
        # assert L == H * W, "input feature has wrong size"

        # x = x.view(B, H, W, C)
        # x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        # x = x.view(B, -1, C // 4)
        x = self.norm(x.clone())

        return x, H, W


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 9, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1))

        self.output_dim = dim
        self.norm = norm_layer(9)
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
        elif isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = self.expand(x)

        x = self.relu1(x)
        x = self.batch1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch2(x)
        x = self.conv3(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)

        x = self.norm(x.clone())

        return x, H, W


class MCPANET(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[2, 4, 8, 16], mlp_ratios=[8, 8, 4, 4], qkv_bias=True, qk_scale=None, sr_ratios=[8, 4, 2, 1],
                 num_conv=1,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[2, 4, 4, 1], num_stages=4, flag=False):
        super().__init__()

        if flag == False:
            self.num_classes = num_classes

        self.depths = depths
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # self.backbone=VAN(img_size,embed_dims,num_heads,mlp_ratios,drop_rate,drop_path_rate,depths,num_stages,dpr,num_conv,qkv_bias,qk_scale,attn_drop_rate,sr_ratios,norm_layer)

        self.patch_embed1 = Head(num_conv)
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // (2 ** (1 + 1)), patch_size=3, stride=2,
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // (2 ** (1 + 2)), patch_size=3, stride=2,
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // (2 ** (1 + 3)), patch_size=3, stride=2,
                                              in_chans=embed_dims[2], embed_dim=embed_dims[3])

        cur = 0

        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[0])
            for j in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])
        cur += depths[0]

        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[1])
            for j in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])
        cur += depths[1]

        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[2])
            for j in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])
        cur += depths[2]

        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias,
                  qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                  norm_layer=norm_layer, sr_ratio=sr_ratios[3])
            for j in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])
        cur += depths[3]

        # multiscale attention
        self.Guide = MS_Perceptron(64, 1, img_size)

        self.bridge = Globol_Perceptron(dims=64, head=1)

        cur_up = 0

        for i in range(self.num_stages):
            up_block = nn.ModuleList([
                Block(dim=embed_dims[num_stages - i - 1], num_heads=num_heads[num_stages - i - 1],
                      mlp_ratio=mlp_ratios[num_stages - i - 1], qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                      attn_drop=attn_drop_rate,
                      drop_path=dpr[sum(depths) - cur_up - 1 - j], norm_layer=norm_layer,
                      sr_ratio=sr_ratios[num_stages - i - 1])
                for j in range(depths[num_stages - i - 1])])
            up_norm = norm_layer(embed_dims[num_stages - i - 1])
            cur_up += depths[num_stages - i - 1]

            setattr(self, f"up_block{i + 1}", up_block)
            setattr(self, f"up_norm{i + 1}", up_norm)

        self.patch_expand1 = PatchExpand(input_resolution=(7, 7), dim0=512, dim1=256)
        self.up_block2 = getattr(self, f"up_block{2}")
        self.up_norm2 = getattr(self, f"up_norm{2}")

        self.patch_expand2 = PatchExpand(input_resolution=(14, 14), dim0=256, dim1=128)
        self.up_block3 = getattr(self, f"up_block{3}")
        self.up_norm3 = getattr(self, f"up_norm{3}")

        self.patch_expand3 = PatchExpand(input_resolution=(28, 28), dim0=128, dim1=64)
        self.up_block4 = getattr(self, f"up_block{4}")
        self.up_norm4 = getattr(self, f"up_norm{4}")

        self.patch_expand4 = FinalPatchExpand_X4(input_resolution=(56, 56), dim=64, dim_scale=4, norm_layer=norm_layer)

        self.concat_back_dim = nn.ModuleList()
        self.concat_back_dim.append(nn.Linear(512, 256))
        self.concat_back_dim.append(nn.Linear(256, 128))
        self.concat_back_dim.append(nn.Linear(128, 64))

        self.head = nn.Conv2d(embed_dims[0], num_classes, kernel_size=(1, 1), stride=(1, 1))
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

    def forward(self, x):
        # print(x.shape)
        # ---------------Encoder-------------------------
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # encoder = self.backbone(x)
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        # x = x.flatten(2).transpose(1, 2)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        outs = self.Guide(outs)

        bridge = self.bridge(outs)  # list

        # sum=[]
        # sum.append(encoder[0]+bridge[0])
        # sum.append(encoder[1]+bridge[1])
        # sum.append(encoder[2] + bridge[2])
        # sum.append(encoder[3] + bridge[3])

        B, c, _, _ = bridge[3].shape
        # print(bridge[3].shape, bridge[2].shape,bridge[1].shape, bridge[0].shape)
        # ---------------Decoder-------------------------
        # print("stage3-----")

        x, H, W = self.patch_expand1(bridge[3].permute(0, 2, 3, 1).view(B, -1, c))
        x2 = bridge[2].view(B, 256, -1).transpose(1, 2)

        cat_x = torch.cat([x, x2], dim=-1)
        x = self.concat_back_dim[0](cat_x)
        # x = x.reshape(B, 14, 14, -1).permute(0, 3, 1, 2).contiguous()
        for blk in self.up_block2:
            x = blk(x, 14, 14)
        # x = x.flatten(2).transpose(1, 2)
        x = self.up_norm2(x)
        x = x.reshape(B, 14, 14, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_expand2(x.permute(0, 2, 3, 1).view(B, -1, 256))
        x3 = bridge[1].view(B, 128, -1).transpose(1, 2)

        cat_x = torch.cat([x, x3], dim=-1)

        x = self.concat_back_dim[1](cat_x)
        # x = x.reshape(B, 28, 28, -1).permute(0, 3, 1, 2).contiguous()
        for blk in self.up_block3:
            x = blk(x, 28, 28)
        # x = x.flatten(2).transpose(1, 2)
        x = self.up_norm3(x)
        x = x.reshape(B, 28, 28, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_expand3(x.permute(0, 2, 3, 1).view(B, -1, 128))
        x4 = bridge[0].view(B, 64, -1).transpose(1, 2)

        cat_x = torch.cat([x, x4], dim=-1)
        x = self.concat_back_dim[2](cat_x)
        # x = x.reshape(B, 56, 56, -1).permute(0, 3, 1, 2).contiguous()
        for blk in self.up_block4:
            x = blk(x, 56, 56)
        # x = x.flatten(2).transpose(1, 2)
        x = self.up_norm4(x)
        x = x.reshape(B, 56, 56, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_expand4(x.permute(0, 2, 3, 1).view(B, -1, 64))

        x = x.transpose(1, 2).reshape(B, 9, 224, 224).contiguous()
        # x = self.head(x)

        return x
