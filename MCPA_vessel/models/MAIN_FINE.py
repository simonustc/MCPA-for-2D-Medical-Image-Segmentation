import torch
import torch.nn.functional as F
import torch.nn as nn
from models.MCPA_MODEL.Global import *
from models.MCPA_MODEL.CP_attention import *

drop = 0.25


class M_EfficientSelfAtten2(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.num_heads = head
        self.dim = dim
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

        self.sr3 = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=4)
        self.sr4 = nn.Conv2d(dim * 2, dim * 2, kernel_size=2, stride=2)

        self.sr5 = nn.Conv2d(dim * 4, dim * 4, kernel_size=2, stride=2)
        self.sr6 = nn.Conv2d(dim * 4, dim * 4, kernel_size=1, stride=1)

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
        q = self.q(x).reshape(B, N, 2, self.dim // 2).permute(0, 2, 1, 3)

        r = N // 15
        h = int(math.sqrt(8 * r))
        tem0 = x[:, :8 * r, :].reshape(B, h, h, C).permute(0, 3, 1, 2)
        tem1 = x[:, 8 * r:12 * r, :].reshape(B, h // 2, h // 2, C * 2).permute(0, 3, 1, 2)
        tem2 = x[:, 12 * r:14 * r, :].reshape(B, h // 4, h // 4, C * 4).permute(0, 3, 1, 2)
        tem3 = x[:, 14 * r:15 * r, :]

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

        kv1 = self.kv1(x1_).reshape(B, -1, 2, 1, self.dim // 2).permute(2, 0, 3, 1, 4)
        kv2 = self.kv2(x2_).reshape(B, -1, 2, 1, self.dim // 2).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]  # B head N C
        k2, v2 = kv2[0], kv2[1]
        attn1 = (q[:, :1] @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, self.dim // 2)
        attn2 = (q[:, 1:] @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, self.dim // 2)
        x1 = torch.cat([x1, x2], dim=-1)
        x = self.proj(x1)
        out = self.proj_drop(x)

        return out

class M_EfficientSelfAtten1(nn.Module):
    def __init__(self, dim, head):
        super().__init__()
        self.num_heads = head
        self.dim = dim
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

        self.sr3 = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=4)
        self.sr4 = nn.Conv2d(dim * 2, dim * 2, kernel_size=2, stride=2)

        self.sr5 = nn.Conv2d(dim * 4, dim * 4, kernel_size=2, stride=2)
        self.sr6 = nn.Conv2d(dim * 4, dim * 4, kernel_size=1, stride=1)

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
        q = self.q(x).reshape(B, N, 2, self.dim // 2).permute(0, 2, 1, 3)

        r = N // 7
        h = int(math.sqrt(4 * r))
        tem0 = x[:, :4 * r, :].reshape(B, h, h, C).permute(0, 3, 1, 2)
        tem1 = x[:, 4 * r:6 * r, :].reshape(B, h // 2, h // 2, C * 2).permute(0, 3, 1, 2)
        tem2 = x[:, 6 * r:7 * r, :]
        # tem2 = x[:, 12 * r:14 * r, :].reshape(B, h // 4, h // 4, C * 4).permute(0, 3, 1, 2)
        # tem3 = x[:, 14 * r:15 * r, :]

        x_1 = self.sr1(tem0).reshape(B, C, -1).permute(0, 2, 1)  # 49 64
        x_2 = self.sr2(tem0).reshape(B, C, -1).permute(0, 2, 1)  # 196 64

        x_3 = self.sr3(tem1).reshape(B, C, -1).permute(0, 2, 1)
        x_4 = self.sr4(tem1).reshape(B, C, -1).permute(0, 2, 1)

        # x_5 = self.sr5(tem2).reshape(B, C, -1).permute(0, 2, 1)
        # x_6 = self.sr6(tem2).reshape(B, C, -1).permute(0, 2, 1)

        x1_ = torch.cat([x_1, x_3, tem2], dim=1)
        x1_ = self.act(self.norm1(x1_))
        # print(x1_.shape)

        x2_ = torch.cat([x_2, x_4, tem2], dim=1)
        x2_ = self.act(self.norm2(x2_))

        kv1 = self.kv1(x1_).reshape(B, -1, 2, 1, self.dim // 2).permute(2, 0, 3, 1, 4)
        kv2 = self.kv2(x2_).reshape(B, -1, 2, 1, self.dim // 2).permute(2, 0, 3, 1, 4)
        k1, v1 = kv1[0], kv1[1]  # B head N C
        k2, v2 = kv2[0], kv2[1]
        attn1 = (q[:, :1] @ k1.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        # v1 = v1 + self.local_conv1(v1.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H // self.sr_ratio, W // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, self.dim // 2)
        attn2 = (q[:, 1:] @ k2.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        # v2 = v2 + self.local_conv2(v2.transpose(1, 2).reshape(B, -1, C // 2).transpose(1, 2).view(B, C // 2, H * 2 // self.sr_ratio, W * 2 // self.sr_ratio)).view(B, C // 2, -1).view(B, self.num_heads // 2, C // self.num_heads, -1).transpose(-1, -2)
        x2 = (attn2 @ v2).transpose(1, 2).reshape(B, N, self.dim // 2)
        x1 = torch.cat([x1, x2], dim=-1)
        x = self.proj(x1)
        out = self.proj_drop(x)

        return out


class MS_Perceptron1(nn.Module):
    def __init__(self, dims):
        super(MS_Perceptron1, self).__init__()
        self.scale = 2
        self.dim = dims
        factor = 2

        # c2: 28 28 128
        self.trans_local1 = cross_perceptron1(in_channels=dims * 2, out_channels=dims * 2, depth=1, heads=2,
                                              patch_size=1)

        # c3: 14 14 256
        self.trans_local2 = cross_perceptron2(in_channels=dims * 4, out_channels=dims * 4, depth=1, heads=2,
                                              patch_size=1)

        # self.global_aware2 = Transformer_global_aware(dim=64, heads=2, dropout=0)

    def forward(self, x: list):

        # [x1,x2,x3]
        outs = []
        x1, x2, x3 = x

        x2_ = self.trans_local1(x1, x2)  # c1 kv c2 q
        x3_ = self.trans_local2(x1, x2_, x3)  # c1 c2 kv c3 q

        outs.append(x1)
        outs.append(x2_)
        outs.append(x3_)

        return outs


class MS_Perceptron2(nn.Module):
    def __init__(self,dims):
        super(MS_Perceptron2, self).__init__()
        self.scale = 2
        self.dim=dims
        factor = 2

        # c2: 28 28 128
        self.trans_local1 = cross_perceptron1(in_channels=self.dim * 2, out_channels=self.dim * 2, depth=1, heads=2, patch_size=1)

        # c3: 14 14 256
        self.trans_local2 = cross_perceptron2(in_channels=self.dim * 4, out_channels=self.dim * 4, depth=1, heads=2, patch_size=1)

        # c4 7 7 512
        self.trans_local3 = cross_perceptron3(in_channels=self.dim * 8, out_channels=self.dim * 8, depth=1, heads=2, patch_size=1)

        # self.global_aware2 = Transformer_global_aware(dim=64, heads=2, dropout=0)

    def forward(self, x: list):
        # [x1,x2,x3,x4]
        outs = []
        x1, x2, x3, x4 = x

        x2_ = self.trans_local1(x1, x2)  # c1 kv c2 q
        x3_ = self.trans_local2(x1, x2_, x3)  # c1 c2 kv c3 q
        x4_ = self.trans_local3(x1, x2_, x3_, x4)  # c1 c2 c3 kv c4 q

        # global aware 1
        # x3_, x4_ = self.global_aware2([x1, x2_, x3_, x4_])

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

class BridgeLayer_4_1(nn.Module):
    def __init__(self, dims, head):
        super().__init__()
        self.dim = dims
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten1(dims, head)
        self.norm2 = nn.LayerNorm(dims)
        self.drop_path = nn.Identity()

        # self.mlp1 = Mlp(in_features=dims, hidden_features=dims * 4, act_layer=nn.GELU(), drop=nn.Dropout())
        # self.mlp2 = Mlp(in_features=2 * dims, hidden_features=dims * 8, act_layer=nn.GELU(), drop=nn.Dropout())
        # self.mlp3 = Mlp(in_features=4 * dims, hidden_features=dims * 16, act_layer=nn.GELU(), drop=nn.Dropout())
        # self.mlp4 = Mlp(in_features=8 * dims, hidden_features=dims * 32, act_layer=nn.GELU(), drop=nn.Dropout())

        # self.channel_attention1 = CAT(128, 16)
        # self.channel_attention2 = CAT(256, 16)
        # self.channel_attention3 = CAT(512, 16)

        self.feedforward = FeedForward(dims, dims * 4, dropout=0)

        self.conv1 = nn.Conv2d(dims, dims * 2, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(dims * 2, dims * 4, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(dims * 4, dims * 8, kernel_size=(1, 1), stride=(1, 1))

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
        C = self.dim
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3 = inputs
            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            # c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            # c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            inputs = torch.cat([c1f, c2f, c3f], -2)

            # split to 2 bridges
            # inputs1 =torch.cat([c1f,c2f],-2)
            # inputs2 = torch.cat([c3f, c4f], -2)

        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        tx = self.norm2(tx1)

        tx = tx + self.drop_path(self.feedforward(self.norm2(tx)))

        # 2 bridges
        # bri1 = inputs1 + self.drop_path(self.attn(self.norm1(inputs1)))
        # bri1 = self.norm2(bri1)
        # bri2 = inputs1 + self.drop_path(self.attn(self.norm1(inputs2)))
        # bri2 = self.norm2(bri2)

        # tem1 = tx[:, :3136, :].reshape(B, -1, C)
        # tem2 = tx[:, 3136:4704, :].reshape(B, -1, C * 2)
        # tem3 = tx[:, 4704:5488, :].reshape(B, -1, C * 4)
        # tem4 = tx[:, 5488:5880, :].reshape(B, -1, C * 8)

        # tem1_1 = tem1.transpose(1, 2).view(B, C, 56, 56)
        # tem2_1 = tem2.transpose(1, 2).view(B, 2*C, 28, 28)
        # tem3_1 = tem3.transpose(1, 2).view(B, 4*C, 14, 14)
        # tem4_1 = tem4.transpose(1, 2).view(B, 8*C, 7, 7)

        # tem1_1 = self.conv1(tem1_1)
        # tem2 = self.channel_attention1(tem1, tem2)
        # tem2_2 = tem2.permute(0, 2, 3, 1).reshape(B, -1, C*2)

        # tem2_1 = self.conv2(tem2)
        # tem3 = self.channel_attention2(tem2, tem3)
        # tem3_2 = tem3.permute(0, 2, 3, 1).reshape(B, -1, C * 4)

        # tem3_1=self.conv3(tem3)
        # tem4 = self.channel_attention3(tem3, tem4)
        # tem4_2 = tem4.permute(0, 2, 3, 1).reshape(B, -1, C * 8)

        # m1f = self.mlp1(tem1, 56, 56).reshape(B, -1, C)
        # m2f = self.mlp2(tem2, 28, 28).reshape(B, -1, C)
        # m3f = self.mlp3(tem3, 14, 14).reshape(B, -1, C)
        # m4f = self.mlp4(tem4, 7, 7).reshape(B, -1, C)

        # t1 = torch.cat([m4f, m1f, m2f, m3f], -2)
        # t1 = self.drop_path(t1)
        #
        # tx2 = tx1 + t1

        return tx

class BridgeLayer_4_2(nn.Module):
    def __init__(self, dims, head):
        super().__init__()
        self.dim = dims
        self.norm1 = nn.LayerNorm(dims)
        self.attn = M_EfficientSelfAtten2(dims, head)
        self.norm2 = nn.LayerNorm(dims)
        self.drop_path = nn.Identity()

        # self.mlp1 = Mlp(in_features=dims, hidden_features=dims * 4, act_layer=nn.GELU(), drop=nn.Dropout())
        # self.mlp2 = Mlp(in_features=2 * dims, hidden_features=dims * 8, act_layer=nn.GELU(), drop=nn.Dropout())
        # self.mlp3 = Mlp(in_features=4 * dims, hidden_features=dims * 16, act_layer=nn.GELU(), drop=nn.Dropout())
        # self.mlp4 = Mlp(in_features=8 * dims, hidden_features=dims * 32, act_layer=nn.GELU(), drop=nn.Dropout())

        # self.channel_attention1 = CAT(128, 16)
        # self.channel_attention2 = CAT(256, 16)
        # self.channel_attention3 = CAT(512, 16)

        self.feedforward = FeedForward(dims, dims * 4, dropout=0)

        self.conv1 = nn.Conv2d(dims, dims * 2, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(dims * 2, dims * 4, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(dims * 4, dims * 8, kernel_size=(1, 1), stride=(1, 1))

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
        C = self.dim
        if (type(inputs) == list):
            # print("-----1-----")
            c1, c2, c3, c4= inputs
            B, C, _, _ = c1.shape
            c1f = c1.permute(0, 2, 3, 1).reshape(B, -1, C)  # 3136*64
            c2f = c2.permute(0, 2, 3, 1).reshape(B, -1, C)  # 1568*64
            c3f = c3.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 980*64
            # c4f = c4.permute(0, 2, 3, 1).reshape(B, -1, C)  # 392*64

            inputs = torch.cat([c1f, c2f, c3f, c4f], -2)

            # split to 2 bridges
            # inputs1 =torch.cat([c1f,c2f],-2)
            # inputs2 = torch.cat([c3f, c4f], -2)

        else:
            B, _, C = inputs.shape

        tx1 = inputs + self.drop_path(self.attn(self.norm1(inputs)))
        tx = self.norm2(tx1)

        tx = tx + self.drop_path(self.feedforward(self.norm2(tx)))

        # 2 bridges
        # bri1 = inputs1 + self.drop_path(self.attn(self.norm1(inputs1)))
        # bri1 = self.norm2(bri1)
        # bri2 = inputs1 + self.drop_path(self.attn(self.norm1(inputs2)))
        # bri2 = self.norm2(bri2)

        # tem1 = tx[:, :3136, :].reshape(B, -1, C)
        # tem2 = tx[:, 3136:4704, :].reshape(B, -1, C * 2)
        # tem3 = tx[:, 4704:5488, :].reshape(B, -1, C * 4)
        # tem4 = tx[:, 5488:5880, :].reshape(B, -1, C * 8)

        # tem1_1 = tem1.transpose(1, 2).view(B, C, 56, 56)
        # tem2_1 = tem2.transpose(1, 2).view(B, 2*C, 28, 28)
        # tem3_1 = tem3.transpose(1, 2).view(B, 4*C, 14, 14)
        # tem4_1 = tem4.transpose(1, 2).view(B, 8*C, 7, 7)

        # tem1_1 = self.conv1(tem1_1)
        # tem2 = self.channel_attention1(tem1, tem2)
        # tem2_2 = tem2.permute(0, 2, 3, 1).reshape(B, -1, C*2)

        # tem2_1 = self.conv2(tem2)
        # tem3 = self.channel_attention2(tem2, tem3)
        # tem3_2 = tem3.permute(0, 2, 3, 1).reshape(B, -1, C * 4)

        # tem3_1=self.conv3(tem3)
        # tem4 = self.channel_attention3(tem3, tem4)
        # tem4_2 = tem4.permute(0, 2, 3, 1).reshape(B, -1, C * 8)

        # m1f = self.mlp1(tem1, 56, 56).reshape(B, -1, C)
        # m2f = self.mlp2(tem2, 28, 28).reshape(B, -1, C)
        # m3f = self.mlp3(tem3, 14, 14).reshape(B, -1, C)
        # m4f = self.mlp4(tem4, 7, 7).reshape(B, -1, C)

        # t1 = torch.cat([m4f, m1f, m2f, m3f], -2)
        # t1 = self.drop_path(t1)
        #
        # tx2 = tx1 + t1

        return tx


class Global_Perceptron_Main(nn.Module):
    def __init__(self, dims, head):
        super().__init__()
        # self.bridge_layer1 = BridgeLayer_4(dims, head)
        # self.bridge_layer2 = BridgeLayer_4(dims, head)
        # self.bridge_layer3 = BridgeLayer_4(dims, head)
        self.bridge_layer4 = BridgeLayer_4_1(dims, head)
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

        # h=14 w=14
        _, _, h, w = x[2].shape
        B, _, C = bridge4.shape
        outs = []
        r1 = h * 4 * w * 4
        r2 = h * 2 * w * 2 * 2 + r1
        r3 = h * w * 4 + r2
        # r4 = h * w * 8 + r3
        sk1 = bridge4[:, :r1, :].reshape(B, w * 4, h * 4, C).permute(0, 3, 1, 2)
        sk2 = bridge4[:, r1:r2, :].reshape(B, w * 2, h * 2, C * 2).permute(0, 3, 1, 2)
        sk3 = bridge4[:, r2:r3, :].reshape(B, h, w, C * 4).permute(0, 3, 1, 2)
        # sk4 = bridge4[:, r3:r4, :].reshape(B, h, w, C * 8).permute(0, 3, 1, 2)

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        # outs.append(sk4)

        return outs


class Global_Perceptron_Fine(nn.Module):
    def __init__(self, dims, head):
        super().__init__()
        # self.bridge_layer1 = BridgeLayer_4(dims, head)
        # self.bridge_layer2 = BridgeLayer_4(dims, head)
        # self.bridge_layer3 = BridgeLayer_4(dims, head)
        self.bridge_layer4 = BridgeLayer_4_2(dims, head)
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

        # h=14 w=14
        _, _, h, w = x[3].shape
        B, _, C = bridge4.shape
        outs = []
        r1 = h * 8 * w * 8
        r2 = h * 4 * w * 4 * 2 + r1
        r3 = h * 2 * w * 2 * 4 + r2
        r4 = h * w * 8 + r3
        sk1 = bridge4[:, :r1, :].reshape(B, w * 8, h * 8, C).permute(0, 3, 1, 2)
        sk2 = bridge4[:, r1:r2, :].reshape(B, w * 4, h * 4, C * 2).permute(0, 3, 1, 2)
        sk3 = bridge4[:, r2:r3, :].reshape(B, h * 2, w * 2, C * 4).permute(0, 3, 1, 2)
        sk4 = bridge4[:, r3:r4, :].reshape(B, h, w, C * 8).permute(0, 3, 1, 2)

        outs.append(sk1)
        outs.append(sk2)
        outs.append(sk3)
        outs.append(sk4)

        return outs


class PatchExpand(nn.Module):
    def __init__(self, dim0, dim1, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
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
        B, L, C = x.shape
        H, W = pair(int(math.sqrt(L)))
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
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        # self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
        self.relu1 = nn.ReLU()
        self.batch1 = nn.BatchNorm2d(64)
        self.conv2 = nn.ConvTranspose2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.batch2 = nn.BatchNorm2d(64)
        self.conv3 = nn.ConvTranspose2d(64, 2, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1))

        self.output_dim = dim
        self.norm = norm_layer(2)
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
        B, L, C = x.shape
        H, W = pair(int(math.sqrt(L)))
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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        if inplanes != planes:
            self.conv0 = conv3x3(inplanes, planes)

        self.inplanes = inplanes
        self.planes = planes

        self.conv1 = conv3x3(planes, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.drop = nn.Dropout2d(p=drop)

    def forward(self, x):
        if self.inplanes != self.planes:
            x = self.conv0(x)
            x = F.relu(x)

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.drop(out)

        out1 = self.conv1(out)
        # out1 = self.relu(out1)

        out2 = out1 + x

        return F.relu(out2)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Initial_Block(nn.Module):

    def __init__(self, planes, layers, kernel=3, block=BasicBlock, inplanes=3):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel - 1) / 2)
        self.inconv = nn.Conv2d(in_channels=inplanes, out_channels=planes,
                                kernel_size=3, stride=1, padding=1, bias=True)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_module_list.append(block(planes * (2 ** i), planes * (2 ** i)))

        # use strided conv instead of pooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_conv_list.append(
                nn.Conv2d(planes * 2 ** i, planes * 2 ** (i + 1), stride=2, kernel_size=kernel, padding=self.padding))

        # create module for bottom block
        self.bottom = block(planes * (2 ** layers), planes * (2 ** layers))

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_conv_list.append(nn.ConvTranspose2d(in_channels=planes * 2 ** (layers - i),
                                                        out_channels=planes * 2 ** max(0, layers - i - 1),
                                                        kernel_size=3,
                                                        stride=2, padding=1, output_padding=1, bias=True))
            self.up_dense_list.append(block(planes * 2 ** max(0, layers - i - 1), planes * 2 ** max(0, layers - i - 1)))

        # multiscale attention
        self.Guide1 = MS_Perceptron1(dims=16)
        self.bridge1 = Global_Perceptron_Main(dims=16, head=1)

        # 4 seq
        self.Guide2 = MS_Perceptron2(dims=16)
        self.bridge2 = Global_Perceptron_Fine(dims=16, head=1)

    def forward(self, x):

        # print('x_ini', x.device)
        out = self.inconv(x)
        out = F.relu(out)

        down_out = []
        up_out = []
        # down branch
        for i in range(0, self.layers):
            out = self.down_module_list[i](out)
            down_out.append(out)
            out = self.down_conv_list[i](out)
            out = F.relu(out)

        if self.layers == 3:
            down_out = self.Guide1(down_out)
            down_out = self.bridge1(down_out)
        elif self.layers == 4:
            down_out = self.Guide2(down_out)
            down_out = self.bridge2(down_out)

        # bottom branch
        out = self.bottom(out)
        bottom = out

        # up branch

        up_out.append(bottom)

        for j in range(0, self.layers):
            out = self.up_conv_list[j](out) + down_out[self.layers - j - 1].contiguous()
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class LadderBlock(nn.Module):

    def __init__(self, planes, layers, kernel=3, block=BasicBlock, inplanes=3):
        super().__init__()
        self.planes = planes
        self.layers = layers
        self.kernel = kernel

        self.padding = int((kernel - 1) / 2)
        self.inconv = block(planes, planes)

        # create module list for down branch
        self.down_module_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_module_list.append(block(planes * (2 ** i), planes * (2 ** i)))

        # use strided conv instead of poooling
        self.down_conv_list = nn.ModuleList()
        for i in range(0, layers):
            self.down_conv_list.append(
                nn.Conv2d(planes * 2 ** i, planes * 2 ** (i + 1), stride=2, kernel_size=kernel, padding=self.padding))

        # create module for bottom block
        self.bottom = block(planes * (2 ** layers), planes * (2 ** layers))

        # create module list for up branch
        self.up_conv_list = nn.ModuleList()
        self.up_dense_list = nn.ModuleList()
        for i in range(0, layers):
            self.up_conv_list.append(
                nn.ConvTranspose2d(planes * 2 ** (layers - i), planes * 2 ** max(0, layers - i - 1), kernel_size=3,
                                   stride=2, padding=1, output_padding=1, bias=True))
            self.up_dense_list.append(block(planes * 2 ** max(0, layers - i - 1), planes * 2 ** max(0, layers - i - 1)))

    def forward(self, x):
        out = self.inconv(x[-1])

        down_out = []
        # down branch
        for i in range(0, self.layers):
            out = out + x[-i - 1]
            out = self.down_module_list[i](out)
            down_out.append(out)

            out = self.down_conv_list[i](out)
            out = F.relu(out)

        # bottom branch
        out = self.bottom(out)
        bottom = out

        # up branch
        up_out = []
        up_out.append(bottom)

        for j in range(0, self.layers):
            out = self.up_conv_list[j](out) + down_out[self.layers - j - 1]
            # out = F.relu(out)
            out = self.up_dense_list[j](out)
            up_out.append(out)

        return up_out


class Final_LadderBlock(nn.Module):

    def __init__(self, planes, layers, kernel=3, block=BasicBlock, inplanes=3):
        super().__init__()
        self.block = LadderBlock(planes, layers, kernel=kernel, block=block)

    def forward(self, x):
        out = self.block(x)
        return out[-1]


class MCPANET(nn.Module):
    def __init__(self, inplanes=1, num_classes=2, layers=4, filters=10, ):
        super().__init__()
        self.initial_block = Initial_Block(planes=filters, layers=layers, inplanes=inplanes)
        self.final_block = Final_LadderBlock(planes=filters, layers=layers)
        self.final = nn.Conv2d(in_channels=filters, out_channels=num_classes, kernel_size=1)

    def forward(self, x):

        out = self.initial_block(x)
        out = self.final(out[-1])
        out = F.softmax(out, dim=1)
        return out
