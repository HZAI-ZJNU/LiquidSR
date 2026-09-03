import time

import torch
import torch.nn as nn

import torch.nn.functional as F
from timm.models.layers import to_2tuple
from .LiquidImageLayer import LiquidImageLayer


class DSLNB(nn.Module):
    def __init__(
            self, num_head=5, dim=55):
        super(DSLNB, self).__init__()
        m = []
        depth = 2
        num_heads = num_head
        window_size = 16
        resolution = 64
        mlp_ratio = 2.0
        m.append(BasicLayer(dim=dim,
                            depth=depth,
                            resolution=resolution,
                            num_heads=num_heads,
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=True, qk_scale=None,
                            norm_layer=nn.LayerNorm))
        self.transformer_body = nn.Sequential(*m)

    def forward(self, x):
        res = self.transformer_body(x)
        return res


class BasicLayer(nn.Module):
    def __init__(self, dim, resolution, embed_dim=50, depth=2, num_heads=8, window_size=8,
                 mlp_ratio=1., qkv_bias=True, qk_scale=None, norm_layer=None):

        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.depth = depth
        self.window_size = window_size
        m_body = []
        for i in range(2):
            m_body.append(LiquidImageLayer(in_channels=dim))
            m_body.append(SwinTransformerBlock(dim=dim, resolution=resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 norm_layer=norm_layer))


        self.body = nn.Sequential(*m_body)
        self.patch_embed = PatchEmbed(
            embed_dim=dim, norm_layer=norm_layer)
        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        if mod_pad_h != 0 or mod_pad_w != 0:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x, h, w

    def forward(self, x):
        x, h, w = self.check_image_size(x)
        _, _, H, W = x.size()
        x_size = (H, W)
        x = self.patch_embed(x)
        for blk in self.body:
            x = blk(x, x_size)
        x = self.patch_unembed(x, x_size)
        if h != H or w != W:
            x = x[:, :, 0:h, 0:w].contiguous()
        return x
class DynamicNeuron(nn.Module):

    def __init__(self, channels, activation='tanh'):
        super().__init__()
        self.channels = channels

        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

        self.tau = nn.Parameter(torch.Tensor(1, channels, 1, 1))

        self.activation = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        nn.init.orthogonal_(self.conv.weight.squeeze(1))
        # nn.init.zeros_(self.bias)
        nn.init.uniform_(self.tau, 0.5, 2.0)

    def forward(self,  x):

        tau = F.softplus(self.tau) + 1e-6
        dxdt = (-x + self.activation(self.conv(x))) / tau
        return dxdt
class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.dim = dim
        self.resolution = to_2tuple(resolution)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale)

        mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.mlp = FeedForward(dim,mlp_hidden_dim)
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        H, W = x_size
        B, L, C = x.shape

        # x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        # mask = self.calculate_mask(x_size).to(x.device)
        shifted_x = self.attn(shifted_x, H, W, mask=None)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        # x = x.view(B, H * W, C)
        x = x.permute(0, 3, 1, 2)
        # FFN
        x = (x + self.mlp(x))
        x = x.permute(0,2,3,1).view(B, H * W, C)
        return x


class LocalModule(nn.Sequential):
    def __init__(self, channels):
        super().__init__()
        self.add_module('pointwise_prenorm_0', nn.BatchNorm2d(channels))
        self.add_module('pointwise_conv_0', nn.Conv2d(channels, channels, kernel_size=1, bias=False))
        self.add_module('depthwise_conv',
                        nn.Conv2d(channels, channels, padding=1, kernel_size=3, groups=channels, bias=False))
        self.add_module('pointwise_prenorm_1', nn.BatchNorm2d(channels))
        self.add_module('pointwise_conv_1', nn.Conv2d(channels, channels, kernel_size=1, bias=False))


class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.local = LocalModule(dim)

        self.dynamicNeuron = DynamicNeuron(dim)


    def forward(self, x, H, W, mask=None):
        temp = x.permute(0,3,1,2).contiguous()
        local = self.local(temp)
        local = local + temp
        local2 = self.dynamicNeuron(local)
        local2 = local2.permute(0, 2, 3, 1).contiguous()
        qkv = self.qkv(local2)

        # partition windows
        qkv = window_partition(qkv, self.window_size[0])  # nW*B, window_size, window_size, C

        B_, _, _, C = qkv.shape

        qkv = qkv.view(-1, self.window_size[0] * self.window_size[1], C)  # nW*B, window_size*window_size, C

        N = self.window_size[0] * self.window_size[1]
        C = C // 3

        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        # merge windows
        x = x.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(x, self.window_size[0], H, W)  # B H' W' C
        x = x+local.permute(0, 2, 3, 1).contiguous()
        return x


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super(FeedForward, self).__init__()
        hidden_features = hidden_features or in_features

        self.project_in = nn.Conv2d(in_features, hidden_features , kernel_size=1)
        # self.depthwise_conv = nn.Conv2d(hidden_features, hidden_features,
        #                                 kernel_size=3, padding=1, groups=hidden_features)
        self.act = act_layer()
        self.project_out = nn.Conv2d(hidden_features, in_features, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        # x = self.depthwise_conv(x)
        x = self.act(x)
        x = self.project_out(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=50, norm_layer=None):
        super().__init__()

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=50):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops

