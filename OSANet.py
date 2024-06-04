import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class PA(nn.Module):
    def __init__(self, dim, mode: str):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.sigmoid = nn.Sigmoid()
        self.dim = dim

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class LeFF(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear1 = nn.Sequential(nn.Linear(in_features, hidden_features),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, groups=hidden_features, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_features, out_features))
        self.dim = in_features
        self.hidden_dim = hidden_features

    def forward(self, x, x_size):
        # bs x hw x c
        B, L, C = x.shape
        H, W = x_size
        x = self.linear1(x)
        # spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=H, w=W)
        # bs,hidden_dim,32x32
        x = self.dwconv(x)
        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=H, w=W)
        x = self.linear2(x)

        return x


class LePEAttention(nn.Module):
    def __init__(self, channel, direction, band_width, num_heads, attn_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.channel = channel
        self.band_width = band_width
        self.num_heads = num_heads
        head_dim = channel // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.direction = direction
        self.H_sp = band_width
        self.W_sp = band_width
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x, x_size):
        B, L, C = x.shape
        H, W = x_size
        if self.direction == 0:
            self.H_sp = H
        if self.direction == 1:
            self.W_sp = W
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        pad_h = (self.H_sp - (H % self.H_sp)) % self.H_sp
        pad_w = (self.W_sp - (W % self.W_sp)) % self.W_sp
        x = F.pad(x, (0, pad_w, 0, pad_h))
        b, c, h, w = x.shape
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x, [h, w]

    def forward(self, x_size, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H, W = x_size
        B, L, C = q.shape

        q, q_size = self.im2cswin(q, x_size)
        k, k_size = self.im2cswin(k, x_size)
        v, v_size = self.im2cswin(v, x_size)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v)
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, q_size[0], q_size[1])  # B H' W' C

        x = x[:, :H, :W, :].contiguous().view(B, -1, C)

        return x


class CST(nn.Module):
    def __init__(self,
                 band_width,
                 channel,
                 num_heads=6,
                 mlp_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 ):
        super().__init__()
        self.band_width = band_width
        self.channel = channel
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(channel, channel * 3, bias=qkv_bias)
        self.norm1 = norm_layer(channel)
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.patch_norm = patch_norm
        self.proj = nn.Linear(channel, channel)
        self.attn = nn.ModuleList(LePEAttention(channel=channel, direction=i,
                                                band_width=band_width, num_heads=num_heads,
                                                attn_drop=attn_drop_rate, qk_scale=qk_scale)
                                  for i in range(2))

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.mlp = LeFF(in_features=channel, hidden_features=channel * mlp_ratio, out_features=channel,
                        act_layer=nn.GELU)

        # self.mlp = Gated_Conv_FeedForward(channel)
        self.norm2 = norm_layer(channel)

    def forward(self, x, x_size):
        b, l, c = x.shape
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(b, -1, 3, c).permute(2, 0, 1, 3).contiguous()

        x1 = self.attn[0](x_size, qkv[:, :, :, :c // 2])
        x2 = self.attn[1](x_size, qkv[:, :, :, c // 2:])
        attened_x = torch.cat([x1, x2], dim=2)

        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x), x_size))

        return x


class CSAT(nn.Module):
    def __init__(self,
                 band_width,
                 depth,
                 channel,
                 angRes,
                 num_heads=6,
                 mlp_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 grid=False):
        super().__init__()
        self.band_width = band_width
        self.depth = depth
        self.channel = channel
        self.angRes = angRes
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.patch_norm = patch_norm
        self.pape = PA(dim=channel, mode='ang')
        self.cst = nn.ModuleList([CST(band_width=band_width, channel=channel, num_heads=num_heads,
                                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                      attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                      norm_layer=norm_layer, patch_norm=patch_norm)
                                  for i in range(depth)])

    def forward(self, x):
        # x:[b, c, a*a, h, w]
        b, c, a, h, w = x.shape
        x = rearrange(x, 'b c (a1 a2) h w -> (b h w) c a1 a2', a1=self.angRes, a2=self.angRes)
        x = self.pape(x)

        B, C, A1, A2 = x.shape
        x_size = [A1, A2]
        x = self.ang2token(x)

        for layer in self.cst:
            x = layer(x, x_size)

        x = rearrange(x, '(b h w) (a1 a2) c -> b c (a1 a2) h w', h=h, w=w, a1=A1, a2=A2)

        return x

    def ang2token(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class inter_CSAT(nn.Module):
    def __init__(self,
                 band_width,
                 depth,
                 channel,
                 angRes,
                 num_heads=6,
                 mlp_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 ):
        super().__init__()
        self.band_width = band_width
        self.depth = depth
        self.channel = channel
        self.angRes = angRes
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.patch_norm = patch_norm
        self.pape = PA(dim=channel, mode='ang')
        self.cst = nn.ModuleList([CST(band_width=band_width, channel=channel, num_heads=num_heads,
                                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                      attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                      norm_layer=norm_layer, patch_norm=patch_norm)
                                  for i in range(depth)])
        self.n = 2

    def forward(self, x):
        # x:[b, c, a*a, h, w]
        b, c, a, h, w = x.shape
        x = rearrange(x, 'b c (a1 a2) (ph h) (pw w) -> (b ph pw) c (h a1) (w a2)', a1=self.angRes, h=self.n,
                      w=self.n)
        x = self.pape(x)

        B, C, A1, A2 = x.shape
        x_size = [A1, A2]
        x = rearrange(x, 'b c h w -> b (h w) c')

        for layer in self.cst:
            x = layer(x, x_size)

        x = rearrange(x, '(b ph pw) (h a1 w a2) c -> b c (a1 a2) (ph h) (pw w) ',
                      ph=h // self.n, pw=w // self.n, a1=self.angRes, a2=self.angRes,
                      h=self.n, w=self.n)
        return x


class CSST(nn.Module):
    def __init__(self,
                 band_width,
                 depth,
                 channel,
                 angRes,
                 num_heads=6,
                 mlp_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 grid=False):
        super().__init__()
        self.band_width = band_width
        self.depth = depth
        self.channel = channel
        self.angRes = angRes
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.patch_norm = patch_norm
        self.pape = PA(dim=channel, mode='spa')
        self.cst = nn.ModuleList([CST(band_width=band_width, channel=channel, num_heads=num_heads,
                                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                      attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                      norm_layer=norm_layer, patch_norm=patch_norm)
                                  for i in range(depth)])

    def forward(self, x):
        # x:[b, c, a*a, h, w]
        b, c, a, h, w = x.shape
        x = rearrange(x, 'b c a h w -> (b a) c h w')

        x = self.pape(x)

        B, C, H, W = x.shape
        x_size = [H, W]
        x = rearrange(x, '(b a) c h w -> (b a) (h w) c', b=b, a=a, h=H, w=W)

        for layer in self.cst:
            x = layer(x, x_size)

        x = rearrange(x, '(b a) (h w) c -> b c a h w', b=b, a=a, h=H, w=W)

        return x


class inter_CSST(nn.Module):
    def __init__(self,
                 band_width,
                 depth,
                 channel,
                 angRes,
                 num_heads=6,
                 mlp_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 grid=True):
        super().__init__()
        self.band_width = band_width
        self.depth = depth
        self.channel = channel
        self.angRes = angRes
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.patch_norm = patch_norm
        self.pape = PA(dim=channel, mode='spa')
        self.patch_size = 4
        self.cst = nn.ModuleList([CST(band_width=band_width, channel=channel, num_heads=num_heads,
                                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                      attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                      norm_layer=norm_layer, patch_norm=patch_norm)
                                  for i in range(depth)])

    def forward(self, x):
        # x:[b, c, a*a, h, w]
        b, c, a, h, w = x.shape
        x = rearrange(x, 'b c (a1 a2) (h ph) (w pw) -> (b h w) c (a1 ph) (a2 pw)', ph=self.patch_size,
                      pw=self.patch_size, a1=self.angRes, a2=self.angRes)
        x = self.pape(x)
        B, C, H, W = x.shape
        x_size = [H, W]
        x = rearrange(x, 'b c h w -> b (h w) c')

        for layer in self.cst:
            x = layer(x, x_size)

        x = rearrange(x, '(b h w) (a1 ph a2 pw) c -> b c (a1 a2) (h ph) (w pw)',
                      a1=self.angRes, a2=self.angRes, h=h // self.patch_size, w=w // self.patch_size,
                      ph=self.patch_size, pw=self.patch_size)
        return x


class CSET(nn.Module):
    def __init__(self,
                 band_width,
                 depth,
                 channel,
                 angRes,
                 num_heads=6,
                 mlp_ratio=2,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 grid=False):
        super().__init__()
        self.band_width = band_width
        self.depth = depth
        self.channel = channel
        self.angRes = angRes
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.patch_norm = patch_norm
        self.pape = PA(dim=channel, mode='epi')
        self.cst = nn.ModuleList([CST(band_width=band_width, channel=channel, num_heads=num_heads,
                                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate,
                                      # attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                                      norm_layer=norm_layer, patch_norm=patch_norm)
                                  for i in range(depth)])

    def forward(self, x):
        # x:[b, c, a*a, h, w]
        # vertical and horizontal are parameter-sharing
        b, c, a, h, w = x.shape

        # horizon
        x = rearrange(x, 'b c (a1 a2) h w -> (b h a1) c w a2 ', a1=self.angRes, a2=self.angRes, h=h, w=w)
        x = self.pape(x)
        B, C, W, A2 = x.shape
        x_size = [W, A2]
        x = rearrange(x, '(b h a1) c w a2 -> (b h a1) (w a2) c', a1=self.angRes, a2=A2, h=h, w=W)

        for layer in self.cst:
            x = layer(x, x_size)

        x = rearrange(x, '(b h a1) (w a2) c -> (b h a1) c w a2 ', a1=self.angRes, a2=A2, h=h, w=W)

        # vertical
        x = rearrange(x, '(b h a1) c w a2 -> (b w a2) c h a1 ', a1=self.angRes, a2=self.angRes, h=h, w=w)
        B, C, H, A1 = x.shape
        x_size = [H, A1]
        x = rearrange(x, '(b w a2) c h a1 -> (b w a2) (h a1) c', a1=A1, a2=self.angRes, h=H, w=w)

        for layer in self.cst:
            x = layer(x, x_size)

        x = rearrange(x, '(b w a2) (h a1) c -> b c (a1 a2) h w ', a1=A1, a2=self.angRes, h=H, w=w)
        return x


class intra_sa_block(nn.Module):
    def __init__(self,
                 band_width,
                 depth,
                 channel,
                 angRes,
                 num_heads,
                 ):
        super().__init__()
        self.csat = CSAT(band_width=band_width, depth=depth, channel=channel, angRes=angRes, num_heads=num_heads)
        self.csst = CSST(band_width=band_width, depth=depth, channel=channel, angRes=angRes, num_heads=num_heads)

    def forward(self, x):
        x = self.csat(x)
        x = self.csst(x)
        return x


class inter_sa_block(nn.Module):
    def __init__(self,
                 band_width,
                 depth,
                 channel,
                 angRes,
                 num_heads,
                 ):
        super().__init__()
        self.csat = inter_CSAT(band_width=band_width, depth=depth, channel=channel, angRes=angRes, num_heads=num_heads)
        self.csst = inter_CSST(band_width=band_width, depth=depth, channel=channel, angRes=angRes, num_heads=num_heads)

    def forward(self, x):
        x = self.csat(x)
        x = self.csst(x)
        return x


class omni_spa_ang_block(nn.Module):
    def __init__(self,
                 depth,
                 channel,
                 angRes,
                 ):
        super().__init__()
        self.channel = channel
        self.intra = intra_sa_block(band_width=1, depth=depth, channel=channel, angRes=angRes, num_heads=6)
        self.inter = inter_sa_block(band_width=1, depth=depth, channel=channel, angRes=angRes, num_heads=6)
        self.epi = CSET(band_width=1, depth=depth, channel=channel, angRes=angRes, num_heads=6)
        self.conv = nn.Sequential(
            nn.Conv3d(channel * 2, channel * 2, kernel_size=(1, 3, 3), padding=(0, 1, 1), groups=channel * 2,
                      bias=False),
            nn.Conv3d(channel * 2, channel, kernel_size=1, bias=False))

    def forward(self, x):
        x1 = self.epi(self.inter(x))
        x2 = self.epi(self.intra(x))
        out = torch.cat((x1, x2), dim=1)
        return self.conv(out) + x


class Net(nn.Module):
    def __init__(self, angRes, upscale_factor, channels):
        super(Net, self).__init__()
        self.channels = channels
        self.angRes = angRes
        self.factor = upscale_factor
        blocks = 4

        ##################### Initial Convolution ####################
        self.conv_init0 = nn.Sequential(
            nn.Conv3d(1, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.conv_init = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.body = self.Make_Layer(layer_num=blocks, block=omni_spa_ang_block)

        ####################### UP Sampling ###########################
        self.upsampling = nn.Sequential(
            nn.Conv2d(channels, channels * self.factor ** 2, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.PixelShuffle(self.factor),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, lr, info=None):
        # Bicubic
        lr_upscale = interpolate(lr, self.angRes, scale_factor=self.factor, mode='bicubic')
        # [B(atch), 1, A(ngRes)*h(eight)*S(cale), A(ngRes)*w(idth)*S(cale)]

        # reshape
        lr = rearrange(lr, 'b c (a1 h) (a2 w) -> b c (a1 a2) h w', a1=self.angRes, a2=self.angRes)
        # [B, C(hannels), A^2, h, w]

        # Initial Convolution
        buffer = self.conv_init0(lr)
        buffer_init = self.conv_init(buffer) + buffer  # [B, C, A^2, h, w]

        buffer_out = self.body[0](buffer_init)  # [B, C, A^2, h, w]# [B, C, A^2, h, w]
        buffer_out = self.body[1](buffer_out)
        buffer_out = self.body[2](buffer_out)
        buffer_out = self.body[3](buffer_out)

        # Up-Sampling
        buffer_out = rearrange(buffer_out, 'b c (a1 a2) h w -> b c (a1 h) (a2 w)', a1=self.angRes, a2=self.angRes)
        buffer_out = self.upsampling(buffer_out)
        out = buffer_out + lr_upscale

        return out

    def Make_Layer(self, layer_num, block):
        layer = nn.ModuleList()
        for i in range(layer_num):
            layer.append(block(depth=2, channel=self.channels, angRes=self.angRes))
        return layer


def interpolate(x, angRes, scale_factor, mode):
    [B, _, H, W] = x.size()
    h = H // angRes
    w = W // angRes
    x_upscale = x.view(B, 1, angRes, h, angRes, w)
    x_upscale = x_upscale.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * angRes ** 2, 1, h, w)
    x_upscale = F.interpolate(x_upscale, scale_factor=scale_factor, mode=mode, align_corners=False)
    x_upscale = x_upscale.view(B, angRes, angRes, 1, h * scale_factor, w * scale_factor)
    x_upscale = x_upscale.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, 1, H * scale_factor,
                                                                      W * scale_factor)  # [B, 1, A*h*S, A*w*S]
    return x_upscale


if __name__ == '__main__':
    torch.cuda.set_device("cuda:1")
    net = Net(5, 4, 60).cuda()
    print(net)
    from thop import profile

    #
    # from ptflops import get_model_complexity_info
    # with torch.no_grad():
    #     flops, params = get_model_complexity_info(net, (1, 160, 160), as_strings=True, print_per_layer_stat=True)
    # print("%s |%s" % (flops, params))
    input = torch.randn(1, 1, 160, 160).cuda()
    total = sum([param.nelement() for param in net.parameters()])
    flops, params = profile(net, inputs=(input,))
    print('   Number of parameters: %.2fM' % (total / 1e6))
    print('   Number of FLOPs: %.2fG' % (flops / 1e9))

    # Number of parameters: 1.38M / 1.43M
    # Number of FLOPs: 53.85G / 55.12G
