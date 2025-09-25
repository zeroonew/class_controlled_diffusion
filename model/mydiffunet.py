import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint as cp
from typing import Optional, Tuple

# --------------------
# 辅助函数 & 简单实现
# --------------------
def linear(in_features, out_features, bias=True):
    return nn.Linear(in_features, out_features, bias=bias)


def conv_nd(dims, in_channels, out_channels, kernel_size, stride=1, padding=0):
    if dims == 2:
        return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    raise NotImplementedError(f"dims={dims} not supported")


def timestep_embedding(timesteps: th.Tensor, dim: int, max_period: int = 10000):
    """
    Sinusoidal timestep embedding.
    timesteps: (N,) integer tensor
    returns: (N, dim)
    """
    half = dim // 2
    device = timesteps.device
    dtype = timesteps.dtype
    freqs = th.exp(-th.log(th.tensor(max_period, device=device, dtype=dtype)) * th.arange(half, device=device, dtype=dtype) / half)
    args = timesteps[:, None].float() * freqs[None]
    emb = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        emb = th.cat([emb, th.zeros_like(emb[:, :1])], dim=-1)
    return emb


def convert_module_to_f16(module):
    """将模块参数转为float16（仅对支持的层）"""
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.GroupNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                m.weight.data = m.weight.data.half()
            if hasattr(m, "bias") and getattr(m, "bias", None) is not None:
                m.bias.data = m.bias.data.half()


def convert_module_to_f32(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.GroupNorm)):
            if hasattr(m, "weight") and m.weight is not None:
                m.weight.data = m.weight.data.float()
            if hasattr(m, "bias") and getattr(m, "bias", None) is not None:
                m.bias.data = m.bias.data.float()


def zero_module(module):
    """将模块权重初始化为0，用于Diffusion输出层"""
    for p in module.parameters():
        p.data.zero_()
    return module


def normalization(channels):
    """GroupNorm归一化，组数取 min(32, channels)"""
    num_groups = min(32, channels)
    return nn.GroupNorm(num_groups, channels)


def checkpoint(fn, args, params, use_checkpoint):
    """简单封装 torch.checkpoint，兼容原调用"""
    if use_checkpoint:
        return cp.checkpoint(fn, *args)
    else:
        return fn(*args)


# --------------------
# 基础模块
# --------------------
class TimestepEmbedSequential(nn.Sequential):
    """Sequential but support passing timestep embedding to layers that accept (x, emb)."""

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, (ResBlock, AttentionBlock)):
                x = layer(x, emb)
            else:
                # layer expects only x
                x = layer(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, stride=1, padding=1),
        )

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, stride=1, padding=1)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, stride=1, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1, stride=1, padding=0)

    def forward(self, x, emb):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)

        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm = self.out_layers[0]
            out_rest = self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """多头自注意力块（适用于 2D feature maps）"""

    def __init__(self, channels, use_checkpoint=False, num_heads=1, num_head_channels=-1, use_new_attention_order=False):
        super().__init__()
        self.channels = channels
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.use_new_attention_order = use_new_attention_order

        if num_head_channels != -1:
            self.num_heads = channels // num_head_channels

        self.norm = normalization(channels)
        self.qkv = conv_nd(2, channels, channels * 3, 1)
        self.proj = zero_module(conv_nd(2, channels, channels, 1))

    def forward(self, x, emb=None):
        """
        x: (B, C, H, W)
        returns: (B, C, H, W)
        """
        b, c, h, w = x.shape
        x_norm = self.norm(x)  # (B, C, H, W)
        qkv = self.qkv(x_norm)  # (B, 3C, H, W)
        # reshape to (B, 3, num_heads, head_dim, HW)
        head_dim = c // self.num_heads
        qkv = qkv.reshape(b, 3, self.num_heads, head_dim, h * w)  # (B,3,H,hd,HW)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # each (B, num_heads, head_dim, HW)

        # compute attention
        scale = (head_dim) ** -0.5
        attn = th.einsum("b h d i, b h d j -> b h i j", q * scale, k)  # (B, num_heads, HW, HW)
        attn = F.softmax(attn, dim=-1)
        out = th.einsum("b h i j, b h d j -> b h d i", attn, v)  # (B, num_heads, head_dim, HW)
        out = out.reshape(b, c, h, w)
        out = self.proj(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, channels, conv_resample, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv_resample = conv_resample
        if conv_resample:
            # stride=2 for downsample conv
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, stride=2, padding=1)
        else:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, emb=None):
        if self.conv_resample:
            return self.conv(x)
        else:
            return self.pool(x)


class Upsample(nn.Module):
    def __init__(self, channels, conv_resample, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv_resample = conv_resample
        if conv_resample:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, stride=1, padding=1)

    def forward(self, x, emb=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.conv_resample:
            x = self.conv(x)
        return x


# ------------------------------
# 支持类别控制的 DiffUNet（核心）
# ------------------------------
class DiffUNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        large_model=False,
        use_fp16=False,
        num_classes: Optional[int] = None,
    ):
        super().__init__()

        if large_model:
            model_channels = 256
            num_res_blocks = 2
            attention_resolutions = "8,16,32"
        else:
            model_channels = 128
            num_res_blocks = 1
            attention_resolutions = "16"

        dropout = 0.1
        conv_resample = True
        dims = 2
        self.num_classes = num_classes
        use_checkpoint = False
        num_heads = 4
        num_head_channels = 64
        num_heads_upsample = -1
        use_scale_shift_norm = True
        resblock_updown = True
        use_new_attention_order = False

        # out_channels handling kept consistent with original snippet
        # out_channels = 6 if out_channels == 3 else out_channels
        out_channels = out_channels
        channel_mult = (1, 1, 2, 2, 4, 4)

        img_size = 256
        attention_ds = []
        for res in attention_resolutions.split(","):
            attention_ds.append(img_size // int(res))
        attention_resolutions = tuple(attention_ds)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.img_size = img_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # 时间步嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # 类别嵌入（如果需要）
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=time_embed_dim)

        # 编码器输入块
        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, stride=1, padding=1))])
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, down=True)
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # 中瓶颈
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order),
            ResBlock(ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm),
        )
        self._feature_size += ch

        # 解码器
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(ch + ich, time_embed_dim, dropout, out_channels=int(model_channels * mult), dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm)
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads_upsample, num_head_channels=num_head_channels, use_new_attention_order=use_new_attention_order))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(ResBlock(ch, time_embed_dim, dropout, out_channels=out_ch, dims=dims, use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm, up=True) if resblock_updown else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # 最后输出层（保持和原来行为一致）
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, stride=1, padding=1)),
        )

    # ---------------------
    # Forward 相关
    # ---------------------
    def forward(self, x: th.Tensor, t: th.Tensor, y: Optional[th.Tensor] = None, type_t: str = "noise_level"):
        """
        x: (B, C, H, W)
        t: timestep tensor (B,) or sigma-like tensor depending on type_t
        y: optional class labels (B,)
        """
        # 如果图很大则走patch路径
        H, W = x.shape[-2], x.shape[-1]
        if H <= 520 and W <= 520:
            # pad to multiple of 32 for U-Net architecture
            pad_h = (32 - (H % 32)) % 32
            pad_w = (32 - (W % 32)) % 32
            if pad_h != 0 or pad_w != 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
            if type_t == "timestep":
                out = self.forward_diffusion(x, t, y=y)
            elif type_t == "noise_level":
                out = self.forward_denoise(x, t, y=y)
            else:
                raise ValueError('type_t must be either "timestep" or "noise_level"')
            # crop back
            if pad_h != 0 or pad_w != 0:
                out = out[..., :H, :W]
            return out
        else:
            return self.patch_forward(x, t, y=y, type_t=type_t, patch_size=512)

    def patch_forward(self, x, t, y=None, type_t="noise_level", patch_size=512):
        # pad to multiples of patch_size
        B, C, H, W = x.shape
        pad_h = (patch_size - (H % patch_size)) % patch_size
        pad_w = (patch_size - (W % patch_size)) % patch_size
        if pad_h != 0 or pad_w != 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        else:
            x_padded = x
        H_pad, W_pad = x_padded.shape[-2], x_padded.shape[-1]
        E_padded = x_padded.new_zeros(B, C, H_pad, W_pad)
        h_patches = H_pad // patch_size
        w_patches = W_pad // patch_size
        for i in range(h_patches):
            for j in range(w_patches):
                h_start = i * patch_size
                w_start = j * patch_size
                patch = x_padded[..., h_start:h_start + patch_size, w_start:w_start + patch_size]
                E_patch = self.forward(patch, t, y=y, type_t=type_t)
                E_padded[..., h_start:h_start + patch_size, w_start:w_start + patch_size] = E_patch
        E = E_padded[..., :H, :W]
        return E

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward_diffusion(self, x: th.Tensor, timesteps: th.Tensor, y: Optional[th.Tensor] = None):
        """
        前向（timestep 条件）模式：融合时间步嵌入与类别嵌入
        timesteps: (B,) integer tensor (0..T-1)
        """
        assert (y is not None) == (self.num_classes is not None), "num_classes set => must provide y, otherwise y must be None"

        hs = []
        # 时间步嵌入
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # 类别融合
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],), f"y shape must be ({x.shape[0]},), got {y.shape}"
            class_emb = self.label_emb(y)
            time_emb = time_emb + class_emb

        # 编码器
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, time_emb)
            hs.append(h)

        # 中间
        h = self.middle_block(h, time_emb)

        # 解码器
        for module in self.output_blocks:
            skip = hs.pop()
            h = th.cat([h, skip], dim=1)
            h = module(h, time_emb)

        h = h.type(x.dtype)
        return self.out(h)

    def get_alpha_prod(self, beta_start=0.1 / 1000, beta_end=20 / 1000, num_train_timesteps=1000):
        betas = th.linspace(beta_start, beta_end, num_train_timesteps, dtype=th.float32)
        alphas = 1.0 - betas
        alphas_cumprod = th.cumprod(alphas, dim=0)

        sqrt_alphas_cumprod = th.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = th.sqrt(1.0 - alphas_cumprod)
        reduced_alpha_cumprod = th.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)
        sqrt_recip_alphas_cumprod = th.sqrt(1.0 / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = th.sqrt(1.0 / alphas_cumprod - 1)
        return (
            reduced_alpha_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            sqrt_1m_alphas_cumprod,
            sqrt_alphas_cumprod,
        )

    def find_nearest(self, array: th.Tensor, value: th.Tensor):
        """
        array: (L,) tensor of candidates
        value: (B,) tensor of query values
        returns: (B,) indices in array nearest to each value
        """
        # ensure shapes
        array = array.to(value.device)
        # compute abs differences and argmin along array dim
        # (B, L)
        diffs = (array[None, :] - value[:, None]).abs()
        idx = diffs.argmin(dim=1)
        return idx

    def forward_denoise(self, x: th.Tensor, sigma: th.Tensor, y: Optional[th.Tensor] = None):
        """
        denoise-mode forward (noise-level conditioning), returns denoised image in [0,1]
        sigma: scalar or tensor of shape (B,) or (B,1,1,1)
        """
        sigma = self._handle_sigma(sigma, batch_size=x.size(0), ndim=x.ndim, device=x.device, dtype=x.dtype)
        alpha = 1 / (1 + 4 * sigma ** 2)
        x_proc = alpha.sqrt() * (2 * x - 1)
        sigma_proc = sigma * alpha.sqrt()

        (
            reduced_alpha_cumprod,
            sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod,
            sqrt_1m_alphas_cumprod,
            sqrt_alphas_cumprod,
        ) = self.get_alpha_prod()

        # compute timesteps by matching sqrt_1m_alphas_cumprod to sigma_proc * 2
        # sigma_proc has shape (B,1,1,1) -> squeeze to (B,)
        val = (sigma_proc.squeeze(-1).squeeze(-1).squeeze(-1) * 2).to(x.device)
        timesteps = self.find_nearest(sqrt_1m_alphas_cumprod.to(x.device), val)
        timesteps = timesteps.to(x.device)

        noise_est_sample_var = self.forward_diffusion(x_proc, timesteps, y=y)
        noise_est = noise_est_sample_var[:, :3, ...]
        denoised = (x_proc - noise_est * sigma_proc * 2) / sqrt_alphas_cumprod.to(x.device)[timesteps].view(-1, 1, 1, 1)
        denoised = denoised.clamp(-1, 1)
        return (denoised + 1) / 2

    def _handle_sigma(self, sigma, batch_size, ndim, device, dtype):
        """确保 sigma 形状与类型正确，返回 shape (B, 1, 1, 1) 或 (B,) depending usage"""
        if not isinstance(sigma, th.Tensor):
            sigma = th.tensor(sigma, device=device, dtype=dtype)
        if sigma.ndim == 0:
            sigma = sigma.unsqueeze(0).repeat(batch_size)
        sigma = sigma.to(device=device, dtype=dtype)
        # turn into shape (B, 1, 1, 1) for broadcasting in some ops
        if sigma.ndim == 1:
            sigma = sigma.view(-1, 1, 1, 1)
        while sigma.ndim < ndim:
            sigma = sigma.unsqueeze(-1)
        return sigma
