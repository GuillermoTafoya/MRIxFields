"""Host-staged full-volume inference for the pinned 3D Run C KL-VAE.

Only convolution and pointwise operations are partitioned spatially. Each normalization
group always sees every spatial voxel. This adapter changes floating-point kernel
selection, so results require qualification against the native implementation.
"""
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from fieldbridge.models.autoencoders import kl_vae
from fieldbridge.models.film import FiLMLayer


def _store_part(dest, value, shape, dim, start):
    """Copy one computed range into a host tensor of the operation's actual dtype."""
    if dest is None:
        dest = torch.empty(shape, dtype=value.dtype, device="cpu")
    dest.narrow(dim, start, value.shape[dim]).copy_(value)
    return dest


def _slab_conv(layer, x, slab_depth, scale=1):
    """Run a zero-padded Conv3d, optionally preceded by integer nearest upsampling."""
    if type(layer) is not nn.Conv3d or layer.padding_mode != "zeros":
        raise TypeError("Expected a zero-padded Conv3d.")
    dims = tuple(v * scale for v in x.shape[2:])
    stride, pad, dilation, kernel = layer.stride, layer.padding, layer.dilation, layer.kernel_size
    out_dims = tuple((n + 2*p - d*(k-1) - 1)//s + 1
                     for n, p, d, k, s in zip(dims, pad, dilation, kernel, stride))
    shape = (x.shape[0], layer.out_channels, *out_dims)
    device = layer.weight.device
    result = None
    for start in range(0, out_dims[0], slab_depth):
        stop = min(start + slab_depth, out_dims[0])
        # Global coordinate of the first and one past the last input needed.
        lo = start * stride[0] - pad[0]
        hi = (stop - 1) * stride[0] - pad[0] + dilation[0] * (kernel[0] - 1) + 1
        left, right = max(lo, 0), min(hi, dims[0])
        src_start = left // scale
        src_stop = (right + scale - 1) // scale
        tile = x[:, :, src_start:src_stop].to(device=device, memory_format=torch.contiguous_format)
        if scale != 1:
            tile = F.interpolate(tile, scale_factor=scale, mode="nearest")
            offset = left - src_start * scale
            tile = tile[:, :, offset:offset + right - left]
        if left != lo or right != hi:
            tile = F.pad(tile, (0, 0, 0, 0, left-lo, hi-right))
        val = F.conv3d(tile, layer.weight, layer.bias, stride,
                       (0, pad[1], pad[2]), dilation, layer.groups)
        result = _store_part(result, val, shape, 2, start)
        del val, tile
    return result


def _norm_act(norm, activation, x):
    """Each native GroupNorm call covers one complete independent channel group."""
    if type(norm) is not nn.GroupNorm or norm.weight is None or norm.bias is None:
        raise TypeError("This Run C adapter requires affine GroupNorm.")
    if type(activation) not in (nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Sigmoid, nn.Tanh):
        raise TypeError("Unsupported residual pointwise activation.")
    width = norm.num_channels // norm.num_groups
    result = None
    for first in range(0, norm.num_channels, width):
        last = first + width
        group = x[:, first:last].to(device=norm.weight.device, memory_format=torch.contiguous_format)
        value = F.group_norm(group, 1, norm.weight[first:last], norm.bias[first:last], norm.eps)
        value = activation(value)
        result = _store_part(result, value, x.shape, 1, first)
        del group, value
    return result


def _residual(block, x, slab_depth):
    if type(block) is not kl_vae._ResBlock:
        raise TypeError("Unsupported residual block.")
    normed = _norm_act(block.norm1, block.act1, x)
    branch = _slab_conv(block.conv1, normed, slab_depth)
    del normed
    normed = _norm_act(block.norm2, block.act2, branch)
    del branch
    branch = _slab_conv(block.conv2, normed, slab_depth)
    del normed
    skip = x if type(block.skip) is nn.Identity else _slab_conv(block.skip, x, slab_depth)
    device = block.conv2.weight.device
    dtype = torch.promote_types(branch.dtype, skip.dtype)
    result = branch if branch.dtype == dtype else torch.empty(branch.shape, dtype=dtype, device="cpu")
    for start in range(0, branch.shape[2], slab_depth):
        cut = (slice(None), slice(None), slice(start, start + slab_depth))
        result[cut].copy_(branch[cut].to(device) + skip[cut].to(device))
    return result


def _upsample(up, h, slab_depth):
    if type(up) is not nn.Sequential or len(up) != 2 or type(up[0]) is not nn.Upsample:
        raise TypeError("Unsupported decoder upsampler.")
    if up[0].mode != "nearest" or up[0].scale_factor != 2.0 or up[0].size is not None:
        raise TypeError("This adapter supports the pinned nearest-neighbor x2 upsampler.")
    return _slab_conv(up[1], h, slab_depth, scale=2)


def _film(layer, x, conditioning, slab_depth):
    if type(layer) is not FiLMLayer:
        raise TypeError("Unsupported domain conditioning.")
    device = layer.projection.weight.device
    conditioning = conditioning.to(device=device, dtype=x.dtype)
    result = None
    for start in range(0, x.shape[2], slab_depth):
        tile = x[:, :, start:start + slab_depth].to(device)
        value = layer(tile, conditioning)
        result = _store_part(result, value, x.shape, 2, start)
        del tile, value
    return result


def enable_low_memory(encoder, decoder, slab_depth=8):
    """Adapt only the two inference entry points; preserve the module/weight graph.

    Install once on loaded eval-mode models. The caller supplies inference mode,
    device placement, the encoder FP32 policy, and the decoder autocast context.
    The default is eight output planes per convolution.
    """
    if type(encoder) is not kl_vae.KLVAEEncoder or type(decoder) is not kl_vae.KLVAEDecoder:
        raise TypeError("Expected the pinned Run C KL-VAE classes.")
    if encoder.spatial_dims != 3 or decoder.spatial_dims != 3 or decoder.output_activation != "none":
        raise ValueError("Expected a 3D encoder and linear-head 3D decoder.")
    if type(slab_depth) is not int or slab_depth < 1:
        raise ValueError("slab_depth must be a positive integer.")
    if encoder.training or decoder.training:
        raise ValueError("Call eval() on both models before installing.")
    if "encode_dist" in vars(encoder) or "decode" in vars(decoder):
        raise ValueError("Inference entry points already overridden.")

    def encode_dist(model, x, domain=None):
        if model.training or torch.is_grad_enabled():
            raise RuntimeError("Host staging is for inference only.")
        kl_vae._validate_input_tensor(x, spatial_dims=3, channels=model.in_channels,
                                      downsample_factor=model.downsample_factor)
        h = _slab_conv(model.stem, x, slab_depth)
        # Keep no handle to the input of an entire residual stack; only the current
        # residual has to survive until its skip addition.
        for stack, down in ((model.res1, model.down1), (model.res2, model.down2), (model.res3, None)):
            for block in stack:
                h = _residual(block, h, slab_depth)
            if down is not None:
                h = _slab_conv(down, h, slab_depth)
        mean, logvar = _slab_conv(model.to_dist, h, slab_depth).to(x.device).chunk(2, dim=1)
        return mean, logvar.clamp(kl_vae._LOGVAR_MIN, kl_vae._LOGVAR_MAX)

    def decode(model, z, domain):
        if model.training or torch.is_grad_enabled():
            raise RuntimeError("Host staging is for inference only.")
        if z.ndim != 5 or z.shape[1] != model.latent_channels:
            raise ValueError("Unexpected latent dimensions.")
        cond = model.domain_conditioner(domain, batch_size=z.shape[0], device=z.device) if model.domain_conditioner is not None else None
        h = _slab_conv(model.from_latent, z, slab_depth)
        for stack, film, up in ((model.res1, model.domain_film1, model.up1),
                                (model.res2, model.domain_film2, model.up2),
                                (model.res3, model.domain_film3, None)):
            for block in stack:
                h = _residual(block, h, slab_depth)
            if cond is not None:
                h = _film(film, h, cond, slab_depth)
            if up is not None:
                h = _upsample(up, h, slab_depth)
        return _slab_conv(model.to_image, h, slab_depth).to(z.device)

    encoder.encode_dist = MethodType(encode_dist, encoder)
    decoder.decode = MethodType(decode, decoder)
