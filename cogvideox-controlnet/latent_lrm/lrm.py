# Copyright (c) 2024, Ziwen Chen.

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from easydict import EasyDict as edict
from einops import rearrange
from gsplat import rasterization
from typing import Optional, Tuple
try:
    from .transformer import TransformerBlock
    from .mamba2 import Mamba2Block
    # from .loss import PerceptualLoss
    from .autoencoder_magvit import CogVideoXEncoder3D, DiagonalGaussianDistribution, AutoencoderKLCogVideoX
except:
    from transformer import TransformerBlock
    from mamba2 import Mamba2Block
    # from loss import PerceptualLoss
    from autoencoder_magvit import CogVideoXEncoder3D, DiagonalGaussianDistribution


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Processor(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 24
        self.dims = 1024

        self.block_type = 'mmmmmmmtmmmmmmmtmmmmmmmt'
        if len(self.block_type) > self.num_layers:
            self.block_type = self.block_type[:self.num_layers]
        elif len(self.block_type) < self.num_layers:
            self.block_type = self.block_type * (self.num_layers // len(
                self.block_type)) + self.block_type[:self.num_layers % len(self.block_type)]

        self.merge_at = []
        if isinstance(self.dims, int):
            self.dims = [self.dims]
        assert len(self.dims) == len(self.merge_at) + 1

        self.blocks = nn.ModuleList()
        if len(self.merge_at) > 0:
            self.resize_blocks = nn.ModuleList()
            self.merge_blocks = nn.ModuleList()
        dim_cur = self.dims[0]
        for i, s in enumerate(self.block_type):
            if i in self.merge_at:
                dim_next = self.dims[self.merge_at.index(i) + 1]
                self.resize_blocks.append(nn.Linear(dim_cur, dim_next))
                self.merge_blocks.append(
                    nn.Conv2d(dim_cur, dim_next, kernel_size=2,
                              stride=2, padding=0, bias=True, groups=dim_cur)
                )
                dim_cur = dim_next
            if s == "t":
                self.blocks.append(TransformerBlock(dim_cur, 64))
                self.blocks[-1].apply(_init_weights)
            elif s == "m":
                self.blocks.append(Mamba2Block(dim_cur, 256))
            else:
                raise ValueError(f"Invalid block type {s}")

    def run_one_block(self, i):
        def _run_one_block(x, num_global_tokens, v, h, w):
            if i in self.merge_at:
                if num_global_tokens > 0:
                    global_tokens, image_tokens = x[:,
                                                    :num_global_tokens], x[:, num_global_tokens:]
                    global_tokens = self.resize_blocks[self.merge_at.index(i)](
                        global_tokens)
                else:
                    image_tokens = x
                image_tokens = rearrange(
                    image_tokens, "b (v h w) d -> (b v) d h w", v=v, h=h, w=w)
                image_tokens = self.merge_blocks[self.merge_at.index(i)](
                    image_tokens)
                h = h // 2
                w = w // 2
                image_tokens = rearrange(
                    image_tokens, "(b v) d h w -> b (v h w) d", v=v, h=h, w=w)
                if num_global_tokens > 0:
                    x = torch.cat([global_tokens, image_tokens], dim=1)
                else:
                    x = image_tokens
            x = self.blocks[i](x)
            return x, h, w
        return _run_one_block

    def forward(self, x, num_global_tokens, v, h, w, use_checkpoint=True):
        """
        x: (B, L, D)
        Returns: B and D remain the same, L might change if there are merge layers
        """
        batch, seq_len, _ = x.shape
        num_image_tokens = seq_len - num_global_tokens
        assert num_image_tokens == v * h * w

        for i in range(self.num_layers):
            if use_checkpoint:
                x, h, w = torch.utils.checkpoint.checkpoint(self.run_one_block(
                    i), x, num_global_tokens, v, h, w, use_reentrant=False)
            else:
                x, h, w = self.run_one_block(i)(x, num_global_tokens, v, h, w)

        return x, h, w


class GaussianRenderer(torch.autograd.Function):
    @staticmethod
    def render(xyz, feature, scale, rotation, opacity, test_c2w, test_intr,
               W, H, sh_degree, near_plane, far_plane):
        opacity = opacity.sigmoid().squeeze(-1)
        scale = scale.exp()
        rotation = F.normalize(rotation, p=2, dim=-1)
        test_w2c = test_c2w.float().inverse().unsqueeze(0)  # (1, 4, 4)
        test_intr_i = torch.zeros(3, 3).to(test_intr.device)
        test_intr_i[0, 0] = test_intr[0]
        test_intr_i[1, 1] = test_intr[1]
        test_intr_i[0, 2] = test_intr[2]
        test_intr_i[1, 2] = test_intr[3]
        test_intr_i[2, 2] = 1
        test_intr_i = test_intr_i.unsqueeze(0)  # (1, 3, 3)
        rendering, _, _ = rasterization(xyz, rotation, scale, opacity, feature,
                                        test_w2c, test_intr_i, W, H, sh_degree=sh_degree,
                                        near_plane=near_plane, far_plane=far_plane,
                                        render_mode="RGB",
                                        backgrounds=torch.ones(
                                            1, 3).to(test_intr.device),
                                        rasterize_mode='classic')  # (1, H, W, 3)
        return rendering  # (1, H, W, 3)

    @staticmethod
    def forward(ctx, xyz, feature, scale, rotation, opacity, test_c2ws, test_intr,
                W, H, sh_degree, near_plane, far_plane):
        ctx.save_for_backward(xyz, feature, scale, rotation,
                              opacity, test_c2ws, test_intr)
        ctx.W = W
        ctx.H = H
        ctx.sh_degree = sh_degree
        ctx.near_plane = near_plane
        ctx.far_plane = far_plane
        with torch.no_grad():
            B, V, _ = test_intr.shape
            renderings = torch.zeros(B, V, H, W, 3).to(xyz.device)
            for ib in range(B):
                for iv in range(V):
                    renderings[ib, iv:iv+1] = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib],
                                                                      test_c2ws[ib, iv], test_intr[ib, iv], W, H, sh_degree, near_plane, far_plane)
        renderings = renderings.requires_grad_()
        return renderings

    @staticmethod
    def backward(ctx, grad_output):
        xyz, feature, scale, rotation, opacity, test_c2ws, test_intr = ctx.saved_tensors
        xyz = xyz.detach().requires_grad_()
        feature = feature.detach().requires_grad_()
        scale = scale.detach().requires_grad_()
        rotation = rotation.detach().requires_grad_()
        opacity = opacity.detach().requires_grad_()
        W = ctx.W
        H = ctx.H
        sh_degree = ctx.sh_degree
        near_plane = ctx.near_plane
        far_plane = ctx.far_plane
        with torch.enable_grad():
            B, V, _ = test_intr.shape
            for ib in range(B):
                for iv in range(V):
                    rendering = GaussianRenderer.render(xyz[ib], feature[ib], scale[ib], rotation[ib], opacity[ib],
                                                        test_c2ws[ib, iv], test_intr[ib, iv], W, H, sh_degree, near_plane, far_plane)
                    rendering.backward(grad_output[ib, iv:iv+1])

        return xyz.grad, feature.grad, scale.grad, rotation.grad, opacity.grad, None, None, None, None, None, None, None

# 完整模型整合


class Tokenizer(nn.Module):
    def __init__(self, o_dim=1024, version=2):
        super().__init__()
        if version == 2:
            self.camera_branch = nn.Sequential(
                nn.Conv3d(
                    in_channels=6,
                    out_channels=1024,
                    kernel_size=(4, 16, 16),
                    stride=(4, 16, 16),
                    padding=(2, 0, 8)
                ),
                nn.LayerNorm(1024)
            )
            self.latent_branch = nn.Sequential(
                nn.Conv2d(16, 1024, kernel_size=2, stride=2, padding=(0, 1)),
                nn.LayerNorm(1024),
            )
        else:
            self.camera_branch = nn.Sequential(
                nn.Conv3d(
                    in_channels=6,
                    out_channels=1024,
                    kernel_size=(4, 16, 16),
                    stride=(4, 16, 16),
                    padding=(2, 0, 0)
                ),
                nn.LayerNorm(1024)
            )
            self.latent_branch = nn.Sequential(
                nn.Conv2d(16, 1024, kernel_size=2, stride=2),
                nn.LayerNorm(1024),
            )
        self.fusion = nn.Linear(2048, o_dim)

    def forward(self, latent, camera_embedding):
        # latent 先转化成 (B*t C W H)
        B, C, t, H, W = latent.shape
        latent = rearrange(latent, 'B C t H W -> (B t) C H W')
        latent = self.latent_branch[0](latent)
        latent = rearrange(
            latent, '(B t) C H W -> B (t H W) C', t=t, B=B)  # B N C
        latent = self.latent_branch[1](latent)  # B N C

        camera_embedding = self.camera_branch[0](camera_embedding)  # b c t h w
        b, c, tt, hh, ww = camera_embedding.shape
        camera_embedding = rearrange(
            camera_embedding, "B C t H W -> B (t H W) C ")  # B N C
        camera_embedding = self.camera_branch[1](camera_embedding)  # B N C

        # (B N 2C) ->(B N C)
        return self.fusion(torch.concat([latent, camera_embedding], dim=-1)), tt, ww, hh


class LongLRM(nn.Module):
    def __init__(self, version=2):
        super().__init__()
        self.dtype = torch.float32  # Default dtype
        self.dim_start = 1024
        self.dim_out = 1024
        self.num_global_tokens = 2
        sh_degree = 0
        if self.num_global_tokens > 0:
            self.global_token_init = nn.Parameter(
                torch.randn(1, self.num_global_tokens, self.dim_start))
            nn.init.trunc_normal_(self.global_token_init, std=0.02)
        # self.vae = self.load_video_encoder()
        self.version = version
        self.tokenizer = Tokenizer(self.dim_start, self.version)
        self.input_layernorm = nn.LayerNorm(self.dim_start, bias=False)
        self.processor = Processor()
        self.layernorm = nn.LayerNorm(self.dim_out, bias=False)
        if version == 2:
            self.tokenDecoder = nn.ConvTranspose3d(
                in_channels=self.dim_out,
                out_channels=(1 + (sh_degree + 1) ** 2 * 3 + 3 + 4 + 1),
                kernel_size=(5, 8, 8),
                stride=(4, 8, 8),
                padding=(2, 0, 2),  # padding=(2, 0, 0),
            )
        else:
            self.tokenDecoder = nn.ConvTranspose3d(
                in_channels=self.dim_out,
                out_channels=(1 + (sh_degree + 1) ** 2 * 3 + 3 + 4 + 1),
                kernel_size=(5, 8, 8),
                stride=(4, 8, 8),
                padding=(2, 0, 0)
            )
        # gaussian
        self.gaussians = edict()
        self.gaussians.max_dist = 500
        self.gaussians.sh_degree = 0
        self.gaussians.near_plane = 0.01
        self.gaussians.far_plane = 1000000.0
        self.gaussians.scale_bias = -6.9
        self.gaussians.scale_max = -1.2
        self.gaussians.opacity_bias = -2.0

    def to(self, *args, **kwargs):
        # Let PyTorch handle device/dtype changes
        ret = super().to(*args, **kwargs)
        # Update dtype if specified
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]
        return ret

    def get_ray(self, c2ws, intr, B, V, H, W, device):
        ray_o = c2ws[:, :, :3, 3].unsqueeze(
            2).expand(-1, -1, H * W, -1).float()  # (B, V, H*W, 3) # camera origin
        x, y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
        x = (x.to(intr.dtype) + 0.5).view(1, 1, -
                                          1).expand(B, V, -1).to(device).contiguous()
        y = (y.to(intr.dtype) + 0.5).view(1, 1, -
                                          1).expand(B, V, -1).to(device).contiguous()
        # unproject to camera space
        x = (x - intr[:, :, 2:3]) / intr[:, :, 0:1]
        y = (y - intr[:, :, 3:4]) / intr[:, :, 1:2]
        ray_d = torch.stack([x, y, torch.ones_like(x)],
                            dim=-1).float()  # (B, V, H*W, 3)
        ray_d = F.normalize(ray_d, p=2, dim=-1)
        # (B, V, H*W, 3)
        ray_d = ray_d @ c2ws[:, :, :3, :3].transpose(-1, -2).contiguous()
        return ray_o, ray_d

    def forward(self, image_shape, image_latent, input_intr, input_c2ws, use_checkpoint=True, decoder=True):
        """
        input_images: (B, C, V, H, W)
        input_intr: (B, V, 4), (fx, fy, cx, cy)
        input_c2ws: (B, V, 4, 4)
        pos_avg_inv: (B, 4, 4)
        scene_scale: (B)
        """

        B, V, _, H, W = image_shape
        device = image_latent.device
        # Embed camera info
        ray_o, ray_d = self.get_ray(input_c2ws, input_intr, B, V, H, W, device)
        camera_embedding = torch.cat(
            [torch.cross(ray_o, ray_d, dim=-1), ray_d], dim=-1)
        camera_embedding = rearrange(
            camera_embedding, "b v (h w) c -> b c v h w", h=H, w=W)  #

        # Pachify
        # VAE编码
        # image_latent = self.vae_encoder(rearrange(input_images, "b v c h w -> b c v h w"))
        camera_embedding = camera_embedding.to(image_latent.dtype)
        # Token化处理
        image_tokens, v, ww, hh = self.tokenizer(
            image_latent, camera_embedding)  # 将图像和相机信息转换为tokens
        if self.num_global_tokens > 0:
            global_tokens = self.global_token_init.expand(B, -1, -1)
            # (B, num_global_tokens+V*hh*ww, D)
            tokens = torch.cat([global_tokens, image_tokens], dim=1)
        else:
            tokens = image_tokens
        tokens = self.input_layernorm(tokens)

        # Process tokens
        tokens, hh, ww = self.processor(
            tokens, self.num_global_tokens, v, hh, ww, use_checkpoint=use_checkpoint)
        # 计算完后进行下采样
        H1 = H // 2
        W1 = W // 2
        # Decode tokens
        image_tokens = tokens[:, self.num_global_tokens:]  # (B, V*hh*ww, D)
        image_tokens = self.layernorm(image_tokens)
        image_tokens = rearrange(
            image_tokens, 'B (v hh ww) d -> B d v hh ww', v=v, hh=hh, ww=ww)  # 重排tokens形状
        if decoder:
            # (B, V*hh*ww, ph*pw*(1 + (sh_degree+1)**2*3 + 3 + 4 + 1))
            gaussians = self.tokenDecoder(image_tokens)
            gaussians = rearrange(
                gaussians, "b  d v hh ww  -> b (v hh ww) d", b=B, v=V, hh=H1, ww=W1)  # B c*H1*W1 d

            dist, feature, scale, rotation, opacity = torch.split(
                gaussians, [1, (self.gaussians.sh_degree + 1) ** 2 * 3, 3, 4, 1], dim=-1)
            feature = feature.view(
                B, V*H1*W1, (self.gaussians.sh_degree + 1) ** 2, 3).contiguous()
            scale = (
                scale + self.gaussians.scale_bias).clamp(max=self.gaussians.scale_max)
            opacity = opacity + self.gaussians.opacity_bias

            # Align gaussian means to pixel centers
            dist = dist.sigmoid() * self.gaussians.max_dist  # (B, V*H1*W1, 1)
            ray_o, ray_d = self.get_ray(
                input_c2ws, input_intr*0.5, B, V, H1, W1, device)
            xyz = dist * ray_d.reshape(B, -1, 3) + \
                ray_o.reshape(B, -1, 3)  # (B, V*H1*W1, 3)

            gaussians = {
                "xyz": xyz.float(),
                "feature": feature.float(),
                "scale": scale.float(),
                "rotation": rotation.float(),
                "opacity": opacity.float()
            }

            # Render images at test views
            xyz = gaussians["xyz"]
            feature = gaussians["feature"]
            scale = gaussians["scale"]
            rotation = gaussians["rotation"]
            opacity = gaussians["opacity"]
            # 恢复到原size计算loss

            with torch.autocast(enabled=False, device_type="cuda"):
                # if use_checkpoint:
                # cannot simply use torch checkpoint as memory reduction relies on the loop through views
                renderings = GaussianRenderer.apply(xyz, feature, scale, rotation, opacity, input_c2ws, input_intr, W, H,
                                                    self.gaussians.sh_degree, self.gaussians.near_plane,
                                                    self.gaussians.far_plane)
            renderings = renderings.permute(
                0, 4, 1,  2, 3).contiguous()  # (B,  3, V, H, W)
            # renderings = renderings.permute(0, 1, 4,  2, 3).contiguous() # (B,  V, C, H, W)
            return renderings
            # render_latent
            # render_latent = self.vae_encoder(renderings)
            # return renderings,render_latent
        return image_tokens
