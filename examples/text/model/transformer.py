# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/facebookresearch/DiT
# which is released under NonCommercial-4.0 license
# Part of this implementation is adapted from https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
# which is released under MIT license
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license


import math

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from torch import Tensor, nn

from . import rotary
from .gumbel_vector_quantizer import GumbelVectorQuantizer


def bias_dropout_add_scale(
    x: Tensor, scale: Tensor, residual: Tensor | None, prob: float, training: bool
) -> Tensor:
    return residual + scale * F.dropout(x, p=prob, training=training)


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale) + shift


class LayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones([dim]))
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = F.layer_norm(x.float(), [self.dim])

        return x * self.weight[None, None, :]


class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(time: Tensor, dim: int, max_period: int = 10000) -> Tensor:
        """Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=time.device)
        args = time[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, time: Tensor) -> Tensor:
        t_freq = self.timestep_embedding(time=time, dim=self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cond_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert dim % n_heads == 0, "dim must be devisable by n_heads"

        self.n_heads = n_heads
        self.dim = dim
        self.dropout = dropout
        self.head_dim = self.dim // self.n_heads

        # Self attention
        self.norm1 = LayerNorm(dim=dim)
        self.qw = nn.Linear(dim, dim, bias=False)
        self.kw = nn.Linear(dim, dim, bias=False)
        self.vw = nn.Linear(dim, dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        # MLP
        self.norm2 = LayerNorm(dim=dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True),
        )

        # AdaLN modulation (9 parameters for msa/cross/mlp)
        n_params = 6
        self.n_params = n_params
        self.adaLN_modulation = nn.Linear(cond_dim, n_params * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(
        self,
        x: Tensor,
        rotary_cos_sin: Tensor,
        c: Tensor,
    ) -> Tensor:
        batch_size, seq_len = x.shape[0], x.shape[1]

        # Get modulation parameters for all layers (9 or 6 total)
        modulation_params = self.adaLN_modulation(c)[:, None].chunk(
            self.n_params, dim=2
        )
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation_params

        # Self attention
        x_skip = x
        x = modulate(x=self.norm1(x), shift=shift_msa, scale=scale_msa)

        q = self.qw(x)
        k = self.kw(x)
        v = self.vw(x)

        q, k, v = (
            item.view(batch_size, seq_len, self.n_heads, self.head_dim)
            for item in (q, k, v)
        )

        with torch.amp.autocast("cuda", enabled=False):
            cos, sin = rotary_cos_sin
            original_dtype = q.dtype

            q = rotary.apply_rotary_emb_torch(
                x=q.float(), cos=cos.float(), sin=sin.float()
            ).to(original_dtype)
            k = rotary.apply_rotary_emb_torch(
                x=k.float(), cos=cos.float(), sin=sin.float()
            ).to(original_dtype)

        q, k, v = (item.transpose(1, 2) for item in (q, k, v))
        x = F.scaled_dot_product_attention(query=q, key=k, value=v)
        x = rearrange(x, "b h s d -> b s (h d)", b=batch_size)
        x = bias_dropout_add_scale(
            x=self.attn_out(x),
            scale=gate_msa,
            residual=x_skip,
            prob=self.dropout,
            training=self.training,
        )

        # MLP
        x = bias_dropout_add_scale(
            x=self.mlp(modulate(x=self.norm2(x), shift=shift_mlp, scale=scale_mlp)),
            scale=gate_mlp,
            residual=x,
            prob=self.dropout,
            training=self.training,
        )

        return x


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm_final = LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate(x=self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)

        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size: int, masked: bool, config: DictConfig):
        super().__init__()

        if isinstance(config, dict):
            config = OmegaConf.create(config)

        self.config = config
        self.vocab_size = vocab_size

        add_token = 1 if masked else 0
        self.masked = masked

        self.vocab_embed = nn.Embedding(self.vocab_size + add_token, config.hidden_size)
        self.vocab_embed_speech = nn.Embedding(
            config.vocab_size_speech + add_token, config.hidden_size
        )

        self.time_embedding = TimestepEmbedder(hidden_size=config.cond_dim)

        self.rotary_emb = rotary.Rotary(dim=config.hidden_size // config.n_heads)

        self.blocks = nn.ModuleList(
            [
                DDiTBlock(
                    dim=config.hidden_size,
                    n_heads=config.n_heads,
                    cond_dim=config.cond_dim,
                    dropout=config.dropout,
                )
                for _ in range(config.n_blocks)
            ]
        )

        self.quantizer = GumbelVectorQuantizer(
            dim=config.hidden_size,
            num_vars=config.n_quantizers,
            vq_dim=config.hidden_size,
            time_first=True,
            combine_groups=False,
            groups=3,
            temp=(2, 0.5, 0.999995),
        )

        self.output_layer = DDitFinalLayer(
            hidden_size=config.hidden_size,
            out_channels=vocab_size + add_token,
            cond_dim=config.cond_dim,
        )

        self.output_layer_speech = DDitFinalLayer(
            hidden_size=config.hidden_size,
            out_channels=config.vocab_size_speech + add_token,
            cond_dim=config.cond_dim,
        )

    # Internal forward method for inference
    def forward(
        self,
        x_t_speech: Tensor,
        time: Tensor,
        padding_mask_speech: Tensor,
        x_t_text: Tensor = None,
        padding_mask_text: Tensor = None,
        codebook_prob: float = 0.0,
        inference_block: int = None,
    ) -> Tensor:
        batch_size = x_t_speech.shape[0]

        x = self.vocab_embed_speech(x_t_speech.long())

        # Get time embeddings
        c = F.silu(self.time_embedding(time=time))

        if x_t_text is not None:
            assert x_t_text.shape[0] == x_t_speech.shape[0], f"{x_t_text.shape[0]},{x_t_speech.shape[0]}"
            x_text = self.vocab_embed(x_t_text.long())
            
            # 1. Calculate how much padding each tensor needs to reach target_size
            target_size = max(x.shape[1], x_text.shape[1])
            pad_x = target_size - x.shape[1]
            pad_text = target_size - x_text.shape[1]

            # 2. Pad the sequence dimension (dim 1) 
            # F.pad expects (last_dim_front, last_dim_back, second_last_front, second_last_back...)
            x_padded = F.pad(x, (0, 0, 0, pad_x)) 
            x_text_padded = F.pad(x_text, (0, 0, 0, pad_text))

            # 3. Stack them along the batch dimension (dim 0)
            x = torch.cat([x_text_padded, x_padded], dim=0)

            # 4. Handle padding masks similarly
            mask_text_padded = F.pad(padding_mask_text, (0, pad_text), value=True)
            mask_speech_padded = F.pad(padding_mask_speech, (0, pad_x), value=True)
            padding_mask = torch.cat([mask_text_padded, mask_speech_padded], dim=0)

            # 5. Double the time embedding
            c = torch.cat([c, c], dim=0)
        else:
            padding_mask = padding_mask_speech

        rotary_cos_sin = self.rotary_emb(x=x)

        inference_block = len(self.blocks) if inference_block is None else inference_block

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for i in range(inference_block):
                x = x * ~padding_mask[:, :, None]
                x = self.blocks[i](
                    x=x,
                    rotary_cos_sin=rotary_cos_sin,
                    c=c,
                )
                x = x * ~padding_mask[:, :, None]

        if codebook_prob > 0.0:
            q = self.quantizer(x)

            # q["x"]: B x T x C
            # Sample indexs according to the codebook prob
            random_idx = torch.randperm(q["x"].size(1))[
                : int(q["x"].size(1) * codebook_prob)
            ]
            # Make weight for q
            q_w = q["x"].new_zeros(q["x"].size(1))
            q_w[random_idx] = 1.0
            # Combine quantized codes and encoder output
            x = q_w.view(-1, 1) * q["x"] + (-q_w + 1).view(-1, 1) * x
            x = x * ~padding_mask[:, :, None]

        # Apply final layer with full precision
        if x_t_text is not None:
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x_out = self.output_layer(x=x[:batch_size], c=c[:batch_size])
                z = self.output_layer_speech(x=x[batch_size:], c=c[batch_size:])

                x_out = x_out * ~padding_mask[:batch_size, :, None]
                x_out = x_out[:,:x_t_text.shape[-1]]
                z = z * ~padding_mask[batch_size:, :, None]
                z = z[:,:x_t_speech.shape[-1]]

                if self.masked:
                    x_out[:,:,-1] = 0.0
                    z[:,:,-1] = 0.0

            return x_out, z
        else:
            with torch.amp.autocast("cuda", dtype=torch.float32):
                x_out = self.output_layer(x=x, c=c)
                x_out = x_out * ~padding_mask[:, :, None]

                if self.masked:
                    x_out[:,:,-1] = 0.0
            return x_out
