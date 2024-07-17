"""
Implementation from https://github.com/DarshanDeshpande/jax-models/blob/main/jax_models/models/van.py
"""

import jax 
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Union, Tuple, Callable, Sequence, Iterable


conv_init = nn.initializers.variance_scaling(
    scale=2.0, mode="fan_out", distribution="normal"
)
default_init = nn.initializers.variance_scaling(
    scale=1.0, mode="fan_in", distribution="normal"
)

def trunc_norm_init(key, shape, dtype=jnp.float32, std=0.02, mean=0.0):
    return std * jax.random.truncated_normal(key, -2, 2, shape) + mean


class OverlapPatchEmbed(nn.Module):
    emb_dim: int = 768
    patch_size: int = 16
    stride: int = 4
    kernel_init: Callable = nn.initializers.xavier_normal()
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        conv = nn.Conv(
            self.emb_dim,
            (self.patch_size, self.patch_size),
            self.stride,
            padding=(
                (self.patch_size // 2, self.patch_size // 2),
                (self.patch_size // 2, self.patch_size // 2),
            ),
            kernel_init=self.kernel_init,
            name="proj",
        )(inputs)
        norm = nn.BatchNorm(
            momentum=0.9, use_running_average=deterministic, name="norm"
        )(conv)
        return norm


class DepthwiseConv2D(nn.Module):
    kernel_shape: Union[int, Sequence[int]] = (1, 1)
    stride: Union[int, Sequence[int]] = (1, 1)
    padding: Union[str, Sequence[Tuple[int, int]]] = "SAME"
    channel_multiplier: int = 1
    use_bias: bool = True
    weights_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Optional[Callable] = nn.initializers.zeros

    @nn.compact
    def __call__(self, input):
        w = self.param(
            "kernel",
            self.weights_init,
            self.kernel_shape + (1, self.channel_multiplier * input.shape[-1]),
        )
        if self.use_bias:
            b = self.param(
                "bias", self.bias_init, (self.channel_multiplier * input.shape[-1],)
            )

        conv = jax.lax.conv_general_dilated(
            input,
            w,
            self.stride,
            self.padding,
            (1,) * len(self.kernel_shape),
            (1,) * len(self.kernel_shape),
            ("NHWC", "HWIO", "NHWC"),
            input.shape[-1],
        )
        if self.use_bias:
            bias = jnp.broadcast_to(b, conv.shape)
            return conv + bias
        else:
            return conv


class TransformerMLP(nn.Module):
    dim: int = 256
    out_dim: int = 256
    dropout: float = 0.2
    use_dwconv: bool = False
    conv_kernel_init: Callable = nn.initializers.xavier_normal()
    linear: bool = False
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        x = nn.Conv(self.dim, (1, 1), kernel_init=self.conv_kernel_init, name="fc1")(
            inputs
        )

        x = DepthwiseConv2D((3, 3), name="dwconv", weights_init=self.conv_kernel_init)(
            x
        )

        x = nn.gelu(x)
        x = nn.Dropout(self.dropout)(x, deterministic)
        x = nn.Conv(
            self.out_dim, (1, 1), kernel_init=self.conv_kernel_init, name="fc2"
        )(x)
        x = nn.Dropout(self.dropout)(x, deterministic)

        return x


class Attention(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(
            self.dim,
            (5, 5),
            1,
            padding=[[2, 2], [2, 2]],
            feature_group_count=self.dim,
            kernel_init=default_init,
            name="conv0",
        )(inputs)
        x = nn.Conv(
            self.dim,
            (7, 7),
            1,
            padding=[[9, 9], [9, 9]],
            feature_group_count=self.dim,
            kernel_dilation=3,
            kernel_init=default_init,
            name="conv_spatial",
        )(x)
        x = nn.Conv(self.dim, (1, 1), kernel_init=default_init, name="conv1")(x)
        return inputs * x


class SpatialAttention(nn.Module):
    d_model: int

    @nn.compact
    def __call__(self, inputs):
        x = nn.Conv(self.d_model, (1, 1), kernel_init=default_init, name="proj_1")(
            inputs
        )
        x = nn.gelu(x)
        x = Attention(self.d_model, name="spatial_gating_unit")(x)
        x = nn.Conv(self.d_model, (1, 1), kernel_init=default_init, name="proj_2")(x)
        return x + inputs


class DropPath(nn.Module):
    """
    Implementation referred from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py
    """

    dropout_prob: float = 0.1
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, input, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )
        if deterministic:
            return input
        keep_prob = 1 - self.dropout_prob
        shape = (input.shape[0],) + (1,) * (input.ndim - 1)
        rng = self.make_rng("drop_path")
        random_tensor = keep_prob + jax.random.uniform(rng, shape)
        random_tensor = jnp.floor(random_tensor)
        return jnp.divide(input, keep_prob) * random_tensor
    

class Block(nn.Module):
    dim: int
    mlp_hidden_dim: int
    droppath: float = 0.0
    dropout: float = 0.0
    init_value: float = 1e-2
    deterministic: Optional[bool] = None

    def scale_init(self, key, shape, value):
        return jnp.full(shape, value)

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        deterministic = nn.merge_param(
            "deterministic", self.deterministic, deterministic
        )

        layer_scale_1 = self.param(
            "layer_scale_1", self.scale_init, (self.dim,), self.init_value
        )
        layer_scale_2 = self.param(
            "layer_scale_2", self.scale_init, (self.dim,), self.init_value
        )

        norm1 = nn.BatchNorm(
            momentum=0.9, epsilon=1e-5, name="norm1", use_running_average=deterministic
        )(inputs)
        attn = SpatialAttention(self.dim, name="attn")(norm1)
        scaled = jnp.expand_dims(jnp.expand_dims(layer_scale_1, 0), 0)
        scaled = scaled * attn
        drop_path = DropPath(self.droppath)(scaled, deterministic)
        inputs = inputs + drop_path

        norm2 = nn.BatchNorm(
            momentum=0.9, epsilon=1e-5, name="norm2", use_running_average=deterministic
        )(inputs)
        mlp = TransformerMLP(
            self.mlp_hidden_dim,
            self.dim,
            self.dropout,
            use_dwconv=True,
            conv_kernel_init=conv_init,
            name="mlp",
        )(norm2, deterministic)

        scaled = jnp.expand_dims(jnp.expand_dims(layer_scale_2, 0), 0) * mlp
        drop_path = DropPath(self.droppath)(scaled, deterministic)
        out = inputs + drop_path
        return out


class VAN(nn.Module):
    embed_dims: Iterable = (64, 128, 256, 512)
    mlp_ratios: Iterable = (4, 4, 4, 4)
    dropout: float = 0.0
    drop_path: float = 0.0
    depths: Iterable = (3, 4, 6, 3)
    num_stages: int = 4
    attach_head: bool = True
    num_classes: int = 1000
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        #dpr = [x.item() for x in jnp.linspace(0, self.drop_path, sum(self.depths))]
        dpr = jnp.linspace(0, self.drop_path, sum(self.depths))
        cur = 0

        x = inputs

        for i in range(self.num_stages):
            x = OverlapPatchEmbed(
                self.embed_dims[i],
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                kernel_init=conv_init,
                name=f"patch_embed{i+1}",
            )(x, deterministic)

            batch, height, width, channels = x.shape

            for j in range(self.depths[i]):
                x = Block(
                    self.embed_dims[i],
                    self.mlp_ratios[i] * self.embed_dims[i],
                    dpr[cur + j],
                    self.dropout,
                    name=f"block{i+1}{j}",
                )(x, deterministic)

            cur += self.depths[i]

            x = x.reshape(x.shape[0], -1, x.shape[-1])
            x = nn.LayerNorm(name=f"norm{i+1}")(x)

            if i != self.num_stages - 1:
                x = x.reshape(batch, height, width, -1)

        x = jnp.mean(x, axis=1)

        if self.attach_head:
            x = nn.Dense(self.num_classes, kernel_init=trunc_norm_init, name="head")(x)

        return x