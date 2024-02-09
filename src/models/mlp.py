import jax.numpy as jnp
from jax import nn as jnn
from flax import linen as nn

class MLP(nn.Module): 
    act_fn : callable
    output_dim: int
    hidden_dim: int = 64
    num_layers: int = 3

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = self.act_fn(x)
        x = nn.Dense(self.output_dim)(x) 
        return x