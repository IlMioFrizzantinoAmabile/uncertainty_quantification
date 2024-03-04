import jax
import numpy as np
import jax.numpy as jnp
from jax.experimental import sparse as jsparse


class Dense_sketch:
    def __init__(self, key, dim_in, dim_out, density=None):       
        if density is None:
            density = 1 / np.sqrt(dim_in) # use numpy to keep density static
        val =  np.sqrt(1 / density) / np.sqrt(dim_out)

        nse = int(dim_in * dim_out * density)
        key_i, key_j, key_val = jax.random.split(key, 3)
        ind_i = jax.random.randint(key_i, (nse, ), 0, dim_out)
        ind_j = jax.random.randint(key_j, (nse, ), 0, dim_in)
        indices = jnp.stack((ind_i, ind_j)).T
        values = val * jax.random.rademacher(key_val, (nse, ))
        self.matrix = jsparse.BCOO((values, indices), shape=(dim_out, dim_in))
    def __matmul__(self, v):
        return self.matrix @ v
    
