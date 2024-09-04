import jax
import jax.numpy as jnp
import warnings

jax.jit
def inverse_discrete_cos_transform(X):
    # here X is a r x n array where r is the size of the batch
    # and n is the dimension of the vectors
    n = X.shape[-1]
    r = X.shape[0]
    X_v = X / 2
    X_v = X_v.at[:, 0].multiply(jnp.sqrt(n) * 2)
    X_v = X_v.at[:, 1:].multiply(jnp.sqrt(n / 2) * 2)

    k = jnp.arange(n) * jnp.pi / (2 * n)

    W_r = jnp.cos(k)
    W_i = jnp.sin(k)

    V_t_r = X_v
    V_t_i = jnp.concatenate([X_v[:, :1] * 0, -jnp.flip(X_v, [1])[:, :-1]], axis=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    # This older implementation let CUDA FFT prchestrate the batched
    # execution of the FFT, and that has a limit size of 2 ** 27.
    # v = jnp.fft.irfft(jax.lax.complex(V_r, V_i), n=V_i.shape[1], axis=1)
    # 
    # The following implementation behaves differently if a batch is give.
    # First, it raises a warning, and then it runs the batch sequentially.
    # I this sequential implementation becomes intended behavior, then
    # we should re-implement with a jax-native sequential construct.
    if r == 1:
        v = jnp.fft.irfft(jax.lax.complex(V_r, V_i), n=V_i.shape[1], axis=1)
    else:
        warnings.warn(f'''We are using a very slow sequential implementation of the FFT here. (size {V_i.shape[0]})
                      The program was not expected to enter this branch.
                      If the program entered this branch, it is becasue a CUDA-batched version of
                      FFT is being executed.''')
        v = jnp.array([jnp.fft.irfft(jax.lax.complex(V_r[i], V_i[i]), n=V_i.shape[1]) for i in range(V_i.shape[0])])


    
    x = v[:, :n - (n // 2)]
    y = jnp.flip(v, [1])[:, :n // 2]

    if n % 2:
        y = jnp.concatenate([y, jnp.zeros((v.shape[0],1))], axis=1)
    
    x = jnp.expand_dims(x, 2)
    y = jnp.expand_dims(y, 2)
    z = jnp.concatenate([x, y], axis=2)
    z = jnp.reshape(z, (v.shape[0], x.shape[1] + y.shape[1]))
    
    if n % 2:
        z = z[:, :-1]
    return z

# This sketch calls jnp.fft.irfft, that in turns calls CUDA FFT (CUFFT)
# CUFFT has a limit on the size of the batched version of it operator
# that is 2 ** 27. See the following for reference
# https://stackoverflow.com/questions/13187443/nvidia-cufft-limit-on-sizes-and-batches-for-fft-with-scikits-cuda
#
# Thus, I gave an alternate implementation of inverse_discrete_cos_transform.
class SRFT_sketch:
    def __init__(self, key, dim_in, dim_out, padding=0):
        # pad dimension with a zero
        self.padding = padding
        self.dim_in = dim_in + padding
        self.dim_out = dim_out
        self.D = jax.random.rademacher(key, (self.dim_in, 1))
        self.P = jax.random.choice(key, jnp.array(range(self.dim_in)), shape=(self.dim_out,))

    def __matmul__(self, B):
        if B.ndim == 1:
            # pad dimension with padding zeros
            B = jnp.concatenate([B, jnp.zeros(self.padding)])
            ans = self.D[:, 0] * B
            ans = ans[:, None]
        elif B.ndim == 2:
            # pad dimension with padding zeros
            B = jnp.concatenate([B, jnp.zeros((self.padding, B.shape[1]))])    
            ans = self.D * B
        else:
            raise NotImplementedError
        ans = inverse_discrete_cos_transform(ans.T).T
        ans = ans[self.P, :] * jnp.sqrt(self.dim_in / self.dim_out)
        if B.ndim == 1:
            ans = ans[:, 0]
        
        return ans
    



def get_biggest_prime_factor(n):
    last_prime = 1
    while n % 2 == 0:
        last_prime = 2
        n = n // 2
    for i in range(3,int(n**0.5)+1,2):
        while n % i== 0:
            last_prime = i
            n = n // i
    if n > 2:
        last_prime = n
    return last_prime

def get_smallest_greater_value_with_good_factorization(n, max_prime_allowed=127):
    while get_biggest_prime_factor(n) > max_prime_allowed:
        n += 1
    return n

def get_optimal_padding(n):
    new_value = get_smallest_greater_value_with_good_factorization(n)
    return new_value - n