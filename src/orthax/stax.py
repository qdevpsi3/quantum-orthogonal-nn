from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from orthax.common import apply_orthogonal

__all__ = ['OrthogonalDense']


def t_init(key, shape, dtype=jnp.float_):
    return np.pi * (jax.random.uniform(key, shape, dtype) * 2 - 1)


def b_init(key, shape, dtype=jnp.float_):
    return jnp.zeros(shape, jax.dtypes.canonicalize_dtype(dtype))


def OrthogonalDense(out_dim, t_init=t_init, b_init=b_init):
    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim, )
        t_key, b_key = jax.random.split(rng)
        t_shape = ((2 * input_shape[-1] - out_dim - 1) * out_dim // 2, )
        t = t_init(t_key, t_shape)
        b = b_init(b_key, (out_dim, ))
        return output_shape, (t, b)

    def apply_fun(params, inputs, **kwargs):
        t, b = params
        out = apply_orthogonal(t, inputs, out_dim)
        out = out[:, -out_dim:] + b
        return out

    return init_fun, apply_fun
