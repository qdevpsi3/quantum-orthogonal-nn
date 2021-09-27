from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

from .common import (_apply_orthogonal_, _get_orthogonal_wires_,
                     _get_parallel_wires_)

__all__ = ['OrthogonalDense']


def t_init(key, shape, dtype=jnp.float_):
    return np.pi * (jax.random.uniform(key, shape, dtype) * 2 - 1)


def b_init(key, shape, dtype=jnp.float_):
    return jnp.zeros(shape, jax.dtypes.canonicalize_dtype(dtype))


def OrthogonalDense(in_dim, out_dim, t_init=t_init, b_init=b_init):
    list_wires = _get_orthogonal_wires_(in_dim, out_dim)
    parallel_wires = _get_parallel_wires_(list_wires)
    parallel_wires = list(map(jnp.array, parallel_wires))

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim, )
        t_key, b_key = jax.random.split(rng)
        t = t_init(t_key, (len(list_wires), ))
        b = b_init(b_key, (out_dim, ))
        return output_shape, (t, b)

    def apply_fun(params, inputs, **kwargs):
        t, b = params
        out = _apply_orthogonal_(t, inputs, parallel_wires)
        out = out[:, -out_dim:] + b
        return out

    return init_fun, apply_fun
