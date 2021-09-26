from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np


def t_init(key, shape, dtype=jnp.float_):
    return np.pi * (jax.random.uniform(key, shape, dtype) * 2 - 1)


def b_init(key, shape, dtype=jnp.float_):
    return jnp.zeros(shape, jax.dtypes.canonicalize_dtype(dtype))


def _get_parallel_wires_(list_wires, depth=0):
    pos = defaultdict(lambda: depth)
    p_wires = [[]]
    for wires in list_wires:
        if type(wires) == int:
            wires = [wires]
        g_pos = max([pos[wire] for wire in wires])
        if len(p_wires) - 1 < g_pos:
            p_wires.append([])
        p_wires[g_pos].append(wires)
        for wire in wires:
            pos[wire] = g_pos + 1
    return p_wires


def _get_orthogonal_wires_(input_size, output_size):
    list_wires = [(j - 1, j) for i in range(1, input_size)
                  for j in range(i, max(0, i - output_size), -1)]
    return list_wires


def OrthogonalDense(in_dim, out_dim, t_init=t_init, b_init=b_init):
    list_wires = _get_orthogonal_wires_(in_dim, out_dim)
    parallel_wires = _get_parallel_wires_(list_wires)
    parallel_wires = list(map(jnp.array, parallel_wires))

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim, )
        params = []
        for p_wires in parallel_wires:
            rng, init_key = jax.random.split(rng)
            params.append(t_init(init_key, (p_wires.shape[0], )))
        params.append(b_init(rng, (out_dim, )))
        return output_shape, tuple(params)

    def apply_fun(params, inputs, **kwargs):
        t, b = params[:-1], params[-1]
        out = inputs
        for p_wires, thetas in zip(parallel_wires, t):
            cos_t, sin_t = jnp.cos(thetas), jnp.sin(thetas)
            unitaries = jnp.stack(
                [jnp.stack([cos_t, sin_t]),
                 jnp.stack([-sin_t, cos_t])])
            unitaries = unitaries.transpose(2, 0, 1)
            states = jnp.stack([out[:, p_wires[:, 0]], out[:, p_wires[:, 1]]],
                               -1)
            states = states.transpose(1, 0, 2)
            results = jax.vmap(jnp.dot)(states, unitaries).transpose(1, 0, 2)
            out = jax.ops.index_update(out, jax.ops.index[:, p_wires[:, 0]],
                                       results[:, :, 0])
            out = jax.ops.index_update(out, jax.ops.index[:, p_wires[:, 1]],
                                       results[:, :, 1])

        out = out[:, -out_dim:] + b
        return out

    return init_fun, apply_fun
