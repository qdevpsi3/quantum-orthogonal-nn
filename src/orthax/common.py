from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np


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


def _reshape_thetas(thetas, parallel_wires):
    lengths = [0] + list(map(len, parallel_wires))
    idxs = np.cumsum(lengths)
    thetas = [thetas[idxs[i]:idxs[i + 1]] for i in range(idxs.shape[0] - 1)]
    return thetas


def _apply_orthogonal_(thetas, inputs, parallel_wires):
    cos_thetas = _reshape_thetas(jnp.cos(thetas), parallel_wires)
    sin_thetas = _reshape_thetas(jnp.sin(thetas), parallel_wires)
    out = inputs
    for p_wires, cos_t, sin_t in zip(parallel_wires, cos_thetas, sin_thetas):
        unitaries = jnp.stack(
            [jnp.stack([cos_t, sin_t]),
             jnp.stack([-sin_t, cos_t])])
        unitaries = unitaries.transpose(2, 0, 1)
        states = jnp.stack([out[:, p_wires[:, 0]], out[:, p_wires[:, 1]]], -1)
        states = states.transpose(1, 0, 2)
        results = jnp.matmul(states, unitaries).transpose(1, 0, 2)
        out = jax.ops.index_update(out, jax.ops.index[:, p_wires[:, 0]],
                                   results[:, :, 0])
        out = jax.ops.index_update(out, jax.ops.index[:, p_wires[:, 1]],
                                   results[:, :, 1])
    return out
