from re import X

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp

__all__ = ['apply_orthogonal']


def apply_orthogonal(thetas, inputs, output_size):
    """ Applies a sequence of orthogonal transformations to a sequence of inputs.
        <thetas> is a sequence of scalars, with length (2 * n - d - 1) * d // 2.
        <inputs> is a sequence of vectors, each of shape n.
        <output_size> is the desired output size, equal to d.
    """
    input_size, out = inputs.shape[-1], inputs
    if input_size == output_size:
        output_size -= 1
    end_idxs = np.concatenate([
        np.arange(1, input_size - 1),
        input_size - np.arange(1, output_size + 1)
    ])
    start_idxs = np.concatenate([
        np.arange(end_idxs.shape[0] + output_size - input_size) % 2,
        np.arange(input_size - output_size)
    ])
    slice_sizes = end_idxs - start_idxs + 1
    thetas = jnp.stack([lax.cos(thetas), lax.sin(thetas)], -1)
    thetas_idxs = np.cumsum(slice_sizes // 2)
    for start_index, slice_size, thetas_index in zip(start_idxs, slice_sizes,
                                                     thetas_idxs):
        slice_thetas = lax.dynamic_slice_in_dim(thetas, thetas_index,
                                                slice_size // 2, 0)
        slice_out = lax.dynamic_slice_in_dim(out, start_index, slice_size, -1)
        slice_out = lax.reshape(slice_out,
                                (slice_out.shape[0], *slice_thetas.shape))
        slice_res = jax.vmap(lax.mul, (0, None))(slice_out, slice_thetas)
        slice_res = jax.vmap(lax.batch_matmul, (0, None))(slice_res,
                                                          jnp.array([
                                                              [1., -1.],
                                                              [1., 1.],
                                                          ]))
        slice_res = lax.reshape(slice_res, (slice_res.shape[0], slice_size))
        out = lax.dynamic_update_slice_in_dim(out, slice_res, start_index, -1)
    return out
