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
    input_size = inputs.shape[-1]
    max_size = max(input_size, output_size)
    min_size = min(input_size, output_size)
    if max_size == min_size:
        min_size -= 1
    end_idxs = np.concatenate(
        [np.arange(1, max_size - 1), max_size - np.arange(1, min_size + 1)])
    start_idxs = np.concatenate([
        np.arange(end_idxs.shape[0] + min_size - max_size) % 2,
        np.arange(max_size - min_size)
    ])
    slice_sizes = end_idxs - start_idxs + 1
    if input_size < output_size:
        slice_sizes = slice_sizes[::-1]
        start_idxs = start_idxs[::-1]
        out = jnp.concatenate([
            jnp.zeros((*inputs.shape[:-1], output_size - input_size)), inputs
        ],
                              axis=-1)
    else:
        out = inputs
    thetas = jnp.stack([lax.cos(thetas), lax.sin(thetas)], -1)
    thetas_idxs = np.cumsum(slice_sizes // 2)
    for start_index, slice_size, thetas_index in zip(start_idxs, slice_sizes,
                                                     thetas_idxs):
        # bug: it was not using all thetas, fixed below
        slice_thetas = lax.dynamic_slice_in_dim(thetas, thetas_index-slice_size//2,
                                                slice_size // 2, 0)
        slice_out = lax.dynamic_slice_in_dim(out, start_index, slice_size, -1)
        slice_out = lax.reshape(slice_out,
                                (slice_out.shape[0], *slice_thetas.shape))
        # bug: wrong negative sign, fixed below 
        slice_mat = jnp.array([
            [slice_thetas[:, 0], slice_thetas[:, 1]],
            [-slice_thetas[:, 1], slice_thetas[:, 0]],
        ]).transpose(2, 0, 1)
        slice_res = lax.batch_matmul(slice_out.transpose(1, 0, 2), slice_mat)
        slice_res = slice_res.transpose(1, 0, 2)
        slice_res = lax.reshape(slice_res, (slice_res.shape[0], slice_size))
        out = lax.dynamic_update_slice_in_dim(out, slice_res, start_index, -1)
    if input_size > output_size:
        out = out[:, -output_size:]
    return out
