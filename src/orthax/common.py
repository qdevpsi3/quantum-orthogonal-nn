import jax
import jax.numpy as jnp
import numpy as np


def _get_slices_(input_size, output_size):
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
    return start_idxs, slice_sizes


def _apply_orthogonal_(thetas, inputs, output_size):
    input_size, out = inputs.shape[-1], inputs
    start_idxs, slice_sizes = _get_slices_(input_size, output_size)
    thetas = jnp.stack([jax.lax.cos(thetas), jax.lax.sin(thetas)], -1)
    start_thetas = 0
    for start_index, slice_size in zip(start_idxs, slice_sizes):
        slice_thetas = jax.lax.dynamic_slice_in_dim(thetas, start_thetas,
                                                    slice_size // 2, 0)
        slice_out = jax.lax.dynamic_slice_in_dim(out, start_index, slice_size,
                                                 -1)
        slice_out = jax.lax.reshape(slice_out,
                                    (slice_out.shape[0], *slice_thetas.shape))
        slice_res = jax.vmap(jax.lax.mul, (0, None))(slice_out, slice_thetas)
        slice_res = jax.vmap(jax.lax.batch_matmul,
                             (0, None))(slice_res,
                                        jnp.array([[1., -1.], [1., 1.]]))
        slice_res = jax.lax.reshape(slice_res,
                                    (slice_res.shape[0], slice_size))
        out = jax.lax.dynamic_update_slice_in_dim(out, slice_res, start_index,
                                                  -1)
        start_thetas += slice_size // 2
    return out
