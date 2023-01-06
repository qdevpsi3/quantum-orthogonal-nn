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
        Implementaion of Figure 12 of https://arxiv.org/abs/2106.07198
    """
    input_size = inputs.shape[-1]
    max_size = max(input_size, output_size)
    min_size = min(input_size, output_size)
    
    if max_size == min_size:
        min_size -= 1
    
    slice_end_idxs = np.concatenate([
        np.arange(1, max_size - 1), 
        max_size - np.arange(1, min_size + 1)
    ])
    print('debug', 'slice_end_idxs:\n', slice_end_idxs)
    
    
    slice_start_idxs = np.concatenate([
        np.arange(slice_end_idxs.shape[0] + min_size - max_size) % 2, # [0, 1, 0, 1, ...]
        np.arange(max_size - min_size)
    ])

    slice_sizes = slice_end_idxs - slice_start_idxs + 1
    
    if input_size < output_size:
        slice_start_idxs = slice_start_idxs[::-1]
        slice_sizes = slice_sizes[::-1]
        
        # add zeros to inputs
        out = jnp.concatenate([
            jnp.zeros((*inputs.shape[:-1], output_size - input_size)), inputs
        ], axis=-1)  # (batch_size, output_size)

    else:
        out = inputs


    print('debug', 'slice_start_idxs:\n', slice_start_idxs)
    print('debug', 'slice_sizes:\n', slice_sizes)

    thetas = jnp.stack([lax.cos(thetas), lax.sin(thetas)], -1) # (len(thetas), 2)
    thetas_idxs = np.cumsum(slice_sizes // 2) 

    print('debug', 'thetas_idxs:\n', thetas_idxs)
    print('debug', 'thetas:\n', '     cosθ        sinθ')
    print(thetas)
    print()

    cnt = 0
    print('debug', cnt, 'out:\n', out)
    for start_index, slice_size, thetas_index in zip(slice_start_idxs, slice_sizes, thetas_idxs):
        
        # bug: it was not using all thetas, fixed below
        slice_thetas = lax.dynamic_slice_in_dim(thetas, thetas_index-slice_size//2,
                                                slice_size//2, 0)  # (n_RBS, 2)
        print('debug', cnt, 'slice_thetas:\n', slice_thetas)
       
        slice_out = lax.dynamic_slice_in_dim(out, start_index, slice_size, -1) # (batch_size, slice_size)

        print('debug', cnt, 'slice_out:\n', slice_out)
        
        # distribute input features to each RBS
        slice_out = lax.reshape(slice_out,
                                (slice_out.shape[0], *slice_thetas.shape)) # (batch_size, n_RBS, 2)
        
        print('debug', cnt, 'slice_out_after_reshape:\n', slice_out)
        slice_out = slice_out.transpose(1, 0, 2) # (n_RBS, batch_size, 2)
        
        # bug: wrong negative sign, fixed below 
        slice_mat = jnp.array([
            [slice_thetas[:, 0], slice_thetas[:, 1]],
            [-slice_thetas[:, 1], slice_thetas[:, 0]],
        ]).transpose(2, 0, 1)  # (n_RBS, 2, 2)
        print('debug', cnt, 'slice_mat:\n', slice_mat)
        print()

        slice_res = lax.batch_matmul(slice_out, slice_mat)  # (n_RBS, batch_size, 2)
        slice_res = slice_res.transpose(1, 0, 2)            # (batch_size, n_RBS, 2)
        slice_res = lax.reshape(slice_res, (slice_res.shape[0], slice_size)) # (batch_size, slice_size=n_RBS*2)
        out = lax.dynamic_update_slice_in_dim(out, slice_res, start_index, -1)

        cnt +=1
        print('debug', cnt, 'out:\n', out)


    if input_size > output_size:
        out = out[:, -output_size:]
    
    return out





if __name__ == "__main__":

    # # n=d=5, n_params = (2 * n - d - 1) * d // 2 = 10
    # thetas = jnp.array([1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])  
    # inputs = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]) #  (batch_size=1, 5)
    # output_size = 5

    # # n=4 d=2, n_params = (2 * n - d - 1) * d // 2 = 5
    # thetas = jnp.array([1., 2., 3., 4., 5., 6.])  
    # inputs = np.array([[0.1, 0.2, 0.3, 0.4]]) #  (batch_size=1, 4)
    # output_size = 2

    # # n=8 d=4, n_params = (2 * n - d - 1) * d // 2 = 22
    # thetas = jnp.arange(22)+1.0
    # inputs = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]) #  (batch_size=1, 8)
    # output_size = 4

    # n=4 d=8, n_params = (2 * d - n - 1) * n // 2 = 22
    thetas = jnp.arange(22)+1.0
    inputs = np.array([[0.1, 0.2, 0.3, 0.4]]) #  (batch_size=1, 4)
    output_size = 8

    out = apply_orthogonal(thetas, inputs, output_size)


