import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk
from orthax.common import apply_orthogonal

__all__ = ['OrthogonalLinear', 'OrthogonalMLP']


class OrthogonalLinear(hk.Module):
    def __init__(self,
                 output_size,
                 normalize_inputs=False,
                 with_bias=True,
                 t_init=None,
                 b_init=None,
                 name=None):

        super().__init__(name=name)
        self.output_size = output_size
        self.normalize_inputs = normalize_inputs
        self.with_bias = with_bias
        self.t_init = t_init or hk.initializers.RandomUniform(minval=-np.pi,
                                                              maxval=np.pi)
        self.b_init = b_init or hk.initializers.Constant(0.)

    def __call__(self, inputs):
        input_size = inputs.shape[-1]
        output_size = self.output_size

        if self.normalize_inputs:
            norm = jnp.linalg.norm(inputs, axis=1)[..., None]
            out = inputs / jax.lax.stop_gradient(norm)
        else:
            out = inputs
        max_size = max(input_size, output_size)
        min_size = min(input_size, output_size)
        thetas_shape = ((2 * max_size - min_size - 1) * min_size // 2, )
        thetas = hk.get_parameter("thetas",
                                  shape=thetas_shape,
                                  dtype=out.dtype,
                                  init=self.t_init)

        out = apply_orthogonal(thetas, out, output_size)

        if self.with_bias:
            b = hk.get_parameter("bias",
                                 shape=[output_size],
                                 dtype=out.dtype,
                                 init=self.b_init)
            out += b
        return out


class OrthogonalMLP(hk.Module):
    def __init__(self,
                 output_sizes,
                 normalize_inputs=False,
                 with_bias=True,
                 t_init=None,
                 b_init=None,
                 activation=jax.nn.sigmoid,
                 activate_final=False,
                 name=None):

        super().__init__(name=name)

        self.with_bias = with_bias
        self.normalize_inputs = normalize_inputs
        self.t_init = t_init
        self.b_init = b_init
        self.activation = activation
        self.activate_final = activate_final
        layers = []
        output_sizes = tuple(output_sizes)
        for index, output_size in enumerate(output_sizes):
            layers.append(
                OrthogonalLinear(output_size=output_size,
                                 normalize_inputs=normalize_inputs,
                                 t_init=t_init,
                                 b_init=b_init,
                                 with_bias=with_bias,
                                 name='orthogonal_linear_{}'.format(index)))
        self.layers = tuple(layers)
        self.output_size = output_sizes[-1] if output_sizes else None

    def __call__(self, inputs):
        num_layers = len(self.layers)

        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (num_layers - 1) or self.activate_final:
                out = self.activation(out)

        return out
