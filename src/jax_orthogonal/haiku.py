from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

import haiku as hk


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

        list_wires = _get_orthogonal_wires_(input_size, output_size)
        parallel_wires = _get_parallel_wires_(list_wires)
        parallel_wires = list(map(jnp.array, parallel_wires))

        for i, p_wires in enumerate(parallel_wires):
            thetas = hk.get_parameter("thetas_depth_{}".format(i),
                                      shape=[p_wires.shape[0]],
                                      dtype=out.dtype,
                                      init=self.t_init)
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

        out = out[:, -output_size:]

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
