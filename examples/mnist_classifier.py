import array
import gzip
import os
import struct
import sys
import urllib.request
from os import path

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.decomposition import PCA

import orthax





def mnist_raw():
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"

    _DATA = "/tmp/"

    def _download(url, filename):
        """Download a url to a file in the JAX data temp directory."""

        if not path.exists(_DATA):
            os.makedirs(_DATA)
        out_file = path.join(_DATA, filename)
        if not path.isfile(out_file):
            urllib.request.urlretrieve(url, out_file)
            print("downloaded {} to {}".format(url, _DATA))

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            _ = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            _, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()),
                            dtype=np.uint8).reshape(num_data, rows, cols)

    for filename in [
            "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]:
        _download(base_url + filename, filename)

    train_images = parse_images(path.join(_DATA, "train-images-idx3-ubyte.gz"))
    train_labels = parse_labels(path.join(_DATA, "train-labels-idx1-ubyte.gz"))
    test_images = parse_images(path.join(_DATA, "t10k-images-idx3-ubyte.gz"))
    test_labels = parse_labels(path.join(_DATA, "t10k-labels-idx1-ubyte.gz"))

    return train_images, train_labels, test_images, test_labels


def mnist(digits=None):
    def _maybe_filter(images, labels, digits):
        mask = np.isin(labels, digits)
        return images[mask], labels[mask]

    def _partial_flatten(x):
        return np.reshape(x, (x.shape[0], -1))

    def _one_hot(x, d, dtype=np.float32):
        return np.array(x[:, None] == d, dtype)

    train_images, train_labels, test_images, test_labels = mnist_raw()
    if digits is not None:
        train_images, train_labels = _maybe_filter(train_images, train_labels,
                                                   digits)
        test_images, test_labels = _maybe_filter(test_images, test_labels,
                                                 digits)
        train_labels = _one_hot(train_labels, np.array(digits))
        test_labels = _one_hot(test_labels, np.array(digits))
    else:
        train_labels = _one_hot(train_labels, np.arange(10))
        test_labels = _one_hot(test_labels, np.arange(10))

    train_images = _partial_flatten(train_images) / np.float32(255.)
    test_images = _partial_flatten(test_images) / np.float32(255.)

    return train_images, train_labels, test_images, test_labels


def pca(train_x, test_x, n_components=8):
    decomposition = PCA(n_components).fit(train_x)
    train_x = decomposition.transform(train_x)
    test_x = decomposition.transform(test_x)
    return train_x, test_x







# training parameters
seed = 123
batch_size = 50
n_components = 8
digits = [6,9]
learning_rate = 0.001
train_steps = 5000

# network parameters
output_sizes = [4,2]
normalize_inputs = False
with_bias = True
t_init = hk.initializers.RandomUniform(minval=-np.pi, maxval=np.pi)
b_init = hk.initializers.Constant(0.)
activation = jax.nn.sigmoid
activate_final = False









# set random state
random_state = np.random.RandomState(seed)
rng_key = jax.random.PRNGKey(
    random_state.randint(-sys.maxsize - 1, sys.maxsize + 1,
                            dtype=np.int64))

# load data
train_images, train_labels, test_images, test_labels = jax.device_put(mnist(digits))
train_features, test_features = pca(train_images, test_images,
                                    n_components)

# build batch iterator
num_train = train_images.shape[0]
num_complete_batches, leftover = divmod(num_train, batch_size)
num_batches = num_complete_batches + bool(leftover)

def data_stream(batch_size):
    while True:
        perm = random_state.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield train_features[batch_idx], train_labels[batch_idx]

batches = iter(data_stream(batch_size))

# build network
def orthogonal_net(x):
    module = orthax.haiku.OrthogonalMLP(output_sizes, 
                           normalize_inputs,
                           with_bias,
                           t_init,
                           b_init,
                           activation, 
                           activate_final)
    return module(x)
net = hk.without_apply_rng(hk.transform(orthogonal_net))
params = avg_params = net.init(rng_key, next(batches)[0])

# build optimizer
opt = optax.rmsprop(learning_rate)
opt_state = opt.init(params)

# build model
def loss(params, features, labels):
    logits = net.apply(params, features)
    l2_loss = 0.5 * sum(
        jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))
    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]
    return softmax_xent + 1e-4 * l2_loss

@jax.jit
def accuracy(params, features, labels):
    predictions = net.apply(params, features)
    return jnp.mean(
        jnp.argmax(predictions, axis=1) == jnp.argmax(labels, axis=1))

@jax.jit
def update(params, opt_state, features, labels):
    grads = jax.grad(loss)(params, features, labels)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

@jax.jit
def ema_update(params, avg_params):
    return optax.incremental_update(params, avg_params, step_size=0.001)






# train/eval loop.
for step in range(train_steps):
    batch_features, batch_labels = next(batches)
    if step % 100 == 0:
        # evaluate classification accuracy on train & test sets.
        train_accuracy = accuracy(avg_params, batch_features, batch_labels)
        test_accuracy = accuracy(avg_params, test_features, test_labels)
        train_accuracy, test_accuracy = jax.device_get(
            (train_accuracy, test_accuracy))
        print(f"[Step {step}] Train / Test accuracy: "
                f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

    # update params
    params, opt_state = update(params, opt_state, batch_features, batch_labels)
    avg_params = ema_update(params, avg_params)
