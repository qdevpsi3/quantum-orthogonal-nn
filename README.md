<h1 align="center" style="margin-top: 0px;"> <b>Classical and Quantum Algorithms for Orthogonal Neural Networks</b></h1>
<div align="center" >

[![paper](https://img.shields.io/static/v1.svg?label=Paper&message=arXiv:2106.07198&color=b31b1b)](https://arxiv.org/abs/2106.07198)
[![packages](https://img.shields.io/static/v1.svg?label=Made%20with&message=JAX&color=27A59A)](https://github.com/google/jax)
[![license](https://img.shields.io/static/v1.svg?label=License&message=GPL%20v3.0&color=green)](https://www.gnu.org/licenses/gpl-3.0.html)
[![exp](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/qdevpsi3/quantum-orthogonal-nn/blob/main/mnist_classifier.ipynb)
</div>

## **Description**
This repository contains an <ins>unofficial</ins> implementation of the <ins>Quantum-inspired Orthogonal Neural Network</ins> and its application to the  <ins>MNIST</ins> dataset classification problem as in :

- Paper : **Classical and Quantum Algorithms for Orthogonal Neural Networks**
- Authors : **Kerenidis, Landman and Mathur**
- Date : **2021**

## **Details**

The function `orthogonal_network_builder` returns a trainable orthogonal network ( using [dm-haiku](https://github.com/deepmind/dm-haiku)) with the following set of parameters :

| Parameters | Description|
|:-:|:-:|
| `output_sizes` | Sequence of layer sizes. |
| `with_bias` | Whether or not to apply a bias in each layer. |
| `activation` | Activation function to apply between layers. |
| `activate_final` | Whether or not to activate the final layer. |
| `normalize` | Whether or not to normalize layer inputs. |

## **Usage**
To run the experiments :

- Option 1 : Open in [Colab](https://colab.research.google.com/github/qdevpsi3/quantum-orthogonal-nn/blob/main/mnist_classifier.ipynb).
- Option 2 : Run on local machine. First, you need to clone this repository and execute the following commands to install the required packages :
```
$ cd quantum-orthogonal-nn
$ pip install -r requirements.txt
```
You can run an experiment using the following command :
```
$ python mnist_classifier.py
```

By default, the hyperparameters are set to :
```python
seed = 123
batch_size = 50
n_components = 8
digits = [6,9]
output_sizes = [4,2]
with_bias = False
activation = jax.nn.selu
activate_final = False
normalize = False
learning_rate = 0.001
train_steps = 5000
```