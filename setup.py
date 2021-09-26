from setuptools import find_packages, setup

setup(name='jax_orthogonal',
      version='0.0.1',
      install_requires=['numpy', 'jax', 'dm-haiku'],
      packages=find_packages('src'),
      package_dir={'': 'src'})
