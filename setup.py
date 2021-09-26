from setuptools import find_packages, setup

setup(name='orthax',
      version='0.0.1',
      install_requires=['numpy', 'jax', 'dm-haiku'],
      packages=find_packages('src'),
      package_dir={'': 'src'})
