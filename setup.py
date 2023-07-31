from setuptools import setup, find_packages

setup(name='tempo_models',
      version='0.1',
      description='Implementations of various temporal attention mechanisms.',
      packages=find_packages(),
      package_dir={'':'src'}
     )