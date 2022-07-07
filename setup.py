#!/usr/bin/env python
import os
import pathlib
from setuptools import setup, find_packages, find_namespace_packages

import sys

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
with open(os.path.join(HERE, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

# list of requirements for core packages
fuse_requirements = []
with open(os.path.join(HERE, 'fuse/requirements.txt'), 'r') as fh:
    for line in fh:
        if not line.startswith('#'):
            fuse_requirements.append(line.strip())

# list of requirements for fuseimg
fuseimg_requirements = []
with open(os.path.join(HERE, 'fuseimg/requirements.txt'), 'r') as fh:
    for line in fh:
        if not line.startswith('#'):
            fuseimg_requirements.append(line.strip())

# all extra requires
all_requirements = fuseimg_requirements
# version
from fuse.version import __version__
version = __version__

setup(name='fuse-med-ml',
      version=version,
      description='Open-source PyTorch based framework designed to facilitate deep learning R&D in medical imaging',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/IBM/fuse-med-ml/',
      author='IBM Research - Machine Learning for Healthcare and Life Sciences',
      author_email='moshiko.raboh@ibm.com',
      packages=find_namespace_packages(),
      license='Apache License 2.0',
      install_requires=fuse_requirements,
      extras_requires={"fuseimg": fuseimg_requirements, "all": all_requirements},
      )
