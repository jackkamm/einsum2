#!/usr/bin/env python

from distutils.core import setup

setup(name='einsum2',
      version='0.1',
      description='Parallel einsum products of 2 tensors',
      author='Jack Kamm',
      author_email='jackkamm@gmail.com',
      packages=['einsum2'],
      install_requires=['numpy'],
      keywords=['Einstein summation', 'linear algebra', 'tensors'],
      url='https://github.com/jackkamm/einsum2',
      )
