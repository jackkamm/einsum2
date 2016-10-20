#!/usr/bin/env python

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

extensions = [Extension("einsum2.parallel_matmul",
                        sources=["einsum2/parallel_matmul.pyx"],
                        include_dirs=[numpy.get_include()])]

setup(name='einsum2',
      version='0.1',
      description='Parallel einsum products of 2 tensors',
      author='Jack Kamm',
      author_email='jackkamm@gmail.com',
      packages=['einsum2'],
      install_requires=['numpy>=1.9.0'],
      keywords=['Einstein summation', 'linear algebra', 'tensors'],
      url='https://github.com/jackkamm/einsum2',
      ext_modules=cythonize(extensions),
      )
