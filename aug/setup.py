from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
	name='ctoolkits',
	description='Augmentation for Image data.',
	long_description='Toolkits that provide several functions for data augment.'
	version='1.2.1',
	author='Shawn Hu',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension(
        "ctoolkits",
        ["ctoolkits.pyx"],
        include_dirs=[numpy.get_include()]
    )]
)
