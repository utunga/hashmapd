from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(name='hashmapd',
    version='1.0',
    description='Generic hashmapd code',
    packages = ['hashmapd'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("hashmapd.compiled", ["hashmapd/compiled.pyx"], 
            include_dirs=[numpy.get_include()]),            
            ],
)

