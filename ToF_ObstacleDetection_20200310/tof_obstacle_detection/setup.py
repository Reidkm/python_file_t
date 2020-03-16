# import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

# from distutils.core import setup, Extension
# from Cython.Build import cythonize
# import numpy
#
# from distutils.core import setup
# from Cython.Build import cythonize
# from Cython.Compiler import Options
#
# setup(name='TOF Utils',
#       ext_modules=cythonize("tof_utils.pyx"),
#       include_dirs=[numpy.get_include()])


from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize(Extension(
    'c_kik_pcl',
    sources=['c_kik_pcl.pyx'],
    language='c',
    include_dirs=[numpy.get_include()],
    library_dirs=[],
    libraries=[],
    extra_compile_args=[],
    extra_link_args=[]
)))
