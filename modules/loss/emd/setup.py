import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='emd',
    ext_modules=[
        CUDAExtension('emd', [
            'emd.cpp',
            'emd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

os.system('cp build/lib.linux-x86_64-3.6/emd.cpython-36m-x86_64-linux-gnu.so .')
