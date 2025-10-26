import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_compiler = os.environ.get("CXX", "g++")
os.environ["CXX"] = cxx_compiler

setup(
    name='nanovllm_ext',
    ext_modules=[
        CUDAExtension(
            'nanovllm_ext',
            [
                'q8_gemm.cpp',
                'moe_infer.cpp',
                'pybind.cpp'
            ],
            define_macros=[('WITH_CUDA', None), ('WITH_NUMA', None)],
            extra_compile_args={
                'cxx': ['-O3', '-ffast-math', '-Wall', '-march=native', '-fopenmp'],
                'nvcc': ['-O3', '--use_fast_math']
            },
            extra_link_args=['-fopenmp', '-lnuma'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })