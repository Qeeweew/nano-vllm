# FILE: ./setup.py
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Determine compiler based on availability
cxx_compiler = "g++"
if os.system("clang++ --version > /dev/null 2>&1") == 0:
    cxx_compiler = "clang++"

os.environ["CXX"] = cxx_compiler

setup(
    name='nanovllm_ext',
    ext_modules=[
        CppExtension(
            'nanovllm_ext',
            ['q8_gemm.cpp'],
            # Use -O3 for optimization, -march=native to enable all available CPU instructions (like AVX2),
            # and -fopenmp for OpenMP, which is used by ATen's parallelism backend on CPU.
            extra_compile_args=['-O3', '-march=native', '-fopenmp'],
            extra_link_args=['-fopenmp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })