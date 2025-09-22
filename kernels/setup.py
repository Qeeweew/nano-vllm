# setup.py
import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# --- Helper functions to find paths ---
def find_cann_path():
    """Find the CANN installation path."""
    ascend_install_path = os.environ.get("ASCEND_INSTALL_PATH")
    if ascend_install_path and os.path.exists(ascend_install_path):
        return ascend_install_path
    ascend_home_path = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home_path and os.path.exists(ascend_home_path):
        return ascend_home_path
    
    default_path = "/usr/local/Ascend/ascend-toolkit/latest"
    if os.path.exists(default_path):
        return default_path
        
    raise RuntimeError("Could not find CANN installation path. Please set ASCEND_INSTALL_PATH or ASCEND_HOME_PATH.")

# --- Configuration ---
CANN_PATH = find_cann_path()
SOC_VERSION = os.environ.get("SOC_VERSION", "Ascend910B2")

ASCENDC_BUILD_DIR = os.path.join('build', 'ascendc')

# Custom build command
class CustomBuild(build_ext):
    def run(self):
        os.makedirs(ASCENDC_BUILD_DIR, exist_ok=True)
        print("-" * 20, "Running CMake for Ascend C Kernels", "-" * 20)
        cmake_args = [
            f"-DSOC_VERSION={SOC_VERSION}",
            f"-DASCEND_CANN_PACKAGE_PATH={CANN_PATH}",
            "-DCMAKE_BUILD_TYPE=" + ("Debug" if self.debug else "Release")
        ]
        subprocess.check_call(
            ["cmake", "-S", ".", "-B", ASCENDC_BUILD_DIR] + cmake_args,
        )
        subprocess.check_call(
            ["cmake", "--build", ASCENDC_BUILD_DIR, "--", "-j"],
        )
        print("-" * 20, "CMake build finished", "-" * 20)
        super().run()

# Find torch and torch_npu paths
try:
    import torch
    import torch_npu
    from torch.utils.cpp_extension import CppExtension
except ImportError:
    raise ImportError("torch and torch_npu must be installed to build this extension.")

torch_npu_dir = os.path.dirname(os.path.abspath(torch_npu.__file__))

ext_modules = [
    CppExtension(
        name='nanovllm_kernels',
        sources=['pybind11.cpp'],
        
        include_dirs=[
            # Path to our generated header
            os.path.join(ASCENDC_BUILD_DIR, 'include', 'kernels'),
            
            # Path for torch_npu headers
            os.path.join(torch_npu_dir, "include"),
            
            # ***** THE FIX IS HERE: Add the main CANN include directory *****
            os.path.join(CANN_PATH, 'runtime', 'include'),
        ],
        
        library_dirs=[
            # Path to our compiled static library
            os.path.join(ASCENDC_BUILD_DIR, 'lib'),
            
            # Other necessary library paths
            os.path.join(torch_npu_dir, "lib"),
            os.path.join(CANN_PATH, 'lib64'),
            os.path.join(CANN_PATH, 'runtime', 'lib64')
        ],
        
        libraries=[
            'kernels',
            'torch_npu',
            'ascendcl'
        ],
        extra_compile_args=['-g', '-D_GLIBCXX_USE_CXX11_ABI=0']
    )
]

setup(
    name='nanovllm_kernels',
    version='0.1.0',
    author='Your Name',
    description='Ascend C operators',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': CustomBuild
    },
    zip_safe=False,
)