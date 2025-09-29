# setup.py
import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# --- Helper functions to find paths ---
def find_cann_path():
    """Find the CANN installation path."""
    for env_var in ["ASCEND_TOOLKIT_HOME", "ASCEND_INSTALL_PATH", "ASCEND_HOME_PATH"]:
        path = os.environ.get(env_var)
        if path and os.path.exists(path):
            print(f"Found CANN path at: {path} (from ${env_var})")
            return path
    
    default_path = "/usr/local/Ascend/ascend-toolkit/latest"
    if os.path.exists(default_path):
        print(f"Found CANN path at default location: {default_path}")
        return default_path
        
    raise RuntimeError("Could not find CANN installation path. Please set ASCEND_TOOLKIT_HOME.")

# --- Configuration ---
CANN_PATH = find_cann_path()
SOC_VERSION = os.environ.get("SOC_VERSION", "Ascend910B2")
ASCENDC_BUILD_DIR = os.path.join('build', 'ascendc')

# Custom build command to compile Ascend C kernels first
class CustomBuild(build_ext):
    def run(self):
        # Ensure the build directory exists
        os.makedirs(ASCENDC_BUILD_DIR, exist_ok=True)
        
        print("-" * 20, "Running CMake for Ascend C Kernels", "-" * 20)
        cmake_args = [
            f"-DSOC_VERSION={SOC_VERSION}",
            f"-DASCEND_CANN_PACKAGE_PATH={CANN_PATH}",
            "-DCMAKE_BUILD_TYPE=" + ("Debug" if self.debug else "Release")
        ]
        
        # Configure and build the Ascend C kernels static library
        subprocess.check_call(["cmake", "-S", ".", "-B", ASCENDC_BUILD_DIR] + cmake_args)
        subprocess.check_call(["cmake", "--build", ASCENDC_BUILD_DIR, "--", "-j"])
        
        print("-" * 20, "CMake build finished, proceeding with pybind11 extension", "-" * 20)
        # Call the parent class's run method to build the CppExtension
        super().run()

# Find torch and torch_npu paths
try:
    import torch
    import torch_npu
    from torch.utils.cpp_extension import CppExtension, BuildExtension
except ImportError:
    raise ImportError("torch and torch_npu must be installed to build this extension.")

torch_npu_dir = os.path.dirname(os.path.abspath(torch_npu.__file__))
torch_lib_dir = os.path.dirname(os.path.abspath(torch.__file__))

ext_modules = [
    CppExtension(
        name='nanovllm_kernels',
        sources=['pybind11.cpp'],
        
        include_dirs=[
            # Path to our generated Ascend C kernel headers
            os.path.join(ASCENDC_BUILD_DIR, 'include', 'kernels'),
            
            # Path for torch_npu headers
            os.path.join(torch_npu_dir, "include"),
            
            # Path for core CANN runtime headers
            os.path.join(CANN_PATH, 'runtime', 'include'),
        ],
        
        library_dirs=[
            # Path to our compiled Ascend C static library ('libkernels.a')
            os.path.join(ASCENDC_BUILD_DIR, 'lib'),
            
            # Paths for PyTorch, Torch_NPU, and CANN dynamic libraries
            os.path.join(torch_npu_dir, "lib"),
            os.path.join(torch_lib_dir, "lib"),
            os.path.join(CANN_PATH, 'runtime', 'lib64'),
        ],
        
        # Link against these libraries
        libraries=[
            'kernels',   # Our static library
            'torch_npu', # Torch NPU library
            'ascendcl',  # CANN runtime library
            # CppExtension will automatically add c10, torch, torch_python etc.
        ],
        
        # --- FIX IS HERE ---
        # 1. Removed '-D_GLIBCXX_USE_CXX11_ABI=0' to resolve ABI mismatch.
        # 2. Kept '-g' for debug symbols.
        extra_compile_args=['-g'],

        # --- BEST PRACTICE ADDITION ---
        # 3. Add RPATH to help the loader find shared libraries at runtime.
        extra_link_args=[
            f'-Wl,-rpath,{os.path.join(torch_npu_dir, "lib")}',
            f'-Wl,-rpath,{os.path.join(torch_lib_dir, "lib")}',
            f'-Wl,-rpath,{os.path.join(CANN_PATH, "runtime", "lib64")}',
        ]
    )
]

setup(
    name='nanovllm_kernels',
    version='0.1.0',
    author='Huawei Technologies Co., Ltd.',
    description='Custom Ascend C operators for NanoVLLM',
    ext_modules=ext_modules,
    cmdclass={
        # Use our custom build class
        'build_ext': CustomBuild
    },
    zip_safe=False,
)