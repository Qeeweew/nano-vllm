# CMakeLists.txt
cmake_minimum_required(VERSION 3.16.0)
project(ascendc_kernels)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# These will be passed by setup.py
# set(SOC_VERSION "Ascend910B2")
# set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")

# Find the Ascend C CMake module
if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist.")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ascendc_library use to add kernel file to generate ascendc library
# This will create libkernels.a and the aclrtlaunch_store_kvcache.h header
ascendc_library(kernels STATIC
    store_kvcache.cpp
)