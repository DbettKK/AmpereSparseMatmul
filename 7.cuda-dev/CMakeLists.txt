cmake_minimum_required(VERSION 3.9)

set(ROUTINE spmma)

project("${ROUTINE}_example"
        DESCRIPTION  "GPU-Accelerated Sparse Linear Algebra"
        HOMEPAGE_URL "https://docs.nvidia.com/cuda/cusparse/index.html"
        LANGUAGES    CXX CUDA)

set(CMAKE_CXX_STANDARD           14)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CXX_EXTENSIONS         OFF)
set(CMAKE_CUDA_STANDARD          14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS        OFF)

string(REPLACE "/bin/nvcc" "" CUDA_TOOLKIT_PATH ${CMAKE_CUDA_COMPILER})
if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64" AND
    ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(OS_ARCH_NVRTC "sbsa-linux")
elseif (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64" AND
        ${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(OS_ARCH_NVRTC "x86_64-linux")
endif()
set(NVRTC_SHARED ${CUDA_TOOLKIT_PATH}/targets/${OS_ARCH_NVRTC}/lib/libnvrtc.so)

add_executable(${ROUTINE}_example)
add_executable(${ROUTINE}_example_static)

target_sources(${ROUTINE}_example
    PUBLIC ${PROJECT_SOURCE_DIR}/${ROUTINE}_example.cpp
)

target_sources(${ROUTINE}_example_static
    PUBLIC ${PROJECT_SOURCE_DIR}/${ROUTINE}_example.cpp
)

target_include_directories(${ROUTINE}_example
    PUBLIC ${CUDA_TOOLKIT_PATH}/include
    PUBLIC ${CUSPARSELT_PATH}/include
)

target_include_directories(${ROUTINE}_example_static
    PUBLIC ${CUDA_TOOLKIT_PATH}/include
    PUBLIC ${CUSPARSELT_PATH}/include
)

target_link_libraries(${ROUTINE}_example
    PUBLIC cudart
    PUBLIC cusparse
    PUBLIC ${CUSPARSELT_PATH}/lib64/libcusparseLt.so
    PUBLIC ${NVRTC_SHARED}
)

target_link_libraries(${ROUTINE}_example_static
    PUBLIC cudart
    PUBLIC cusparse
    PUBLIC -Wl,--whole-archive ${CUSPARSELT_PATH}/lib64/libcusparseLt_static.a
           -Wl,--no-whole-archive
    PUBLIC ${NVRTC_SHARED}
    PUBLIC ${CMAKE_DL_LIBS}
)