cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(retinaface LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wno-write-strings")	
find_package(CUDA REQUIRED)
message("-- CUDA version: ${CUDA_VERSION}")
enable_language(CUDA)


if(CUDA_VERSION_MAJOR GREATER 9)
    message("-- CUDA ${CUDA_VERSION_MAJOR} detected, enabling SM_72")

    set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}; 
        -gencode arch=compute_72,code=sm_72
    )
endif()

find_package(OpenCV REQUIRED)

if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
list(APPEND CUDA_INCLUDE_DIRS "/usr/local/cuda-10.2/targets/aarch64-linux/include")
message("-- CUDA INCLUDE DIRS: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
message("-- PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message("-- OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}, ${OpenCV_LIBS}")
if(NOT DEFINED ARCH)
  set(ARCH "sm_70")
endif(NOT DEFINED ARCH)


include_directories(
  ${PROJECT_SOURCE_DIR}/src
)

add_library(retinaface STATIC
  ${PROJECT_SOURCE_DIR}/src/cuda/decode.hpp
  ${PROJECT_SOURCE_DIR}/src/cuda/decode_plugin.hpp
  ${PROJECT_SOURCE_DIR}/src/cuda/decode.cu
  ${PROJECT_SOURCE_DIR}/src/cuda/prelu.h
  ${PROJECT_SOURCE_DIR}/src/cuda/prelu.cu

  ${PROJECT_SOURCE_DIR}/src/logging.hpp
  ${PROJECT_SOURCE_DIR}/src/utils.hpp

  ${PROJECT_SOURCE_DIR}/src/base_engine.hpp
  ${PROJECT_SOURCE_DIR}/src/base_engine.cpp

  ${PROJECT_SOURCE_DIR}/src/c_retinaface.h
  ${PROJECT_SOURCE_DIR}/src/c_retinaface.cpp

  ${PROJECT_SOURCE_DIR}/src/retinaface.hpp
  ${PROJECT_SOURCE_DIR}/src/retinaface.cpp

  ${PROJECT_SOURCE_DIR}/src/arcface.hpp
  ${PROJECT_SOURCE_DIR}/src/arcface.cpp
)
target_sources(retinaface
  PUBLIC
  ${PROJECT_SOURCE_DIR}/src/c_retinaface.h
  ${PROJECT_SOURCE_DIR}/src/arcface.hpp
)
target_link_libraries(retinaface
  nvinfer
  cuda
  cudart
  ${CUDA_LIBRARIES}
)
set_target_properties(retinaface PROPERTIES
 CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
# install(TARGETS retinaface DESTINATION .)

install(TARGETS retinaface 
  EXPORT RetinafaceTargets
  DESTINATION ${CMAKE_INSTALL_PREFIX}
)

add_subdirectory(src/cuda)
include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(src/engine)
add_subdirectory(src/example)
