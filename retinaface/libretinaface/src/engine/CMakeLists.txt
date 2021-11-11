

add_library(weight SHARED
  weight.hpp
  weight.cpp
)

add_executable(create_mnet create_mnet.cpp)
target_link_libraries(create_mnet PRIVATE
  nvinfer
  cuda
  cudart
  decodeplugin
  weight
  ${CUDA_LIBRARIES}
)

add_executable(create_r50 create_r50.cpp)
target_link_libraries(create_r50 PRIVATE
  nvinfer
  cuda
  cudart
  decodeplugin
  retinaface
  weight
  ${CUDA_LIBRARIES}
)

add_executable(create_arcface_r50 create_arcface_r50.cpp)
target_link_libraries(create_arcface_r50 PRIVATE
  nvinfer
  cuda
  cudart
  decodeplugin
  retinaface
  weight
  ${CUDA_LIBRARIES}
)

