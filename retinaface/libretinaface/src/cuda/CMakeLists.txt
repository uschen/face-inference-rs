cuda_add_library(decodeplugin SHARED
  decode_plugin.hpp
  decode.hpp
  decode.cu
  prelu.h
  prelu.cu
)
target_link_libraries(decodeplugin nvinfer cudart)
