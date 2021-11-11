#pragma once

#define CHECK(status)                                    \
  do {                                                   \
    auto ret = (status);                                 \
    if (ret != 0) {                                      \
      std::cerr << "Cuda failure: " << ret << std::endl; \
      abort();                                           \
    }                                                    \
  } while (0)

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id

// for mnet
#define VIS_THRESH 0.6
