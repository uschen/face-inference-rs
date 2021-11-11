#include <NvInfer.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>

#include "cuda_runtime_api.h"
#include "utils.hpp"

using namespace nvinfer1;

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file) {
  std::cout << "Loading weights: " << file << std::endl;
  std::map<std::string, Weights> weightMap;

  // Open weights file
  std::ifstream input(file);
  assert(input.is_open() && "Unable to load weight file.");

  // Read number of weight blobs
  int32_t count;
  input >> count;
  assert(count > 0 && "Invalid weight map file.");

  while (count--) {
    Weights wt{DataType::kFLOAT, nullptr, 0};
    uint32_t size;

    // Read name and type of blob
    std::string name;
    input >> name >> std::dec >> size;
    wt.type = DataType::kFLOAT;

    // Load blob
    uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
    for (uint32_t x = 0, y = size; x < y; ++x) {
      input >> std::hex >> val[x];
    }
    wt.values = val;

    wt.count = size;
    weightMap[name] = wt;
  }

  return weightMap;
}

nvinfer1::Weights getWeights(
    std::map<std::string, nvinfer1::Weights>& weightMap, std::string key) {
  if (weightMap.count(key) != 1) {
    std::cerr << key << " not existed in weight map, fatal error!!!"
              << std::endl;
    exit(-1);
  }
  return weightMap[key];
}
