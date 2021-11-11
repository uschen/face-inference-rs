#pragma once

#include <map>
#include <vector>

#include "NvInfer.h"
#include "cuda/decode.hpp"

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define BATCH_SIZE 1

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";

std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

nvinfer1::Weights getWeights(
    std::map<std::string, nvinfer1::Weights>& weightMap, std::string key);
