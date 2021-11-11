// https://github.com/wang-xinyu/tensorrtx/blob/44e797bbde2c35bee5cd1219ebf6660370ca907a/arcface/arcface-r50.cpp#L1

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

// #include "../cuda/prelu.h"
#include "../logging.hpp"
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "weight.hpp"

using namespace nvinfer1;

static const int INPUT_H = 112;
static const int INPUT_W = 112;
static const int OUTPUT_SIZE = 512;
static Logger gLogger;

IScaleLayer* addBatchNorm2d(INetworkDefinition* network,
                            std::map<std::string, Weights>& weightMap,
                            ITensor& input, std::string lname, float eps) {
  float* gamma = (float*)weightMap[lname + "_gamma"].values;
  float* beta = (float*)weightMap[lname + "_beta"].values;
  float* mean = (float*)weightMap[lname + "_moving_mean"].values;
  float* var = (float*)weightMap[lname + "_moving_var"].values;
  int len = weightMap[lname + "_moving_var"].count;

  float* scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    scval[i] = gamma[i] / sqrt(var[i] + eps);
  }
  Weights scale{DataType::kFLOAT, scval, len};

  float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
  }
  Weights shift{DataType::kFLOAT, shval, len};

  float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    pval[i] = 1.0;
  }
  Weights power{DataType::kFLOAT, pval, len};

  weightMap[lname + ".scale"] = scale;
  weightMap[lname + ".shift"] = shift;
  weightMap[lname + ".power"] = power;
  IScaleLayer* scale_1 =
      network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
  assert(scale_1);
  return scale_1;
}

ILayer* addPRelu(INetworkDefinition* network,
                 std::map<std::string, Weights>& weightMap, ITensor& input,
                 std::string lname) {
  float* gamma = (float*)weightMap[lname + "_gamma"].values;
  int len = weightMap[lname + "_gamma"].count;

  float* scval_1 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
  float* scval_2 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    scval_1[i] = -1.0;
    scval_2[i] = -gamma[i];
  }
  Weights scale_1{DataType::kFLOAT, scval_1, len};
  Weights scale_2{DataType::kFLOAT, scval_2, len};

  float* shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    shval[i] = 0.0;
  }
  Weights shift{DataType::kFLOAT, shval, len};

  float* pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    pval[i] = 1.0;
  }
  Weights power{DataType::kFLOAT, pval, len};

  auto relu1 = network->addActivation(input, ActivationType::kRELU);
  assert(relu1);
  IScaleLayer* scale1 =
      network->addScale(input, ScaleMode::kCHANNEL, shift, scale_1, power);
  assert(scale1);
  auto relu2 =
      network->addActivation(*scale1->getOutput(0), ActivationType::kRELU);
  assert(relu2);
  IScaleLayer* scale2 = network->addScale(
      *relu2->getOutput(0), ScaleMode::kCHANNEL, shift, scale_2, power);
  assert(scale2);
  IElementWiseLayer* ew1 = network->addElementWise(
      *relu1->getOutput(0), *scale2->getOutput(0), ElementWiseOperation::kSUM);
  assert(ew1);
  return ew1;
}

ILayer* resUnit(INetworkDefinition* network,
                std::map<std::string, Weights>& weightMap, ITensor& input,
                int num_filters, int s, bool dim_match, std::string lname) {
  Weights emptywts{DataType::kFLOAT, nullptr, 0};
  auto bn1 = addBatchNorm2d(network, weightMap, input, lname + "_bn1", 2e-5);
  IConvolutionLayer* conv1 =
      network->addConvolutionNd(*bn1->getOutput(0), num_filters, DimsHW{3, 3},
                                weightMap[lname + "_conv1_weight"], emptywts);
  assert(conv1);
  conv1->setPaddingNd(DimsHW{1, 1});
  auto bn2 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0),
                            lname + "_bn2", 2e-5);
  auto act1 =
      addPRelu(network, weightMap, *bn2->getOutput(0), lname + "_relu1");
  IConvolutionLayer* conv2 =
      network->addConvolutionNd(*act1->getOutput(0), num_filters, DimsHW{3, 3},
                                weightMap[lname + "_conv2_weight"], emptywts);
  assert(conv2);
  conv2->setStrideNd(DimsHW{s, s});
  conv2->setPaddingNd(DimsHW{1, 1});
  auto bn3 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0),
                            lname + "_bn3", 2e-5);

  IElementWiseLayer* ew1;
  if (dim_match) {
    ew1 = network->addElementWise(input, *bn3->getOutput(0),
                                  ElementWiseOperation::kSUM);
  } else {
    IConvolutionLayer* conv1sc = network->addConvolutionNd(
        input, num_filters, DimsHW{1, 1}, weightMap[lname + "_conv1sc_weight"],
        emptywts);
    assert(conv1sc);
    conv1sc->setStrideNd(DimsHW{s, s});
    auto bn1sc = addBatchNorm2d(network, weightMap, *conv1sc->getOutput(0),
                                lname + "_sc", 2e-5);
    ew1 = network->addElementWise(*bn1sc->getOutput(0), *bn3->getOutput(0),
                                  ElementWiseOperation::kSUM);
  }
  assert(ew1);
  return ew1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder,
                          IBuilderConfig* config, DataType dt) {
  INetworkDefinition* network = builder->createNetworkV2(0U);

  // Create input tensor of shape {3, INPUT_H, INPUT_W} with name
  // INPUT_BLOB_NAME
  ITensor* data =
      network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
  assert(data);

  std::map<std::string, Weights> weightMap = loadWeights(
      "./models/weights/"
      "arcface-r50.wts");
  Weights emptywts{DataType::kFLOAT, nullptr, 0};

  IConvolutionLayer* conv0 = network->addConvolutionNd(
      *data, 64, DimsHW{3, 3}, weightMap["conv0_weight"], emptywts);
  assert(conv0);
  conv0->setPaddingNd(DimsHW{1, 1});
  auto bn0 =
      addBatchNorm2d(network, weightMap, *conv0->getOutput(0), "bn0", 2e-5);
  auto relu0 = addPRelu(network, weightMap, *bn0->getOutput(0), "relu0");

  auto s1u1 = resUnit(network, weightMap, *relu0->getOutput(0), 64, 2, false,
                      "stage1_unit1");
  auto s1u2 = resUnit(network, weightMap, *s1u1->getOutput(0), 64, 1, true,
                      "stage1_unit2");
  auto s1u3 = resUnit(network, weightMap, *s1u2->getOutput(0), 64, 1, true,
                      "stage1_unit3");

  auto s2u1 = resUnit(network, weightMap, *s1u3->getOutput(0), 128, 2, false,
                      "stage2_unit1");
  auto s2u2 = resUnit(network, weightMap, *s2u1->getOutput(0), 128, 1, true,
                      "stage2_unit2");
  auto s2u3 = resUnit(network, weightMap, *s2u2->getOutput(0), 128, 1, true,
                      "stage2_unit3");
  auto s2u4 = resUnit(network, weightMap, *s2u3->getOutput(0), 128, 1, true,
                      "stage2_unit4");

  auto s3u1 = resUnit(network, weightMap, *s2u4->getOutput(0), 256, 2, false,
                      "stage3_unit1");
  auto s3u2 = resUnit(network, weightMap, *s3u1->getOutput(0), 256, 1, true,
                      "stage3_unit2");
  auto s3u3 = resUnit(network, weightMap, *s3u2->getOutput(0), 256, 1, true,
                      "stage3_unit3");
  auto s3u4 = resUnit(network, weightMap, *s3u3->getOutput(0), 256, 1, true,
                      "stage3_unit4");
  auto s3u5 = resUnit(network, weightMap, *s3u4->getOutput(0), 256, 1, true,
                      "stage3_unit5");
  auto s3u6 = resUnit(network, weightMap, *s3u5->getOutput(0), 256, 1, true,
                      "stage3_unit6");
  auto s3u7 = resUnit(network, weightMap, *s3u6->getOutput(0), 256, 1, true,
                      "stage3_unit7");
  auto s3u8 = resUnit(network, weightMap, *s3u7->getOutput(0), 256, 1, true,
                      "stage3_unit8");
  auto s3u9 = resUnit(network, weightMap, *s3u8->getOutput(0), 256, 1, true,
                      "stage3_unit9");
  auto s3u10 = resUnit(network, weightMap, *s3u9->getOutput(0), 256, 1, true,
                       "stage3_unit10");
  auto s3u11 = resUnit(network, weightMap, *s3u10->getOutput(0), 256, 1, true,
                       "stage3_unit11");
  auto s3u12 = resUnit(network, weightMap, *s3u11->getOutput(0), 256, 1, true,
                       "stage3_unit12");
  auto s3u13 = resUnit(network, weightMap, *s3u12->getOutput(0), 256, 1, true,
                       "stage3_unit13");
  auto s3u14 = resUnit(network, weightMap, *s3u13->getOutput(0), 256, 1, true,
                       "stage3_unit14");

  auto s4u1 = resUnit(network, weightMap, *s3u14->getOutput(0), 512, 2, false,
                      "stage4_unit1");
  auto s4u2 = resUnit(network, weightMap, *s4u1->getOutput(0), 512, 1, true,
                      "stage4_unit2");
  auto s4u3 = resUnit(network, weightMap, *s4u2->getOutput(0), 512, 1, true,
                      "stage4_unit3");

  auto bn1 =
      addBatchNorm2d(network, weightMap, *s4u3->getOutput(0), "bn1", 2e-5);
  IFullyConnectedLayer* fc1 = network->addFullyConnected(
      *bn1->getOutput(0), 512, weightMap["pre_fc1_weight"],
      weightMap["pre_fc1_bias"]);
  assert(fc1);
  auto bn2 =
      addBatchNorm2d(network, weightMap, *fc1->getOutput(0), "fc1", 2e-5);

  bn2->getOutput(0)->setName(OUTPUT_BLOB_NAME);
  network->markOutput(*bn2->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#ifdef USE_FP16
  config->setFlag(BuilderFlag::kFP16);
#endif
  std::cout << "Building engine, please wait for a while..." << std::endl;
  ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  std::cout << "Build engine successfully!" << std::endl;

  // Don't need the network any more
  network->destroy();

  // Release host memory
  for (auto& mem : weightMap) {
    free((void*)(mem.second.values));
  }

  return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
  // Create builder
  IBuilder* builder = createInferBuilder(gLogger);
  IBuilderConfig* config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an
  // engine
  ICudaEngine* engine =
      createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
  assert(engine != nullptr);

  // Serialize the engine
  (*modelStream) = engine->serialize();

  // Close everything down
  engine->destroy();
  builder->destroy();
}

int main(int argc, char** argv) {
  cudaSetDevice(DEVICE);
  IHostMemory* modelStream{nullptr};
  APIToModel(BATCH_SIZE, &modelStream);
  assert(modelStream != nullptr);

  std::ofstream p("arcface-r50.engine", std::ios::binary);
  if (!p) {
    std::cerr << "could not open plan output file" << std::endl;
    return -1;
  }

  p.write(reinterpret_cast<const char*>(modelStream->data()),
          modelStream->size());
  modelStream->destroy();
  return 1;
}
