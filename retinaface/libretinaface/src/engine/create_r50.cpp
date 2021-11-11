#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <vector>

#include "../cuda/decode_plugin.hpp"
#include "NvInfer.h"
#include "cuda/decode.hpp"
#include "cuda_runtime_api.h"
#include "logging.hpp"
#include "weight.hpp"

using namespace nvinfer1;
static Logger gLogger;

IScaleLayer *addBatchNorm2d(INetworkDefinition *network,
                            std::map<std::string, Weights> &weightMap,
                            ITensor &input, std::string lname, float eps) {
  float *gamma = (float *)weightMap[lname + ".weight"].values;
  float *beta = (float *)weightMap[lname + ".bias"].values;
  float *mean = (float *)weightMap[lname + ".running_mean"].values;
  float *var = (float *)weightMap[lname + ".running_var"].values;
  int len = weightMap[lname + ".running_var"].count;

  float *scval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    scval[i] = gamma[i] / sqrt(var[i] + eps);
  }
  Weights scale{DataType::kFLOAT, scval, len};

  float *shval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
  }
  Weights shift{DataType::kFLOAT, shval, len};

  float *pval = reinterpret_cast<float *>(malloc(sizeof(float) * len));
  for (int i = 0; i < len; i++) {
    pval[i] = 1.0;
  }
  Weights power{DataType::kFLOAT, pval, len};

  weightMap[lname + ".scale"] = scale;
  weightMap[lname + ".shift"] = shift;
  weightMap[lname + ".power"] = power;
  IScaleLayer *scale_1 =
      network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
  assert(scale_1);
  return scale_1;
}

IActivationLayer *bottleneck(INetworkDefinition *network,
                             std::map<std::string, Weights> &weightMap,
                             ITensor &input, int inch, int outch, int stride,
                             std::string lname) {
  Weights emptywts{DataType::kFLOAT, nullptr, 0};

  IConvolutionLayer *conv1 = network->addConvolutionNd(
      input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
  assert(conv1);

  IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0),
                                    lname + "bn1", 1e-5);

  IActivationLayer *relu1 =
      network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
  assert(relu1);

  IConvolutionLayer *conv2 =
      network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3},
                                weightMap[lname + "conv2.weight"], emptywts);
  assert(conv2);
  conv2->setStrideNd(DimsHW{stride, stride});
  conv2->setPaddingNd(DimsHW{1, 1});

  IScaleLayer *bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0),
                                    lname + "bn2", 1e-5);

  IActivationLayer *relu2 =
      network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
  assert(relu2);

  IConvolutionLayer *conv3 =
      network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1},
                                weightMap[lname + "conv3.weight"], emptywts);
  assert(conv3);

  IScaleLayer *bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0),
                                    lname + "bn3", 1e-5);

  IElementWiseLayer *ew1;
  if (stride != 1 || inch != outch * 4) {
    IConvolutionLayer *conv4 = network->addConvolutionNd(
        input, outch * 4, DimsHW{1, 1},
        weightMap[lname + "downsample.0.weight"], emptywts);
    assert(conv4);
    conv4->setStrideNd(DimsHW{stride, stride});

    IScaleLayer *bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0),
                                      lname + "downsample.1", 1e-5);
    ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0),
                                  ElementWiseOperation::kSUM);
  } else {
    ew1 = network->addElementWise(input, *bn3->getOutput(0),
                                  ElementWiseOperation::kSUM);
  }
  IActivationLayer *relu3 =
      network->addActivation(*ew1->getOutput(0), ActivationType::kRELU);
  assert(relu3);
  return relu3;
}

ILayer *conv_bn_relu(INetworkDefinition *network,
                     std::map<std::string, Weights> &weightMap, ITensor &input,
                     int outch, int kernelsize, int stride, int padding,
                     bool userelu, std::string lname) {
  Weights emptywts{DataType::kFLOAT, nullptr, 0};

  IConvolutionLayer *conv1 = network->addConvolutionNd(
      input, outch, DimsHW{kernelsize, kernelsize},
      getWeights(weightMap, lname + ".0.weight"), emptywts);
  assert(conv1);
  conv1->setStrideNd(DimsHW{stride, stride});
  conv1->setPaddingNd(DimsHW{padding, padding});

  IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0),
                                    lname + ".1", 1e-5);

  if (!userelu)
    return bn1;

  IActivationLayer *relu1 =
      network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
  assert(relu1);

  return relu1;
}

IActivationLayer *ssh(INetworkDefinition *network,
                      std::map<std::string, Weights> &weightMap, ITensor &input,
                      std::string lname) {
  auto conv3x3 = conv_bn_relu(network, weightMap, input, 256 / 2, 3, 1, 1,
                              false, lname + ".conv3X3");
  auto conv5x5_1 = conv_bn_relu(network, weightMap, input, 256 / 4, 3, 1, 1,
                                true, lname + ".conv5X5_1");
  auto conv5x5 = conv_bn_relu(network, weightMap, *conv5x5_1->getOutput(0),
                              256 / 4, 3, 1, 1, false, lname + ".conv5X5_2");
  auto conv7x7 = conv_bn_relu(network, weightMap, *conv5x5_1->getOutput(0),
                              256 / 4, 3, 1, 1, true, lname + ".conv7X7_2");
  conv7x7 = conv_bn_relu(network, weightMap, *conv7x7->getOutput(0), 256 / 4, 3,
                         1, 1, false, lname + ".conv7x7_3");
  ITensor *inputTensors[] = {conv3x3->getOutput(0), conv5x5->getOutput(0),
                             conv7x7->getOutput(0)};
  auto cat = network->addConcatenation(inputTensors, 3);
  IActivationLayer *relu1 =
      network->addActivation(*cat->getOutput(0), ActivationType::kRELU);
  assert(relu1);
  return relu1;
}

// Creat the engine using only the API and not any parser.
ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder *builder,
                          IBuilderConfig *config, DataType dt) {
  INetworkDefinition *network = builder->createNetworkV2(0U);

  // Create input tensor with name INPUT_BLOB_NAME
  ITensor *data =
      network->addInput(INPUT_BLOB_NAME, dt,
                        Dims3{3, decodeplugin::INPUT_H, decodeplugin::INPUT_W});
  assert(data);
  const std::string WEIGHT_PATH = "./models/weights/"
                                  "retinaface_r50.wts";
  std::map<std::string, Weights> weightMap = loadWeights(WEIGHT_PATH);
  Weights emptywts{DataType::kFLOAT, nullptr, 0};

  // ------------- backbone resnet50 ---------------
  IConvolutionLayer *conv1 = network->addConvolutionNd(
      *data, 64, DimsHW{7, 7}, weightMap["body.conv1.weight"], emptywts);
  assert(conv1);
  conv1->setStrideNd(DimsHW{2, 2});
  conv1->setPaddingNd(DimsHW{3, 3});

  IScaleLayer *bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0),
                                    "body.bn1", 1e-5);

  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu1 =
      network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
  assert(relu1);

  // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
  IPoolingLayer *pool1 = network->addPoolingNd(*relu1->getOutput(0),
                                               PoolingType::kMAX, DimsHW{3, 3});
  assert(pool1);
  pool1->setStrideNd(DimsHW{2, 2});
  pool1->setPaddingNd(DimsHW{1, 1});

  IActivationLayer *x = bottleneck(network, weightMap, *pool1->getOutput(0), 64,
                                   64, 1, "body.layer1.0.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1,
                 "body.layer1.1.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 256, 64, 1,
                 "body.layer1.2.");

  x = bottleneck(network, weightMap, *x->getOutput(0), 256, 128, 2,
                 "body.layer2.0.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1,
                 "body.layer2.1.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1,
                 "body.layer2.2.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 512, 128, 1,
                 "body.layer2.3.");
  IActivationLayer *layer2 = x;

  x = bottleneck(network, weightMap, *x->getOutput(0), 512, 256, 2,
                 "body.layer3.0.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1,
                 "body.layer3.1.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1,
                 "body.layer3.2.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1,
                 "body.layer3.3.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1,
                 "body.layer3.4.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 256, 1,
                 "body.layer3.5.");
  IActivationLayer *layer3 = x;

  x = bottleneck(network, weightMap, *x->getOutput(0), 1024, 512, 2,
                 "body.layer4.0.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1,
                 "body.layer4.1.");
  x = bottleneck(network, weightMap, *x->getOutput(0), 2048, 512, 1,
                 "body.layer4.2.");
  IActivationLayer *layer4 = x;

  // ------------- FPN ---------------
  auto output1 = conv_bn_relu(network, weightMap, *layer2->getOutput(0), 256, 1,
                              1, 0, true, "fpn.output1");
  auto output2 = conv_bn_relu(network, weightMap, *layer3->getOutput(0), 256, 1,
                              1, 0, true, "fpn.output2");
  auto output3 = conv_bn_relu(network, weightMap, *layer4->getOutput(0), 256, 1,
                              1, 0, true, "fpn.output3");

  float *deval = reinterpret_cast<float *>(malloc(sizeof(float) * 256 * 2 * 2));
  for (int i = 0; i < 256 * 2 * 2; i++) {
    deval[i] = 1.0;
  }
  Weights deconvwts{DataType::kFLOAT, deval, 256 * 2 * 2};
  IDeconvolutionLayer *up3 = network->addDeconvolutionNd(
      *output3->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
  assert(up3);
  up3->setStrideNd(DimsHW{2, 2});
  up3->setNbGroups(256);
  weightMap["up3"] = deconvwts;

  output2 = network->addElementWise(*output2->getOutput(0), *up3->getOutput(0),
                                    ElementWiseOperation::kSUM);
  output2 = conv_bn_relu(network, weightMap, *output2->getOutput(0), 256, 3, 1,
                         1, true, "fpn.merge2");

  IDeconvolutionLayer *up2 = network->addDeconvolutionNd(
      *output2->getOutput(0), 256, DimsHW{2, 2}, deconvwts, emptywts);
  assert(up2);
  up2->setStrideNd(DimsHW{2, 2});
  up2->setNbGroups(256);
  output1 = network->addElementWise(*output1->getOutput(0), *up2->getOutput(0),
                                    ElementWiseOperation::kSUM);
  output1 = conv_bn_relu(network, weightMap, *output1->getOutput(0), 256, 3, 1,
                         1, true, "fpn.merge1");

  // ------------- SSH ---------------
  auto ssh1 = ssh(network, weightMap, *output1->getOutput(0), "ssh1");
  auto ssh2 = ssh(network, weightMap, *output2->getOutput(0), "ssh2");
  auto ssh3 = ssh(network, weightMap, *output3->getOutput(0), "ssh3");

  // ------------- Head ---------------
  auto bbox_head1 =
      network->addConvolutionNd(*ssh1->getOutput(0), 2 * 4, DimsHW{1, 1},
                                weightMap["BboxHead.0.conv1x1.weight"],
                                weightMap["BboxHead.0.conv1x1.bias"]);
  auto bbox_head2 =
      network->addConvolutionNd(*ssh2->getOutput(0), 2 * 4, DimsHW{1, 1},
                                weightMap["BboxHead.1.conv1x1.weight"],
                                weightMap["BboxHead.1.conv1x1.bias"]);
  auto bbox_head3 =
      network->addConvolutionNd(*ssh3->getOutput(0), 2 * 4, DimsHW{1, 1},
                                weightMap["BboxHead.2.conv1x1.weight"],
                                weightMap["BboxHead.2.conv1x1.bias"]);

  auto cls_head1 =
      network->addConvolutionNd(*ssh1->getOutput(0), 2 * 2, DimsHW{1, 1},
                                weightMap["ClassHead.0.conv1x1.weight"],
                                weightMap["ClassHead.0.conv1x1.bias"]);
  auto cls_head2 =
      network->addConvolutionNd(*ssh2->getOutput(0), 2 * 2, DimsHW{1, 1},
                                weightMap["ClassHead.1.conv1x1.weight"],
                                weightMap["ClassHead.1.conv1x1.bias"]);
  auto cls_head3 =
      network->addConvolutionNd(*ssh3->getOutput(0), 2 * 2, DimsHW{1, 1},
                                weightMap["ClassHead.2.conv1x1.weight"],
                                weightMap["ClassHead.2.conv1x1.bias"]);

  auto lmk_head1 =
      network->addConvolutionNd(*ssh1->getOutput(0), 2 * 10, DimsHW{1, 1},
                                weightMap["LandmarkHead.0.conv1x1.weight"],
                                weightMap["LandmarkHead.0.conv1x1.bias"]);
  auto lmk_head2 =
      network->addConvolutionNd(*ssh2->getOutput(0), 2 * 10, DimsHW{1, 1},
                                weightMap["LandmarkHead.1.conv1x1.weight"],
                                weightMap["LandmarkHead.1.conv1x1.bias"]);
  auto lmk_head3 =
      network->addConvolutionNd(*ssh3->getOutput(0), 2 * 10, DimsHW{1, 1},
                                weightMap["LandmarkHead.2.conv1x1.weight"],
                                weightMap["LandmarkHead.2.conv1x1.bias"]);

  // ------------- Decode bbox, conf, landmark ---------------
  ITensor *inputTensors1[] = {bbox_head1->getOutput(0), cls_head1->getOutput(0),
                              lmk_head1->getOutput(0)};
  auto cat1 = network->addConcatenation(inputTensors1, 3);
  ITensor *inputTensors2[] = {bbox_head2->getOutput(0), cls_head2->getOutput(0),
                              lmk_head2->getOutput(0)};
  auto cat2 = network->addConcatenation(inputTensors2, 3);
  ITensor *inputTensors3[] = {bbox_head3->getOutput(0), cls_head3->getOutput(0),
                              lmk_head3->getOutput(0)};
  auto cat3 = network->addConcatenation(inputTensors3, 3);

  auto creator = getPluginRegistry()->getPluginCreator("Decode_TRT", "1");
  PluginFieldCollection pfc;
  IPluginV2 *pluginObj = creator->createPlugin("decode", &pfc);
  ITensor *inputTensors[] = {cat1->getOutput(0), cat2->getOutput(0),
                             cat3->getOutput(0)};
  auto decodelayer = network->addPluginV2(inputTensors, 3, *pluginObj);
  assert(decodelayer);

  decodelayer->getOutput(0)->setName(OUTPUT_BLOB_NAME);
  network->markOutput(*decodelayer->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(1 << 20);
#ifdef USE_FP16
  config->setFlag(BuilderFlag::kFP16);
#endif
  std::cout << "Building engine, please wait for a while..." << std::endl;
  ICudaEngine *engine = builder->buildEngineWithConfig(*network, *config);
  std::cout << "Build engine successfully!" << std::endl;

  // Don't need the network any more
  network->destroy();

  // Release host memory
  for (auto &mem : weightMap) {
    free((void *)(mem.second.values));
    mem.second.values = NULL;
  }

  return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory **modelStream) {
  // Create builder
  IBuilder *builder = createInferBuilder(gLogger);
  IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an
  // engine
  ICudaEngine *engine =
      createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
  assert(engine != nullptr);

  // Serialize the engine
  (*modelStream) = engine->serialize();

  // Close everything down
  engine->destroy();
  builder->destroy();
}

int main(int argc, char **argv) {
  IHostMemory *modelStream{nullptr};
  APIToModel(BATCH_SIZE, &modelStream);
  assert(modelStream != nullptr);

  std::ofstream p("retina_r50.engine", std::ios::binary);
  if (!p) {
    std::cerr << "could not open plan output file" << std::endl;
    return -1;
  }
  p.write(reinterpret_cast<const char *>(modelStream->data()),
          modelStream->size());
  modelStream->destroy();
}
