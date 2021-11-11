#include "arcface.hpp"

#include <NvInfer.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <ostream>

// #include "c_retinaface.h"
// #include "cuda/decode_plugin.hpp"
#include "cuda_runtime_api.h"
#include "logging.hpp"

const char* ARCFACE_INPUT_BLOB_NAME = "data";
const char* ARCFACE_OUTPUT_BLOB_NAME = "prob";

using namespace nvinfer1;

using std::cout;
using std::endl;

static Logger gLogger;

namespace retinaface {
Arcface::Arcface(const std::string& engine_path, bool verbose = false)
    : BaseEngine(engine_path, ARCFACE_INPUT_BLOB_NAME, ARCFACE_OUTPUT_BLOB_NAME,
                 INPUT_H, INPUT_W, OUTPUT_SIZE, verbose){};
Arcface::~Arcface(){};

void Arcface::Infer(float* input, arcface_Detection* output) {
  BaseEngine::DoInference(input, output->x, BATCH_SIZE);
}

}  // namespace retinaface

#ifdef __cplusplus
extern "C" {
#endif

Arcface* Arcface_new(char* path, bool verbose) {
  return new retinaface::Arcface(path, verbose);
}

void Arcface_destroy(Arcface* self) {
  delete reinterpret_cast<retinaface::Arcface*>(self);
}

void Arcface_infer(Arcface* self, arcface_Detection* output, float* input) {
  return self->Infer(input, output);
}

#ifdef __cplusplus
}
#endif
