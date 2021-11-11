#pragma once
#include "base_engine.hpp"
#include "c_retinaface.h"

namespace retinaface {
class Arcface : public BaseEngine {
 public:
  static const int INPUT_H = 112;
  static const int INPUT_W = 112;
  static const int BATCH_SIZE = 1;
  // static const int TOP_K = 5000;
  static const int OUTPUT_SIZE = 512;

  // Create engine from engine path
  Arcface(const std::string& engine_path, bool verbose);

  void Infer(float* input, arcface_Detection* output);

  ~Arcface();
};
}  // namespace retinaface

typedef retinaface::Arcface Arcface;

#ifdef __cplusplus
extern "C" {
#endif

Arcface* Arcface_new(char* path, bool verbose);
void Arcface_destroy(Arcface* self);

void Arcface_infer(Arcface* self, arcface_Detection* output, float* input);

#ifdef __cplusplus
}
#endif
