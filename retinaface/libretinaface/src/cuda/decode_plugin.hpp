#pragma once

namespace decodeplugin {
struct alignas(float) Detection {
  float bbox[4];  // x1 y1 x2 y2
  float class_confidence;
  float landmark[10];
};
static const int INPUT_H = 480;
static const int INPUT_W = 640;
}  // namespace decodeplugin
