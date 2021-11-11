#pragma once

#include <string>
#include <vector>

#include "base_engine.hpp"
#include "c_retinaface.h"
#include "cuda/decode_plugin.hpp"

namespace retinaface {
class Retinaface : public BaseEngine {
public:
  static const int INPUT_H = decodeplugin::INPUT_H;
  static const int INPUT_W = decodeplugin::INPUT_W;
  static const int BATCH_SIZE = 1;
  static const int TOP_K = 5000;
  static const int OUTPUT_SIZE =
      (INPUT_H / 8 * INPUT_W / 8 + INPUT_H / 16 * INPUT_W / 16 +
       INPUT_H / 32 * INPUT_W / 32) *
          2 * 15 +
      1;

  // Create engine from engine path
  Retinaface(const std::string &engine_path, bool verbose = false);
  // Get max allowed batch size
  int getMaxBatchSize();
  // Get max number of detections
  int getMaxDetections();
  // void infer(float* output, int output_size, float* input, int input_size);
  void Infer(retinaface_Detections &output, float *input, int org_img_h,
             int org_img_w, float vis_thresh);
  ~Retinaface();

private:
  void GetRectAdaptLandmark(int img_h, int img_w, float bbox[4], float lmk[10]);
  void nms(std::vector<decodeplugin::Detection> &res, float *output,
           float nms_thresh);
  void nms(retinaface_Detections &res, float *output, float nms_thresh);
};
} // namespace retinaface
typedef retinaface::Retinaface Retinaface;

#ifdef __cplusplus
extern "C" {
#endif
Retinaface *Retinaface_new(char *path, bool verbose);
void Retinaface_destroy(Retinaface *self);
void Retinaface_infer(Retinaface *self, retinaface_Detections *output,
                      float *input, int org_img_h, int org_img_w,
                      float vis_thresh);
#ifdef __cplusplus
}
#endif
