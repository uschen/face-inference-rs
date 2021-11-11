#include "retinaface.hpp"

#include <NvInfer.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <memory>
#include <ostream>

#include "cuda/decode_plugin.hpp"
#include "cuda_runtime_api.h"
#include "logging.hpp"

// retinaface::Engine
using namespace nvinfer1;
using std::cout;
using std::endl;

static Logger gLogger;

const char *INPUT_BLOB_NAME = "data";
const char *OUTPUT_BLOB_NAME = "prob";

namespace retinaface {
Retinaface::Retinaface(const std::string &engine_path, bool verbose)
    : BaseEngine(engine_path, INPUT_BLOB_NAME, OUTPUT_BLOB_NAME, INPUT_H,
                 INPUT_W, OUTPUT_SIZE, verbose){};

Retinaface::~Retinaface(){};

// std::unique_ptr<Retinaface> Retinaface::factory() { return std::move(this); }

// Detection decode_detection_to_detection(float class_confidence, float
// bbox[4],
//                                         float lmk[10]) {
//   return Detection{*bbox, class_confidence, *lmk};
// }

// void infer(float* output, int output_size, float* input, int input_size) {
//   // Pointers to input and output device buffers to pass to engine.

// }
void Retinaface::Infer(retinaface_Detections &output, float *input,
                       int org_img_h, int org_img_w, float vis_thresh) {
  // Run inference
  static float prob[BATCH_SIZE * OUTPUT_SIZE];

  BaseEngine::DoInference(input, prob, BATCH_SIZE);

  std::vector<decodeplugin::Detection> res;
  for (int b = 0; b < BATCH_SIZE; b++) {
    std::vector<decodeplugin::Detection> det;
    nms(det, &prob[b * OUTPUT_SIZE], 0.4);
    for (size_t j = 0; j < det.size(); j++) {
      cout << "j confidence: " << det[j].class_confidence << endl;
      if (det[j].class_confidence < vis_thresh)
        continue;
      // transform back to oringal coordinate
      GetRectAdaptLandmark(org_img_h, org_img_w, det[j].bbox, det[j].landmark);
      res.push_back(det[j]);
    }
  }
  cout << "infer res.size: " << res.size() << endl;

  output.size = res.size();
  // assert(sizeof(Detection) == sizeof(decodeplugin::Detection));
  for (int i = 0; i < res.size(); i++) {
    memcpy(&output.x[sizeof(decodeplugin::Detection) / sizeof(float) * i],
           &res[i], sizeof(decodeplugin::Detection));
  }
  return;
};

void Retinaface::GetRectAdaptLandmark(int img_h, int img_w, float bbox[4],
                                      float lmk[10]) {
  int l, r, t, b;
  float r_w = retinaface::Retinaface::INPUT_W / (img_w * 1.0);
  float r_h = retinaface::Retinaface::INPUT_H / (img_h * 1.0);
  if (r_h > r_w) {
    l = bbox[0] / r_w;
    r = bbox[2] / r_w;
    t = (bbox[1] - (retinaface::Retinaface::INPUT_H - r_w * img_h) / 2) / r_w;
    b = (bbox[3] - (retinaface::Retinaface::INPUT_H - r_w * img_h) / 2) / r_w;
    for (int i = 0; i < 10; i += 2) {
      lmk[i] /= r_w;
      lmk[i + 1] =
          (lmk[i + 1] - (retinaface::Retinaface::INPUT_H - r_w * img_h) / 2) /
          r_w;
    }
  } else {
    l = (bbox[0] - (retinaface::Retinaface::INPUT_W - r_h * img_w) / 2) / r_h;
    r = (bbox[2] - (retinaface::Retinaface::INPUT_W - r_h * img_w) / 2) / r_h;
    t = bbox[1] / r_h;
    b = bbox[3] / r_h;
    for (int i = 0; i < 10; i += 2) {
      lmk[i] =
          (lmk[i] - (retinaface::Retinaface::INPUT_W - r_h * img_w) / 2) / r_h;
      lmk[i + 1] /= r_h;
    }
  }
  bbox[0] = l;
  bbox[1] = t;
  bbox[2] = r;
  bbox[3] = b;
}

float iou(float lbox[4], float rbox[4]) {
  float interBox[] = {
      std::max(lbox[0], rbox[0]), // left
      std::min(lbox[2], rbox[2]), // right
      std::max(lbox[1], rbox[1]), // top
      std::min(lbox[3], rbox[3]), // bottom
  };

  if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
    return 0.0f;

  float interBoxS = (interBox[1] - interBox[0]) * (interBox[3] - interBox[2]);
  return interBoxS /
         ((lbox[2] - lbox[0]) * (lbox[3] - lbox[1]) +
          (rbox[2] - rbox[0]) * (rbox[3] - rbox[1]) - interBoxS + 0.000001f);
}

bool cmp(const decodeplugin::Detection &a, const decodeplugin::Detection &b) {
  return a.class_confidence > b.class_confidence;
}

void Retinaface::nms(std::vector<decodeplugin::Detection> &res, float *output,
                     float nms_thresh = 0.4) {
  std::vector<decodeplugin::Detection> dets;
  for (int i = 0; i < output[0]; i++) {
    if (output[15 * i + 1 + 4] <= 0.1)
      continue;
    decodeplugin::Detection det;
    memcpy(&det, &output[15 * i + 1], sizeof(decodeplugin::Detection));
    dets.push_back(det);
  }
  std::sort(dets.begin(), dets.end(), cmp);
  if (dets.size() > TOP_K)
    dets.erase(dets.begin() + TOP_K, dets.end());
  for (size_t m = 0; m < dets.size(); ++m) {
    auto &item = dets[m];
    res.push_back(item);
    for (size_t n = m + 1; n < dets.size(); ++n) {
      if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
        dets.erase(dets.begin() + n);
        --n;
      }
    }
  }
}
} // namespace retinaface

#ifdef __cplusplus
extern "C" {
#endif

Retinaface *Retinaface_new(char *path, bool verbose) {
  auto i = new retinaface::Retinaface(path, verbose);
  return std::move(i);
}
void Retinaface_destroy(Retinaface *self) {
  delete reinterpret_cast<retinaface::Retinaface *>(self);
}

void Retinaface_infer(Retinaface *self, retinaface_Detections *output,
                      float *input, int org_img_h, int org_img_w,
                      float vis_thresh) {
  retinaface_Detections &out = *output;
  self->Infer(out, input, org_img_h, org_img_w, vis_thresh);
}

#ifdef __cplusplus
}
#endif
