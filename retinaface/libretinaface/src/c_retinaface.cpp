

#include "c_retinaface.h"

#include <cstring>

#include "arcface.hpp"
#include "cuda/decode_plugin.hpp"
#include "retinaface.hpp"

#ifdef __cplusplus
extern "C" {
#endif

int retinaface_Detections_size(retinaface_Detections* self) {
  return self->size;
}
retinaface_Detection retinaface_Detections_at(retinaface_Detections* self,
                                              int idx) {
  retinaface_Detection det;
  if (idx > self->size - 1) {
    return det;
  }
  memcpy(&det, &self->x[15 * idx], sizeof(decodeplugin::Detection));
  return det;
}

#ifdef __cplusplus
}
#endif
