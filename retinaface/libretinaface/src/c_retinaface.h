#ifndef _C_RETINAFACE_H
#define _C_RETINAFACE_H

#include "cuda/decode.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#define C_RETINAFACE_INPUT_H 480
#define C_RETINAFACE_INPUT_W 640
#define C_RETINAFACE_OUTPUT_SIZE                             \
  ((C_RETINAFACE_INPUT_H / 8 * C_RETINAFACE_INPUT_W / 8 +    \
    C_RETINAFACE_INPUT_H / 16 * C_RETINAFACE_INPUT_W / 16 +  \
    C_RETINAFACE_INPUT_H / 32 * C_RETINAFACE_INPUT_W / 32) * \
       2 * 15 +                                              \
   1)

typedef struct retinaface_Detection {
  float bbox[4];  // x1 y1 x2 y2
  float class_confidence;
  float landmark[10];
} retinaface_Detection;

/* Wrapper struct to hold a pointer to
   Rectangle object in C  */
// struct Detections;
typedef struct retinaface_Detections {
  int size;
  float x[C_RETINAFACE_OUTPUT_SIZE];
} retinaface_Detections;

int retinaface_Detections_size(retinaface_Detections* self);
retinaface_Detection retinaface_Detections_at(retinaface_Detections* self,
                                              int idx);

// struct Retinaface;
// typedef struct Retinaface Retinaface;

// Arcface
typedef struct arcface_Detection {
  float x[512];
} arcface_Detection;

// struct Arcface;
// typedef struct Arcface Arcface;

#ifdef __cplusplus
}
#endif

#endif
