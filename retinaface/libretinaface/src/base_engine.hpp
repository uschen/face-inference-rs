#pragma once
#include <NvInfer.h>

#include <string>

namespace retinaface {
class BaseEngine {
 public:
  BaseEngine(const std::string& engine_path, const char* input_blob_name,
             const char* output_blob_name, int input_h, int input_w,
             int output_size, int batch_size, bool verbose = false);
  void DoInference(float* input, float* output, int batchSize);
  //   int GetBatchSize();
  ~BaseEngine();

 protected:
  std::string _input_blob_name;
  std::string _output_blob_name;
  int _input_h;
  int _input_w;
  int _output_size;
  int _batch_size;

  // ?? can't use one context for multi thread access
  nvinfer1::IExecutionContext* _context = nullptr;
  nvinfer1::ICudaEngine* _engine = nullptr;
  nvinfer1::IRuntime* _runtime = nullptr;
  cudaStream_t _stream = nullptr;
};
}  // namespace retinaface
