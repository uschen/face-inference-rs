#include "base_engine.hpp"

#include <fstream>

#include "cuda_runtime_api.h"
#include "logging.hpp"
#include "utils.hpp"

using namespace nvinfer1;
using std::cout;
using std::endl;

static Logger gLogger;

namespace retinaface {
BaseEngine::BaseEngine(const std::string& engine_path,
                       const char* input_blob_name,
                       const char* output_blob_name, int input_h, int input_w,
                       int output_size, int batch_size, bool verbose) {
  cout << "Engine: " << engine_path << endl;
  _input_blob_name = input_blob_name;
  _output_blob_name = output_blob_name;
  _input_h = input_h;
  _input_w = input_w;
  _output_size = output_size;
  _batch_size = batch_size;
  std::ifstream file(engine_path, std::ios::in | std::ios::binary);
  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);

  char* buffer = new char[size];
  file.read(buffer, size);
  file.close();
  cout << "Engine Read File" << endl;

  _runtime = createInferRuntime(gLogger);

  _engine = _runtime->deserializeCudaEngine(buffer, size, nullptr);
  assert(_runtime != nullptr);
  delete[] buffer;

  _context = _engine->createExecutionContext();
  cudaStreamCreate(&_stream);
};

void BaseEngine::DoInference(float* input, float* output, int batchSize) {
  // cout << "DoInference " << endl;

  // auto _context = _engine->createExecutionContext();
  // cudaStream_t _stream;

  // cudaStreamCreate(&_stream);
  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(_engine->getNbBindings() == 2);
  void* buffers[2];

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. Note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  const int inputIndex = _engine->getBindingIndex(_input_blob_name.c_str());
  const int outputIndex = _engine->getBindingIndex(_output_blob_name.c_str());

  // Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[inputIndex],
                   batchSize * 3 * _input_h * _input_w * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex],
                   batchSize * _output_size * sizeof(float)));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA
  // output back to host
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
                        batchSize * 3 * _input_h * _input_w * sizeof(float),
                        cudaMemcpyHostToDevice, _stream));
  _context->enqueue(batchSize, buffers, _stream, nullptr);
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex],
                        batchSize * _output_size * sizeof(float),
                        cudaMemcpyDeviceToHost, _stream));
  cudaStreamSynchronize(_stream);

  // Release buffers
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));

  // if (_stream) cudaStreamDestroy(_stream);
  // if (_context) _context->destroy();
}

// int BaseEngine::GetBatchSize() { return _batch_size; }

BaseEngine::~BaseEngine() {
  if (_stream) cudaStreamDestroy(_stream);
  if (_context) _context->destroy();
  if (_engine) _engine->destroy();
  if (_runtime) _runtime->destroy();
};
}  // namespace retinaface
