# face-inference-rs

An example project to build a face recognition, face detection, and face alignment inference service using GRPC, Rust, TensorRT, CUDA, running on Jetson Nano, Jetson Xavier NX.

## Architecture

### Deep face analysis
* Face Detection: Retinaface by [deepinsight/insightface](https://github.com/deepinsight/insightface)
* Face Recognition: Arcface by [deepinsight/insightface](https://github.com/deepinsight/insightface)

### GRPC Server
* Tonic [hyperium/tonic](https://github.com/hyperium/tonic)
* Protobuf [tokio-rs/prost](https://github.com/tokio-rs/prost)

## Project Structure

* [retinaface](./retinaface): Rust bindings for Arcface and Retinaface C++ implementations.
  * [retinaface/libretinaface](./retinaface/libretinaface): TensorRT implementation of Arcface and Retinaface. Based on [wang-xinyu/tensorrtx](https://github.com/wang-xinyu/tensorrtx).
* [proto](./proto): Protobuf definitions for GRPC server.
* [rs-genproto](./rs-genproto): library that generates rust protobuf for GRPC server.
* [server](./server): Implementation of inference GRPC server.
