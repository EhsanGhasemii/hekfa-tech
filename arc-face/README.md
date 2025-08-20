# onnx-cpp 

You should download propper onnx runtime from [the link](https://pypi.org/project/onnxruntime-gpu/)

and also need to create below file. 
```
nvim onnxruntime-linux-x64-gpu-1.17.3/include/core/providers/cuda/cuda_provider_factory.h
```
and write below headerfile into it. 
```cuda_provider_factory.h
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \param device_id cuda device id, starts from zero.
 */
ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, int device_id);

#ifdef __cplusplus
}
#endif
```


Below Combinations of cuda-drivers and cuda-toolkits has been used.

```
cuda_11.8.0_520.61.05_linux.run
cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
```
