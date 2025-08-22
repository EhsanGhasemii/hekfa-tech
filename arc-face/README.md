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



# Project Installation using Docker with GPU, CUDA, cuDNN, OpenCV, and ONNX Models

This document provides step-by-step instructions for setting up the project in a Docker container with GPU support, CUDA 11.8, cuDNN 8.6, OpenCV, and ONNX models.

## Prerequisites

* Docker installed on the host machine.
* NVIDIA GPU with NVIDIA Container Toolkit.
* X11 server running on the host for graphical applications.

## Step 1: Create Docker Container with Graphics Support

```bash
docker run --gpus all -d --name ubuntu -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ubuntu
```

## Step 2: Download Required File from Google Docs

* Download the file from [Google Docs](YOUR_GOOGLE_DOC_LINK_HERE).

## Step 3: Access Docker Container and Create Project Directory

```bash
docker exec -it ubuntu bash
mkdir -p /app/projects/
```

## Step 4: Transfer CUDA Installer to Docker Container

```bash
docker cp cuda_11.8.0_520.61.05_linux.run ubuntu:/app/projects/
```

## Step 5: Install CUDA Toolkit 11.8

```bash
apt-get update
apt-get install -y libxml2
chmod +x cuda_11.8.0_520.61.05_linux.run
./cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Step 6: Download and Install cuDNN 8.6

* Download cuDNN 8.6 from [Google Drive](YOUR_CUDNN_LINK_HERE) and transfer:

```bash
docker cp cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz ubuntu:/app/projects/
```

* Install cuDNN:

```bash
apt-get update
apt-get install -y xz-utils
tar -xvf cudnn-linux-x86_64-8.6.0.163_cuda11-archive.tar.xz
cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda-11.8/include/
cp cudnn-*-archive/lib/libcudnn* /usr/local/cuda-11.8/lib64/
chmod a+r /usr/local/cuda-11.8/include/cudnn*.h /usr/local/cuda-11.8/lib64/libcudnn*
```

## Step 7: Download Project Archives and Transfer

* Download [hekfa-tech.tar](YOUR_HEKFA_LINK_HERE) and [torcpp.tar](YOUR_TORCPP_LINK_HERE) and transfer:

```bash
docker cp hekfa-tech.tar ubuntu:/app/projects/
docker cp torcpp.tar ubuntu:/app/projects/
```

## Step 8: Install CMake

```bash
apt install cmake
```

## Step 9: Install OpenGL and GTK Dependencies

```bash
grep -R "ubuntu-toolchain-r" /etc/apt/sources.list /etc/apt/sources.list.d/
rm -f /etc/apt/sources.list.d/*toolchain*.list
apt-get update
apt-get install -y \
    libgtk-3-dev libcanberra-gtk3-dev \
    libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev
```

## Step 10: Install OpenCV with FFMPEG and OpenGL Support

```bash
cd /app/projects/hekfa-tech/arc-face/opencv-4.x/
rm -rf build
mkdir -p build
cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_FFMPEG=ON \
      -D WITH_GTK=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      ..
make -j$(nproc)
make install
ldconfig
```

## Step 11: Download and Install ONNX Models

* Download ONNX models from [Google Drive](YOUR_ONNX_MODELS_LINK_HERE) and prepare the directory:

```bash
mkdir -p /root/.insightface/models/buffalo_l/
```

* Transfer models from your local machine:

```bash
docker cp buffalo_l.zip ubuntu:/root/.insightface/models/buffalo_l/
```

* Unzip models inside the container:

```bash
cd /root/.insightface/models/buffalo_l/
apt install unzip
unzip buffalo_l.zip
```

## Step 12: Enable Fast-Forward Graphics Mode

```bash
xhost +local:root
xhost +local:docker
```

## Step 13: Run the Project

1. Extract embeddings of a single image and show the image:

```bash
./build/core image.jpg
```

2. Extract embeddings of two images and compare faces:

```bash
./build/core image1.jpg image2.jpg
```

3. Extract embeddings of an input video and show faces in real-time:

```bash
./build/core video.mp4
```

