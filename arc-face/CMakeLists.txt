cmake_minimum_required(VERSION 3.10)
project(core)

set(CMAKE_CXX_STANDARD 17)

#set(ONNXRUNTIME_DIR "/home/ehsan/main/projects/onnxruntime-1.17.3/")
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-gpu-1.17.0/")


include_directories(${ONNXRUNTIME_DIR}/include)
include_directories(${ONNXRUNTIME_DIR}/include/core/providers/cuda)
#include_directories(${ONNXRUNTIME_DIR}/include/onnxruntime/core/session)
link_directories(${ONNXRUNTIME_DIR}/lib)
include_directories("/app/projects/torcpp/include")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find SQLite3
find_package(SQLite3 REQUIRED)
include_directories(${SQLite3_INCLUDE_DIRS})

add_executable(core ${CMAKE_CURRENT_LIST_DIR}/src/core.cpp)
target_link_libraries(core
  onnxruntime
  ${OpenCV_LIBS}
  ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
  ${SQLite3_LIBRARIES}
)








