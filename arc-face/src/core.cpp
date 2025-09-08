#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <tuple>
#include <map>
#include <sqlite3.h>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
extern "C" {
    #include "darknet.h"
}

#include "vector.h"             // library to print our matrices
#include "cxxopts.hpp"


// YOLO settings
const std::string CFG_FILE = "/root/.insightface/models/yolo/yolov3.cfg";
const std::string WEIGHTS_FILE = "/root/.insightface/models/yolo/yolov3.weights";
const std::string NAMES_FILE = "/root/.insightface/models/yolo/coco.names";
const float CONF_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.4;
const std::string CONFIG_FILE = "/root/.insightface/models/yolo/source.txt";
const int IMG_FRAME_NUM = 1000; 
const float COSINE_THR = 0.30;

size_t NUM_STREAMS = 1;
cv::Size FRAME_SIZE(800, 600);
double DISPLAY_TIME = 2.0; 
int YOLO_PROCESS_INTERVAL = 5;
std::vector<std::string> STREAM_URLS;

std::atomic<bool> running(true);
std::mutex queue_mutex;
std::queue<std::pair<int, cv::Mat>> frame_queue;


bool fileExists(const std::string& path) {
  std::ifstream file(path);
  return file.good();
}

// 
std::string getTime() {
    std::time_t now = std::time(nullptr);
    std::tm* now_tm = std::localtime(&now);

    // Add 03:56:17 to the current time
    now_tm->tm_hour;// += 3;
    now_tm->tm_min;// += 56;
    now_tm->tm_sec;// += 17;
    std::mktime(now_tm); // Normalize the time structure

    char buffer[80];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", now_tm);

    return std::string(buffer);
}

bool readConfig() {
  std::ifstream file(CONFIG_FILE);
  if (!file.is_open()) return false;

  std::string line;
  int width = 0, height = 0;
  while (getline(file, line)) {
    std::stringstream ss(line);
    std::string key;
    std::string value;
    if (getline(ss, key, '=') && getline(ss, value)) {
      if (key == "NUM_STREAMS") NUM_STREAMS = std::stoul(value);
      else if (key == "FRAME_WIDTH") width = std::stoi(value);
      else if (key == "FRAME_HEIGHT") height = std::stoi(value);
      else if (key == "DISPLAY_TIME") DISPLAY_TIME = std::stod(value);
      else if (key == "YOLO_PROCESS_INTERVALL") YOLO_PROCESS_INTERVAL = std::stoi(value);
      else if (key.find("STREAM_URL_") == 0) STREAM_URLS.push_back(value);
    }
  }
  file.close();
  FRAME_SIZE = cv::Size(width, height);
  return NUM_STREAMS > 0 && width > 0 && height > 0 && STREAM_URLS.size() == NUM_STREAMS;
} // bool readConfig()

// function to check input file is an image or not
bool isImage(const std::string& filename) {
    std::string ext;
    size_t pos = filename.find_last_of(".");
    if (pos != std::string::npos) {
        ext = filename.substr(pos + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    return (ext == "jpg"   || ext == "jpeg" || ext == "png" || 
            ext == "bmp"   || ext == "tiff" || ext == "gif" ||
            ext == "pjpeg" || ext == "webp");
}

// function to check input file is an image or not
bool isDB(const std::string& filename) {
    std::string ext;
    size_t pos = filename.find_last_of(".");
    if (pos != std::string::npos) {
        ext = filename.substr(pos + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    return (ext == "db");
}

// function to check input file is a video or not
bool isVideo(const std::string& filename) {
    std::string ext;
    size_t pos = filename.find_last_of(".");
    if (pos != std::string::npos) {
        ext = filename.substr(pos + 1);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    }

    return (ext == "mp4" || ext == "avi" || ext == "mkv" || 
            ext == "mov" || ext == "flv" || ext == "wmv");
}


void captureStream(int idx, const std::string& url) {
  if (isVideo(url)) { 
    std::cout << "url: " << url << std::endl; 
    cv::VideoCapture cap(url);
    if (!cap.isOpened()) {
      std::cerr << "Cannot open stream " << idx << " : " << url << std::endl;
      return;
    }

    cap.set(cv::CAP_PROP_BUFFERSIZE, 3);

    while (running) {
      cv::Mat frame;
      if (cap.read(frame) && !frame.empty()) {
        cv::resize(frame, frame, FRAME_SIZE);
        std::lock_guard<std::mutex> lock(queue_mutex);
        frame_queue.push({idx, frame.clone()});
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    cap.release();
  } // if (isVideo(url))

  else if (isImage(url)) { 
    cv::Mat img = cv::imread(url);
    for (uint32_t i = 0; i < IMG_FRAME_NUM; ++i) { 
      if (!img.empty()) { 
        cv::resize(img, img, FRAME_SIZE);
        std::lock_guard<std::mutex> lock(queue_mutex);
        frame_queue.push({idx, img});
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  } // else if (isImage(url))

  else {
      std::cerr << "Cannot open stream " << idx << " : " << url << std::endl;
      return; 
  } // else
} // void captureStream


void captureStreams(int idx, const std::string& url, const std::string& url2) {
  if (isImage(url) && isImage(url2)) { 
    cv::Mat img  = cv::imread(url);
    cv::Mat img2 = cv::imread(url2);
    for (uint32_t i = 0; i < IMG_FRAME_NUM; ++i) { 
      if (!img.empty()) { 
        cv::Mat hcat; 
        cv::resize(img, img, FRAME_SIZE);
        cv::resize(img2, img2, FRAME_SIZE);
        cv::hconcat(img, img2, hcat); 
        std::lock_guard<std::mutex> lock(queue_mutex);
        frame_queue.push({idx, hcat});
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  } // else if (isImage(url))

  else {
      std::cerr << "Cannot open stream" << idx << ": " << url << ", " << url2 << std::endl;
      return; 
  } // else
} // void captureStreams


void create_anchor_centers(
  float** anchor_centers_x,
  float** anchor_centers_y,
  
  int height, 
  int width, 
  int stride, 
  int _num_anchors
) { 

  * anchor_centers_x = (float*) malloc(height * width * _num_anchors * sizeof(float));
  * anchor_centers_y = (float*) malloc(height * width * _num_anchors * sizeof(float));

  for (uint32_t i = 0; i < height; ++i) { 
    for (uint32_t j = 0; j < width; ++j) { 
      for (uint32_t k = 0; k < _num_anchors; ++k) { 
        (*anchor_centers_x)[i * width * _num_anchors + j * _num_anchors + k] = j * stride; 
        (*anchor_centers_y)[i * width * _num_anchors + j * _num_anchors + k] = i * stride; 
      } // iterate over k from 0 to width
    } // iterate over j from 0 to height
  } // iterate over i from 0 to _num_anchors
} // void create_anchor_centers


std::vector<int> np_where(
  float* data_in, 
  int size, 
  float det_thresh
) { 
  
  // create output result
  std::vector<int> out; 

  for (uint32_t i = 0; i < size; ++i) { 
    if (data_in[i] > det_thresh) { 
      out.push_back(i); 
    } 
  } // iterate over i from 0 to size

  return out; 

} // std::vector<float> np_where

std::vector<std::vector<float>> distance2bbox(
  float** anchor_centers_x, 
  float** anchor_centers_y, 
  float** bbox_preds, 
  uint32_t size
) { 

  // create output matrix
  std::vector<std::vector<float>> out; 

  for (uint32_t i = 0; i < size; ++i) { 
    std::vector<float> temp; 
    float x1 = (*anchor_centers_x)[i] - (*bbox_preds)[4*i+0]; 
    float y1 = (*anchor_centers_y)[i] - (*bbox_preds)[4*i+1]; 
    float x2 = (*anchor_centers_x)[i] + (*bbox_preds)[4*i+2]; 
    float y2 = (*anchor_centers_y)[i] + (*bbox_preds)[4*i+3]; 

    temp.push_back(x1); 
    temp.push_back(y1); 
    temp.push_back(x2); 
    temp.push_back(y2); 

    out.push_back(temp); 
  } // iterate over i from 0 to size

  return out; 

} // std::vector<std::vector<int>> distance2bbox 

void np_extract(
  float* scores,
  std::vector<std::vector<float>> bboxes, 
  std::vector<std::vector<float>> kpss, 
  std::vector<int> pos_inds, 
  std::vector<std::pair<float, std::pair<std::vector<float>, std::vector<float>>>>& out, 
  float det_scale = 1.0                                   
) { 

  for (uint32_t i = 0; i < pos_inds.size(); ++i) { 
    std::vector<float> bbox = bboxes[pos_inds[i]]; 
    for (auto &x: bbox) { 
      x /= det_scale;
    } 
    std::vector<float> kps = kpss[pos_inds[i]]; 
    for (auto &x: kps) { 
      x /= det_scale;
    } 
    float score = scores[pos_inds[i]];
    std::pair<std::vector<float>, std::vector<float>> data_temp = {bbox, kps}; 
    std::pair<float, std::pair<std::vector<float>, std::vector<float>>> temp = {score, data_temp}; 
    out.push_back(temp); 
  } // iterate over i from 0 to pos_inds size

} // std::vector<std::vector<float>> np_extract


std::vector<std::pair<float, std::pair<std::vector<float>, std::vector<float>>>> np_extract(
  std::vector<std::pair<float, std::pair<std::vector<float>, std::vector<float>>>>& data, 
  std::vector<uint32_t> pos_inds 
) { 

  // create output vector
  std::vector<std::pair<float, std::pair<std::vector<float>, std::vector<float>>>> out; 

  for (uint32_t i = 0; i < pos_inds.size(); ++i) { 
    std::pair<float, std::pair<std::vector<float>, std::vector<float>>> temp = data[pos_inds[i]]; 
    out.push_back(temp); 
  } // iterater over i from 0 to size of pos_inds

  return out; 
} // std::vector<std::vector<float>> np_extract



std::vector<std::vector<float>> distance2kps(
  float** anchor_centers_x, 
  float** anchor_centers_y, 
  float** kps_preds, 
  int kps_rows, 
  int kps_cols
) { 

  // create output matrix
  std::vector<std::vector<float>> out; 

  for (uint32_t i = 0; i < kps_rows; ++i) { 
    std::vector<float> temp; 
    for (uint32_t j = 0; j < kps_cols; j=j+2) { 
      float px = (*anchor_centers_x)[i] + (*kps_preds)[i * kps_cols + j]; 
      float py = (*anchor_centers_y)[i] + (*kps_preds)[i * kps_cols + j+1]; 
      temp.push_back(px); 
      temp.push_back(py); 
    } 
    out.push_back(temp); 
  } 

  return out; 
} // std::vector<std::vector<float>> distance2kps 

void argsort(std::vector<std::pair<float, std::pair<std::vector<float>, std::vector<float>>>>& data) {
  std::sort(data.begin(), data.end(),
     [](auto &a, auto &b) { return a.first > b.first; });
}

std::vector<uint32_t> nms(
  std::vector<std::pair<float, std::pair<std::vector<float>, std::vector<float>>>> pre_det, 
  float nms_thresh
) { 

  // create output vector 
  std::vector<uint32_t> keep; 

  std::vector<float> areas; 
  for (uint32_t j = 0; j < pre_det.size(); ++j) { 
    float x1 = pre_det[j].second.first[0]; 
    float y1 = pre_det[j].second.first[1]; 
    float x2 = pre_det[j].second.first[2]; 
    float y2 = pre_det[j].second.first[3]; 
    float temp = (x2-x1+1) * (y2-y1+1); 
    areas.push_back(temp); 
  } // iterate over j from 0 to size of pre_det

  std::vector<uint32_t> order; 
  std::vector<uint32_t> order_temp; 
  for (uint32_t j = 0; j < pre_det.size(); ++j) { 
    order.push_back(j); 
  } // iterater over j from 0 to size of pre_det

  while (order.size() > 0) { 

    uint32_t i = order[0]; 
    keep.push_back(i); 
    float x1 = pre_det[i].second.first[0]; 
    float y1 = pre_det[i].second.first[1]; 
    float x2 = pre_det[i].second.first[2]; 
    float y2 = pre_det[i].second.first[3]; 

  //   printf("(x1, y1): (%f, %f)\n(x2, y2): (%f, %f)\n", x1, y1, x2, y2); 
    order_temp.clear(); 
    for (uint32_t j = 1; j < order.size(); ++j) { 
      float xx1 = std::max(x1, pre_det[order[j]].second.first[0]); 
      float yy1 = std::max(y1, pre_det[order[j]].second.first[1]); 
      float xx2 = std::min(x2, pre_det[order[j]].second.first[2]); 
      float yy2 = std::min(y2, pre_det[order[j]].second.first[3]); 

      float w = std::max(static_cast<float>(0.0), xx2-xx1+1); 
      float h = std::max(static_cast<float>(0.0), yy2-yy1+1); 

      float inter = w * h;
      float ovr = inter / (areas[i] + areas[order[j]] - inter); 

      if (ovr <= nms_thresh) { 
        order_temp.push_back(order[j]); 
      } 

    } // iterate over j from 0 to size of pre_det
    order = order_temp; 
  } // while (order.size() > 0) 

  return keep; 
} // void nms 


cv::Mat arcface_dst = (cv::Mat_<float>(5, 2) <<
    38.2946f, 51.6963f,
    73.5318f, 51.5014f,
    56.0252f, 71.7366f,
    41.5493f, 92.3655f,
    70.7299f, 92.2041f
);

// trnasform
std::pair<cv::Mat, cv::Mat> transform(
    const cv::Mat& data,
    const cv::Point2f& center,
    int output_size,
    float scale,
    float rotation)
{
    float scale_ratio = scale;
    float rot = rotation * CV_PI / 180.0f; // degrees → radians

    // --- Step 1: scale matrix
    cv::Mat t1 = (cv::Mat_<double>(3, 3) <<
        scale_ratio, 0, 0,
        0, scale_ratio, 0,
        0, 0, 1);

    // --- Step 2: translate center to origin
    float cx = center.x * scale_ratio;
    float cy = center.y * scale_ratio;
    cv::Mat t2 = (cv::Mat_<double>(3, 3) <<
        1, 0, -cx,
        0, 1, -cy,
        0, 0, 1);

    // --- Step 3: rotation
    double cos_r = cos(rot);
    double sin_r = sin(rot);
    cv::Mat t3 = (cv::Mat_<double>(3, 3) <<
        cos_r, -sin_r, 0,
        sin_r,  cos_r, 0,
        0,      0,     1);

    // --- Step 4: translate back to output center
    cv::Mat t4 = (cv::Mat_<double>(3, 3) <<
        1, 0, output_size / 2.0,
        0, 1, output_size / 2.0,
        0, 0, 1);

    // --- Final affine transformation matrix
    cv::Mat T = t4 * t3 * t2 * t1;  // order matches Python's (t1 + t2 + t3 + t4)

    // OpenCV warpAffine needs 2x3 matrix
    cv::Mat M = T(cv::Rect(0, 0, 3, 2)).clone();

    // --- Warp the image
    cv::Mat cropped;
    cv::warpAffine(data, cropped, M, cv::Size(output_size, output_size),
                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0));

    return {cropped, M};
} // std::pair<cv::Mat, cv::Mat> transform



// Create ModelHandler
struct ModelHandler { 
  Ort::Env env;
  Ort::SessionOptions session_options;   
  Ort::Session session;
  std::string input_name;
  std::vector<std::string> output_names;

  ModelHandler(
    const std::string& model_path, 
    const std::string& name="onnx_model", 
    const bool device=false
  )
    : env(ORT_LOGGING_LEVEL_WARNING, name.c_str()),
      session_options(), 
      session(env, model_path.c_str(), session_options)
  {
    Ort::AllocatorWithDefaultOptions allocator;

    // model name
    std::cout << name << " model has been initiallized." << std::endl; 

    if (device) { 
      // Enable CUDA (GPU) execution provider
      OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);

      // Re-create the session with CUDA options
      session = Ort::Session(env, model_path.c_str(), session_options);
    }

    // Inputs
    size_t num_inputs = session.GetInputCount();
    std::cout << "Number of inputs: " << num_inputs << std::endl;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    input_name = input_name_ptr.get();
    std::cout << "Input name: " << input_name << std::endl;

    // Outputs
    size_t num_outputs = session.GetOutputCount();
    std::cout << "Number of outputs: " << num_outputs << std::endl;

    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name_ptr = session.GetOutputNameAllocated(i, allocator);
        std::string output_name = output_name_ptr.get();
        output_names.emplace_back(output_name);
        std::cout << "Output " << i << " name: " << output_name << std::endl;
    }

  } // ModelHandler

  std::vector<Ort::Value> run(
    const cv::Mat& blob, 
    const size_t& input_tensor_size, 
    const std::vector<int64_t>& input_tensor_shape
  ) { 

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        reinterpret_cast<float*>(blob.data),
        input_tensor_size,
        input_tensor_shape.data(),
        input_tensor_shape.size()
    );

    const char* in_name[] = { input_name.c_str() };   // one input name
    std::vector<const char*> out_names;

    for (const auto& name : output_names) {
        out_names.push_back(name.c_str());  // Convert std::string to const char*
    }

    // 256: Run detection_session
    auto net_outs = session.Run(
        Ort::RunOptions{nullptr},
        in_name,
        &input_tensor,
        1,
        out_names.data(),
        out_names.size()
    );

    return net_outs;
  } // void run 


}; // struct ModelHandler

// Create FaceApp
struct FaceApp { 
  ModelHandler detection_model; 
  ModelHandler recognition_model; 
  ModelHandler genderage_model; 

  FaceApp(
    const std::string& detection_model_path, 
    const std::string& recognition_model_path, 
    const std::string& genderage_model_path, 
    const bool device=false
  )
    : detection_model  (detection_model_path  , "Detection"  , device),
      recognition_model(recognition_model_path, "Recognition", device), 
      genderage_model  (genderage_model_path  , "GenderAge"  , device)
  {}

  std::pair<std::pair<std::vector<std::vector<float>>, 
                      std::vector<std::pair<bool,int>>>, 
            std::vector<std::pair<std::pair<int, int>,
                                  std::pair<int, int>>>> getEmbeddings(
    const cv::Mat& img
  ) { 

    // Detection model initiallization  
    float input_mean = 127.5f;
    float input_std = 128.0f;

    int fmc = 3;
    std::vector<int> _feat_stride_fpn = {8, 16, 32};
    int _num_anchors = 2;
    bool use_kps = true;
    float nms_thresh = 0.4;
    float det_thresh = 0.5;

    // Assuming input_size is std::pair<int, int> as (width, height)
    std::pair<int64_t, int64_t> input_size = {640, 640};  // example, replace with your actual input size

    // Compute aspect ratios
    float im_ratio = static_cast<float>(img.rows) / img.cols;  // image height / width
    float model_ratio = static_cast<float>(input_size.second) / input_size.first;  // height / width

    int new_width, new_height;

    if (im_ratio > model_ratio) {
        new_height = static_cast<int>(input_size.second);
        new_width = static_cast<int>(new_height / im_ratio);
    } else {
        new_width = static_cast<int>(input_size.first);
        new_height = static_cast<int>(new_width * im_ratio);
    }

    float det_scale = static_cast<float>(new_height) / img.rows;

    // Resize image to new_width x new_height
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_width, new_height));

    // Create black image with input size (height x width x 3)
    cv::Mat det_img = cv::Mat::zeros(input_size.second, input_size.first, CV_8UC3);

    // Copy resized image to the top-left corner of det_img
    resized_img.copyTo(det_img(cv::Rect(0, 0, new_width, new_height)));

    // 254: Get input size from det_img
    cv::Size bolb_input_size(det_img.cols, det_img.rows);  // reversed shape

    // 255: Create blob
    cv::Mat blob = cv::dnn::blobFromImage(
        det_img,                        // image
        1.0 / input_std,               // scalefactor
        bolb_input_size,                    // size
        cv::Scalar(input_mean, input_mean, input_mean),  // mean subtraction
        true,                          // swapRB
        false                          // crop
    );

    // 258-259: Blob shape
    int input_height = blob.size[2];
    int input_width = blob.size[3];

    // 256: Prepare input tensor
    std::vector<int64_t> input_tensor_shape = {1, blob.size[1], blob.size[2], blob.size[3]};
    size_t input_tensor_size = 1;
    for (auto dim : input_tensor_shape) input_tensor_size *= dim;

    auto net_outs = detection_model.run(
      /*const cv::Mat& blob,                           */ blob, 
      /*const size_t& input_tensor_size,               */ input_tensor_size, 
      /*const std::vector<int64_t>& input_tensor_shape */ input_tensor_shape
    ); 

    // create data for bbox_preds
    std::vector<int> sizes = {12800, 3200, 800}; 
    std::vector<float> scores_list;
    std::vector<std::vector<float>> bboxes_list; 
    std::vector<std::pair<float, std::pair<std::vector<float>, std::vector<float>>>> pre_det;
    std::vector<std::vector<float>> kpss_list; 

    for (uint32_t i = 0; i < _feat_stride_fpn.size(); ++i) { 

      int stride = _feat_stride_fpn[i]; 

      // Extract scores matrix ===============================
      Ort::Value& first_output = net_outs[i];
      Ort::TensorTypeAndShapeInfo shape_info = first_output.GetTensorTypeAndShapeInfo();
      std::vector<int64_t> shape = shape_info.GetShape();
      int scores_len = shape_info.GetElementCount();

      // transfer the bbox_preds tensor to simple arrays 
      float* scores = first_output.GetTensorMutableData<float>();
      
      int rows = static_cast<int>(shape[0]);
      int cols = static_cast<int>(shape[1]);

      // Extract bbox_preds matrix ===============================
      Ort::Value& second_output = net_outs[i+fmc];
      shape_info = second_output.GetTensorTypeAndShapeInfo();
      shape = shape_info.GetShape();
      int bbox_preds_len = shape_info.GetElementCount();

      // transfer the bbox_preds tensor to simple arrays 
      float* bbox_preds = second_output.GetTensorMutableData<float>();
      for (uint32_t s = 0; s < bbox_preds_len; ++s) {
        bbox_preds[s] *= stride; 
      }
      
      rows = static_cast<int>(shape[0]);
      cols = static_cast<int>(shape[1]);

      // Extract kps_preds matrix ===============================
      Ort::Value& third_output = net_outs[i+2*fmc];
      shape_info = third_output.GetTensorTypeAndShapeInfo();
      shape = shape_info.GetShape();
      int kps_preds_len = shape_info.GetElementCount();

      // transfer the bbox_preds tensor to simple arrays 
      float* kps_preds = third_output.GetTensorMutableData<float>();

      for (size_t s=0; s < kps_preds_len; ++s) {
          kps_preds[s] *= stride;
      }
      
      int kps_rows = static_cast<int>(shape[0]);
      int kps_cols = static_cast<int>(shape[1]);

      // =======================================================
      int width = input_width / stride; 
      int height = input_height / stride; 
      int K = height * width; 

      float* anchor_centers_x; 
      float* anchor_centers_y; 

      create_anchor_centers(
         /* float** anchor_centers_x */ &anchor_centers_x, 
         /* float** anchor_centers_y */ &anchor_centers_y, 
         /* int height               */ height, 
         /* int width                */ width, 
         /* int stride,              */ stride, 
         /* int _num_anchors         */ _num_anchors
      ); 

      std::vector<int> pos_inds = np_where(
        /*float** data_in, */ scores, 
        /*int size,        */ scores_len, 
        /*float det_thresh */ det_thresh
      ); 

      std::vector<std::vector<float>> bboxes = distance2bbox(
        /* float** anchor_centers_x, */ &anchor_centers_x, 
        /* float** anchor_centers_y, */ &anchor_centers_y, 
        /* float** bbox_preds        */ &bbox_preds, 
        /* uint32_t size             */ height * width * _num_anchors
      ); 

      if (use_kps) { 
        std::vector<std::vector<float>> kpss = distance2kps(
          /* float** anchor_centers_x, */ &anchor_centers_x, 
          /* float** anchor_centers_y, */ &anchor_centers_y, 
          /* float** kps_preds         */ &kps_preds, 
          /* uint32_t kps_rows         */ kps_rows, 
          /* uint32_t kps_cols         */ kps_cols
        ); 

        np_extract(
          /* float* scores,                                          */ scores, 
          /* std::vector<std::vector<float>> bboxes,                 */ bboxes, 
          /* std::vector<std::vector<float>> kpss,                   */ kpss, 
          /* std::vector<int> pos_inds,                              */ pos_inds, 
          /* std::vector<std::pair<float, std::vector<float>>>& out, */ pre_det,
          /* float det_scale = 1.0                                   */ det_scale 
        ); 

      } // if (use_kps)

    } // for i from 0 to _feat_stride_fpn size

    argsort(pre_det); 

    std::vector<uint32_t> keep = nms(
      /* std::vector<std::pair<float, std::vector<float>>> pre_det */ pre_det, 
      /* float nms_thresh                                          */ nms_thresh
    ); 

    std::vector<std::pair<float, std::pair<std::vector<float>, std::vector<float>>>> det = np_extract(
      /* std::vector<std::pair<float, std::vector<float>>>& out, */ pre_det,
      /* std::vector<int> pos_inds,                              */ keep 
    ); 

    // print result
    vec::draw_line(); 

    // ====================================================================
    input_mean = 127.5f;
    input_std  = 127.5f;

    Ort::TypeInfo input_type_info = recognition_model.session.GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_shape = tensor_info.GetShape();
    cv::Size input_size2((int)input_shape[3], (int)input_shape[2]); // width, height

    // create output vector
    std::vector<std::vector<float>> embeddings; 

    for (const auto x: det) { 
      std::vector<float> kps = x.second.second; 

      int image_size = input_size2.width; 
      float ratio; 
      float diff_x;
      if (image_size % 112 == 0) {
        ratio = static_cast<float>(image_size) / 112.0; 
        diff_x = 0.0; 
      } // if (image_size % 112 == 0)
      else { 
        ratio = static_cast<float>(image_size) / 128.0; 
        diff_x = 8.0 * ratio;
      } // else

      // !!!!!!!!!!AVOID HARD CODING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      cv::Mat kps_mat(5, 2, CV_32F, kps.data());
      // !!!!!!!!!!AVOID HARD CODING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      cv::Mat dst = arcface_dst * ratio;
      for (int i = 0; i < dst.rows; i++) {
          dst.at<float>(i, 0) += diff_x;
      }

      cv::Mat M = cv::estimateAffinePartial2D(kps_mat, dst);
//       cv::Mat M = (cv::Mat_<float>(2, 3) <<
//           0.39182017f, 0.34702577f, -227.6744765f, 
//           -0.34702577f, 0.39182017f, 192.44877767f
//       );

      cv::Mat warped;
      cv::warpAffine(
          img,           // input image (cv::Mat)
          warped,        // output image
          M,             // 2x3 transform matrix (cv::Mat)
          cv::Size(image_size, image_size),  // output size (width, height)
          cv::INTER_LINEAR,                  // interpolation method
          cv::BORDER_CONSTANT,               // border handling
          cv::Scalar(0.0, 0.0, 0.0)           // borderValue (black)
      );

      cv::Mat blob = cv::dnn::blobFromImage(
          warped,                         // input image
          1.0 / input_std,                 // scale factor
          input_size2,                     // target size
          cv::Scalar(input_mean, input_mean, input_mean), // mean subtraction
          true,                            // swapRB
          false                            // crop
      );

      std::vector<int64_t> input_shape = {1, blob.size[1], blob.size[2], blob.size[3]}; 
   
      auto output_tensors = recognition_model.run(
        /*const cv::Mat& blob,                           */ blob, 
        /*const size_t& input_tensor_size,               */ blob.total(), 
        /*const std::vector<int64_t>& input_tensor_shape */ input_shape
      ); 

      // 4. Get the embedding result
      float* embedding_data = output_tensors.front().GetTensorMutableData<float>();

      // Optional: store embedding in a std::vector<float>
      size_t embedding_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
      std::vector<float> embedding(embedding_data, embedding_data + embedding_size);

      embeddings.push_back(embedding); 

    } // for (const auto x: det)


    // Attribute model Initiallization ===========================
    input_mean = 0.0; 
    input_std = 1.0; 
    input_size = {96, 96}; 

    cv::Size b_input_size(96, 96); 


    std::vector<std::pair<bool, int>> gender_age; 
    for (const auto x: det) { 
 
      std::vector<float> bbox = x.second.first; 
      float w = bbox[2] - bbox[0]; 
      float h = bbox[3] - bbox[1]; 

      cv::Point2f center((bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2);

      float rotate = 0.0; 
      float _scale = input_size.first / (std::max(w, h) * 1.5); 

      auto [aimg, M] = transform(img, center, input_size.first, _scale, rotate);

      // Create blob
      cv::Mat blob = cv::dnn::blobFromImage(
          aimg,                                            // image
          1.0 / input_std,                                 // scalefactor
          b_input_size,                                    // size
          cv::Scalar(input_mean, input_mean, input_mean),  // mean subtraction
          true,                                            // swapRB
          false                                            // crop
      );
 
      std::vector<int64_t> input_shape = {1, blob.size[1], blob.size[2], blob.size[3]}; 

      auto pred = genderage_model.run(
        /*const cv::Mat& blob,                           */ blob, 
        /*const size_t& input_tensor_size,               */ blob.total(), 
        /*const std::vector<int64_t>& input_tensor_shape */ input_shape
      ); 
      float* pred_f = pred[0].GetTensorMutableData<float>(); 
      bool gender = (pred_f[0] > pred_f[1]) ? 0 : 1;  
      int age = pred_f[2] * 100; 

      gender_age.push_back({gender, age}); 
    } // for (const auto x: det)


    // draw rectangles for each faces
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> locs; 
    for (const auto x: det) { 
      std::vector<float> bbox = x.second.first; 
      int x1 = static_cast<int>(bbox[0]); 
      int y1 = static_cast<int>(bbox[1]); 
      int x2 = static_cast<int>(bbox[2]); 
      int y2 = static_cast<int>(bbox[3]); 

      locs.push_back({{x1, y1}, {x2, y2}}); 

      cv::rectangle(img,
                    cv::Point(x1, y1),
                    cv::Point(x2, y2),
                    cv::Scalar(0, 255, 0),  // Green box
                    2);    

    } // for (const auto x: det)

    return {{embeddings, gender_age}, locs}; 
  } // void getEmbeddings

}; // struce FaceApp


// cosine similarity function
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    cv::Mat A(a), B(b);
    return A.dot(B) / (cv::norm(A) * cv::norm(B));
} // float cosine_similarity


// Helper to convert vector<float> to a space-separated string
std::string vectorToString(const std::vector<float>& vec) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
        oss << vec[i];
        if (i + 1 < vec.size()) oss << " ";
    }
    return oss.str();
}

// Convert space-separated string → vector<float>
std::vector<float> stringToVector(const std::string& str) {
    std::vector<float> vec;
    std::istringstream iss(str);
    float val;
    while (iss >> val) {
        vec.push_back(val);
    }
    return vec;
}

// Helper to print all records in the embeddings table
void printAllEmbeddings(sqlite3* db) {
    const char* select_sql =
        "SELECT id, name, date, vector, path, gender, age, semantics, camera FROM embeddings;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, select_sql, -1, &stmt, nullptr) == SQLITE_OK) {
        std::cout << "=== All records in DB ===\n";
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(stmt, 0);

            const unsigned char* name     = sqlite3_column_text(stmt, 1);
            const unsigned char* date     = sqlite3_column_text(stmt, 2);
            const unsigned char* vec_txt  = sqlite3_column_text(stmt, 3);
            const unsigned char* path     = sqlite3_column_text(stmt, 4);
            const unsigned char* gender   = sqlite3_column_text(stmt, 5);
            int age                       = sqlite3_column_int(stmt, 6);
            const unsigned char* semantics= sqlite3_column_text(stmt, 7);
            const unsigned char* camera   = sqlite3_column_text(stmt, 8);

            std::string vec_str = (const char*)vec_txt;
            std::vector<float> vec = stringToVector(vec_str);

            std::cout << "ID: " << id
                      << " | Name: " << (name ? (const char*)name : "")
                      << " | Date: " << (date ? (const char*)date : "")
                      << " | Path: " << (path ? (const char*)path : "")
                      << " | Gender: " << (gender ? (const char*)gender : "")
                      << " | Age: " << age
                      << " | Semantics: " << (semantics ? (const char*)semantics : "")
                      << " | Camera: " << (camera ? (const char*)camera : "")
                      << " | Vector: ";
            vec::print(vec);
        }
        sqlite3_finalize(stmt);
    } else {
        std::cerr << "Error selecting data: " << sqlite3_errmsg(db) << "\n";
    }
}

// Helper to insert into embeddings table with new schema
int insertToEmbeddings(
  sqlite3* db,
  std::vector<float>& embdd,
  const std::string& images_path, 
  const std::string& image_gender, 
  const int& image_age, 
  char*& errMsg
) {
    int id = 0; 
    const char* select_sql = "SELECT id, vector FROM embeddings;";
    sqlite3_stmt* stmt;

    // First check if similar embedding already exists
    if (sqlite3_prepare_v2(db, select_sql, -1, &stmt, nullptr) == SQLITE_OK) {
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            id = sqlite3_column_int(stmt, 0);
            const unsigned char* vec_txt = sqlite3_column_text(stmt, 1); // "vector" column
            std::string vec_str = (const char*)vec_txt;
            std::vector<float> vec = stringToVector(vec_str);

            float sim = cosine_similarity(embdd, vec);
            if (sim > COSINE_THR) {
                sqlite3_finalize(stmt);
                return id; 
            }
        }
        sqlite3_finalize(stmt);
    } else {
        std::cerr << "Error selecting data: " << sqlite3_errmsg(db) << "\n";
        return -1;
    }

    // If not found, insert a new record
    std::string vec_str = vectorToString(embdd);

    const char* insert_sql =
        "INSERT INTO embeddings "
        "(name, date, vector, path, gender, age, semantics, camera) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?);";

    sqlite3_stmt* insert_stmt;
    if (sqlite3_prepare_v2(db, insert_sql, -1, &insert_stmt, nullptr) == SQLITE_OK) {
        // Example values (replace with your actual metadata)
        std::string name = "person_" + std::to_string(id + 1);
        std::string date = getTime();  // could generate current date
        std::string path = images_path + name + ".jpg";
        std::string gender = image_gender;
        int age = image_age;
        std::string semantics = "Null";
        std::string camera = "Canon";

        sqlite3_bind_text(insert_stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(insert_stmt, 2, date.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(insert_stmt, 3, vec_str.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(insert_stmt, 4, path.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(insert_stmt, 5, gender.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int(insert_stmt, 6, age);
        sqlite3_bind_text(insert_stmt, 7, semantics.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(insert_stmt, 8, camera.c_str(), -1, SQLITE_TRANSIENT);

        if (sqlite3_step(insert_stmt) != SQLITE_DONE) {
            std::cerr << "Error inserting data: " << sqlite3_errmsg(db) << "\n";
            return -1; 
        } else {
            std::cout << "Inserted new embedding for: " << name << "\n";
            return id + 1; 
        }
        sqlite3_finalize(insert_stmt);
    } else {
        std::cerr << "Error preparing insert: " << sqlite3_errmsg(db) << "\n";
        return -1; 
    }
}


void processYOLO(network* net, char** class_names, int num_classes, FaceApp& face_app, sqlite3*& db, char*& errMsg) {
  std::string ffmpeg_cmd =
    "ffmpeg -hide_banner -loglevel warning -y "
    "-f rawvideo -vcodec rawvideo -pix_fmt bgr24 -s 640x480 -r 25 -i - "
    "-an -c:v libx264 -preset ultrafast -tune zerolatency "
    "-g 25 -x264-params keyint=25:scenecut=0 "
//     "-pix_fmt yuv420p "
    "-pix_fmt yuv420p output_test.mp4 "
    "-fflags nobuffer -flags low_delay -max_delay 0 -flush_packets 1 "
    "-f mpegts udp://0.0.0.0:1234?pkt_size=1316";

  FILE* ffmpeg_pipe = popen(ffmpeg_cmd.c_str(), "w");
  if (!ffmpeg_pipe) {
    std::cerr << "Error: Cannot open FFmpeg pipe" << std::endl;
    return;
  }
  std::cout << "FFmpeg pipe opened successfully" << std::endl;

  int counter = 0;
  size_t current_stream = 0;
  auto last_switch_time = std::chrono::steady_clock::now();

  while (running) {
    std::cout << counter << " .. " << std::endl; 
    std::pair<int, cv::Mat> data;
    bool has_frame = false;
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (!frame_queue.empty()) {
        data = frame_queue.front();
        frame_queue.pop();
        has_frame = true;
      }
    }

    if (!has_frame) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - last_switch_time).count();
    if (elapsed >= DISPLAY_TIME) {
      current_stream = (current_stream + 1) % NUM_STREAMS;
      last_switch_time = now;
    }

    if (data.first != current_stream) continue;

    cv::Mat frame = data.second;

    if (counter++ % YOLO_PROCESS_INTERVAL != 0) continue;

    // main process 
    std::pair<std::pair<std::vector<std::vector<float>>, 
                        std::vector<std::pair<bool,int>>>, 
              std::vector<std::pair<std::pair<int, int>,
                                    std::pair<int, int>>>> result = face_app.getEmbeddings(frame); 

    std::vector<std::vector<float>> embeddings = result.first.first; 
    std::vector<std::pair<bool,int>> sex = result.first.second; 
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> locs = result.second; 

    for (uint32_t i = 0; i < embeddings.size(); ++i) { 
      printf("%d\t%d\t", sex[i].first, sex[i].second); 
      vec::print(embeddings[i]); 
    }

    // Insert a new embedding
    std::cout << getTime() << std::endl;

    if (!embeddings.empty()) { 
      std::string gender = "Female"; 
      if (sex[0].first) { gender = "Male";}
      std::string path = "/app/images/"; 
      int id = insertToEmbeddings(
        db, 
        embeddings[0], 
        path, 
        gender, 
        sex[0].second, 
        errMsg
      ); 

      // Put the id on final frame 
      cv::putText(frame, std::to_string(id), cv::Point(locs[0].first.first, locs[0].first.second),
                cv::FONT_HERSHEY_SIMPLEX, // font face
                1.0,                      // font scale
                cv::Scalar(235, 125, 125),// color (B,G,R) → white
                2);                       // thickness

      cv::rectangle(frame,
                    cv::Point(locs[0].first.first, locs[0].first.second),
                    cv::Point(locs[0].first.first+25, locs[0].first.second-25),
                    cv::Scalar(0, 0, 255),  // Green box
                    2);    

    }

    cv::Mat output;
    resize(frame, output, cv::Size(640,480));
    fwrite(output.data, 1, output.total() * output.elemSize(), ffmpeg_pipe);
    fflush(ffmpeg_pipe);
  } // while(running)

  pclose(ffmpeg_pipe);
} // void processYOLO



void processYOLO2(network* net, char** class_names, int num_classes, FaceApp& face_app) {
  std::string ffmpeg_cmd =
    "ffmpeg -hide_banner -loglevel warning -y "
    "-f rawvideo -vcodec rawvideo -pix_fmt bgr24 -s 640x480 -r 25 -i - "
    "-an -c:v libx264 -preset ultrafast -tune zerolatency "
    "-g 25 -x264-params keyint=25:scenecut=0 "
//     "-pix_fmt yuv420p "
    "-pix_fmt yuv420p output_test.mp4 "
    "-fflags nobuffer -flags low_delay -max_delay 0 -flush_packets 1 "
    "-f mpegts udp://0.0.0.0:1234?pkt_size=1316";

  FILE* ffmpeg_pipe = popen(ffmpeg_cmd.c_str(), "w");
  if (!ffmpeg_pipe) {
    std::cerr << "Error: Cannot open FFmpeg pipe" << std::endl;
    return;
  }
  std::cout << "FFmpeg pipe opened successfully" << std::endl;

  int counter = 0;
  size_t current_stream = 0;
  auto last_switch_time = std::chrono::steady_clock::now();

  while (running) {
    std::cout << counter << " .. " << std::endl; 
    std::pair<int, cv::Mat> data;
    bool has_frame = false;
    {
      std::lock_guard<std::mutex> lock(queue_mutex);
      if (!frame_queue.empty()) {
        data = frame_queue.front();
        frame_queue.pop();
        has_frame = true;
      }
    }

    if (!has_frame) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      continue;
    }

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - last_switch_time).count();
    if (elapsed >= DISPLAY_TIME) {
      current_stream = (current_stream + 1) % NUM_STREAMS;
      last_switch_time = now;
    }

    if (data.first != current_stream) continue;

    cv::Mat frame = data.second;

    if (counter++ % YOLO_PROCESS_INTERVAL != 0) continue;

    // main process 
    
    // main process 
    std::pair<std::pair<std::vector<std::vector<float>>, 
                        std::vector<std::pair<bool,int>>>, 
              std::vector<std::pair<std::pair<int, int>,
                                    std::pair<int, int>>>> result = face_app.getEmbeddings(frame); 

    std::vector<std::vector<float>> embeddings = result.first.first; 
    std::vector<std::pair<bool,int>> sex = result.first.second; 
    std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>> locs = result.second; 

    for (uint32_t i = 0; i < embeddings.size(); ++i) { 
      printf("%d\t%d\t", sex[i].first, sex[i].second); 
      vec::print(embeddings[i]); 
    }

    float sim = cosine_similarity(embeddings[0], embeddings[1]); 
    std::cout << "Similarity: " << sim << std::endl; 

    cv::Scalar color;
    std::string text;
    if (sim > COSINE_THR) { 
      color = cv::Scalar(0, 255, 0);
      text = "Same person"; 
    } 
    else {
      color = cv::Scalar(0, 0, 255);
      text = "Not Same person"; 
    } 

    cv::Mat output;
    resize(frame, output, cv::Size(640,480));

    // Put the text
      cv::putText(output, text, cv::Point(160, 240),
                cv::FONT_HERSHEY_SIMPLEX, // font face
                1.0,                      // font scale
                color,                    // color (B,G,R) → white
                2);                       // thickness

    fwrite(output.data, 1, output.total() * output.elemSize(), ffmpeg_pipe);
    fflush(ffmpeg_pipe);
  } // while(running)

  pclose(ffmpeg_pipe);
} // void processYOLO2



int main(int argc, char* argv[]) {

    sqlite3* db;
    char* errMsg = nullptr;

    // Open (or create) database
    if (sqlite3_open("embeddings.db", &db) != SQLITE_OK) {
        std::cerr << "Error opening DB: " << sqlite3_errmsg(db) << "\n";
        return 1;
    }

    // Create table if not exists
    const char* create_table_sql =
        "CREATE TABLE IF NOT EXISTS embeddings ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "name TEXT,"
        "date TEXT,"
        "vector TEXT,"
        "path TEXT,"
        "gender TEXT,"
        "age INTEGER,"
        "semantics TEXT,"
        "camera TEXT"
        ");";

//     const char* create_table_sql =
//         "CREATE TABLE IF NOT EXISTS embeddings ("
//         "id INTEGER PRIMARY KEY AUTOINCREMENT,"
//         "label TEXT,"
//         "vector TEXT"
//         ");";
    if (sqlite3_exec(db, create_table_sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "Error creating table: " << errMsg << "\n";
        sqlite3_free(errMsg);
    }

    // print all existing embeddings
    vec::draw_line(); 
    printAllEmbeddings(db); 
    vec::draw_line(); 

    // Initialize FaceApp 
    const std::string detection_model_path = "/root/.insightface/models/buffalo_l/det_10g.onnx";
    const std::string recognition_model_path = "/root/.insightface/models/buffalo_l/w600k_r50.onnx";
    const std::string genderage_model_path = "/root/.insightface/models/buffalo_l/genderage.onnx";

    FaceApp face_app(
      /* const std::string& detection_model_path,   */ detection_model_path,
      /* const std::string& recognition_model_path, */ recognition_model_path,
      /* const std::string& genderage_model_path,   */ genderage_model_path,
      /* const bool device=false                    */ 1
    ); 

    // create a dark net 
    if (!fileExists(CFG_FILE) || !fileExists(WEIGHTS_FILE) || !fileExists(NAMES_FILE)) {
      std::cerr << "Missing YOLO model files!" << std::endl;
      return -1;
    }

    network* net = load_network((char*)CFG_FILE.c_str(), (char*)WEIGHTS_FILE.c_str(), 0);
    set_batch_network(net, 1);

    std::vector<std::string> class_list;
    std::ifstream names_file(NAMES_FILE);
    std::string name;
    while (getline(names_file, name)) class_list.push_back(name);
    names_file.close();

    char** class_names = (char**)calloc(class_list.size(), sizeof(char*));
    for (size_t i=0; i<class_list.size(); ++i) {
        class_names[i] = (char*)calloc(class_list[i].size()+1,sizeof(char));
        strcpy(class_names[i], class_list[i].c_str());
    }

    // main branch to decide what to process in which mode
    vec::draw_line(); 
    try {
      cxxopts::Options options("MyApp", "Process objects or faces from image/video");

      options.add_options()
          ("m,mode", "Mode: 'p' for person, 'c' for car",
              cxxopts::value<std::string>())
          ("i,input", "Input file(s) including image or video",
              cxxopts::value<std::vector<std::string>>())
          ("h,help", "Print usage");

      options.parse_positional({"mode", "input"});

      auto result = options.parse(argc, argv);

      if (result.count("help") || argc == 1) {
        std::cout << "Usage:\n"
          << "  ./test o image.jpg          # Object mode, 1 input\n"
          << "  ./test f face1.jpg          # Face mode, 1 input\n"
          << "  ./test f face1.jpg face2.jpg # Face mode, 2 inputs\n\n";
        std::cout << options.help() << std::endl;
        return 0;
      }

      if (!result.count("mode")) {
          std::cerr << "Error: --mode is required." << std::endl;
          return 1;
      }

      std::string mode = result["mode"].as<std::string>();
      std::vector<std::string> inputs;
      if (result.count("input")) {
          inputs = result["input"].as<std::vector<std::string>>();
      }

      if (inputs.empty()) { 
        if (!readConfig()) {
          std::cerr << "Failed to read source.txt config!" << std::endl;
          return -1;
        } // if (!readConfig())
      } // if (inputs.empty())

      if (mode == "p" || mode == "c" || mode == "pc" || mode == "cp") {
          if (inputs.size() != 1) {
              std::cerr << "Object mode requires exactly 1 input file." << std::endl;
              return 1;
          }
          std::cout << "Running object detection on: " << inputs[0] << std::endl;
      }

      else if (mode == "f") {
        if (inputs.size() > 2) {
          std::cerr << "Face mode requires 1 or 2 input files." << std::endl;
          return 1;
        } // if (inputs.size() > 2)

        else if (inputs.empty()) { 

          std::vector<std::thread> threads;
          for (size_t i=0; i<NUM_STREAMS; ++i) { 
            threads.emplace_back(captureStream, i, STREAM_URLS[i]);
          }

          std::thread yolo_thread(processYOLO, net, class_names, class_list.size(), std::ref(face_app), std::ref(db), std::ref(errMsg));

          std::cout << "Streaming started... Press Ctrl+C to stop." << std::endl;
          for (auto &t : threads) t.join();
          yolo_thread.join();

        } // else if (inputs.empty())

        else if (inputs.size() == 1) {
          
          // extract image file name
          std::string input_file = inputs[0]; 

          std::cout << "Processing an image input file ... ." << std::endl; 

          std::vector<std::thread> threads;
          threads.emplace_back(captureStream, 0, input_file);

          std::thread yolo_thread(processYOLO, net, class_names, class_list.size(), std::ref(face_app), std::ref(db), std::ref(errMsg));

          std::cout << "Streaming started... Press Ctrl+C to stop." << std::endl;
          for (auto &t : threads) t.join();
          yolo_thread.join();

          return 1; 
        } // else if (inputs.size() == 1)

        else if (inputs.size() == 2) { 
          // extract image file name
          std::string input_file = inputs[0]; 
          std::string input_file2 = inputs[1]; 

          std::cout << "Compairing two image input files ... ." << std::endl; 

          std::vector<std::thread> threads;
          threads.emplace_back(captureStreams, 0, input_file, input_file2);

          std::thread yolo_thread(processYOLO2, net, class_names, class_list.size(), std::ref(face_app));

          std::cout << "Streaming started... Press Ctrl+C to stop." << std::endl;
          for (auto &t : threads) t.join();
          yolo_thread.join();

          return 1; 
        } // else if (inputs.size() == 2)
      } // else if (mode == "f")

      else {
          std::cerr << "ERROR) Invalid mode ... ." << std::endl;
          return 1;
      } // else to Invalid mode

    } catch (const std::exception& e) {
      std::cerr << "Error parsing options: " << e.what() << std::endl;
      return 1;
    }

    // Close DB
    sqlite3_close(db);

    // free the dark net
    free_network_ptr(net);
    for (size_t i=0; i<class_list.size(); ++i) free(class_names[i]);
    free(class_names);

    return 0;
}

