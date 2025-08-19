#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <tuple>
#include <map>
#include <sqlite3.h>

#include "vector.h"             // library to print our matrices


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

  FaceApp(
    const std::string& detection_model_path, 
    const std::string& recognition_model_path, 
    const bool device=false
  )
    : detection_model(detection_model_path, "Detection", device),
      recognition_model(recognition_model_path, "Recognition", device)
  {}

  std::vector<std::vector<float>> getEmbeddings(
    const cv::Mat& img
  ) { 

    // Assumptions from earlier context
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
    vec::print(keep, "keep"); 
    for (const auto x: det) { 
      printf("%f\t", x.first); 
      vec::print(x.second.second); 
    } // for loop over elements of pre_det

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
      std::cout << "M: " << M << std::endl; 

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

      vec::print(embedding, "embedding", 8, false);
      std::cout << "embedding shape: " << embedding.size() << std::endl; 

      embeddings.push_back(embedding); 

      vec::draw_line(40, '-'); 
    } // for (const auto x: det)

    return embeddings; 
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
    const char* select_sql = "SELECT id, label, vector FROM embeddings;";
    sqlite3_stmt* stmt;

    if (sqlite3_prepare_v2(db, select_sql, -1, &stmt, nullptr) == SQLITE_OK) {
        std::cout << "=== All records in DB ===\n";
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            int id = sqlite3_column_int(stmt, 0);
            const unsigned char* lbl = sqlite3_column_text(stmt, 1);
            const unsigned char* vec_txt = sqlite3_column_text(stmt, 2);
            std::string vec_str = (const char*)vec_txt;
            std::vector<float> vec = stringToVector(vec_str);

            std::cout << "ID: " << id
                      << " | Label: " << lbl  
                      << " | Vector: ";
            vec::print(vec); 
        }
        sqlite3_finalize(stmt);
    } else {
        std::cerr << "Error selecting data: " << sqlite3_errmsg(db) << "\n";
    }
}

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
        "label TEXT,"
        "vector TEXT"
        ");";
    if (sqlite3_exec(db, create_table_sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
        std::cerr << "Error creating table: " << errMsg << "\n";
        sqlite3_free(errMsg);
    }

    // print all existing embeddings
    vec::draw_line(); 
    printAllEmbeddings(db); 
    vec::draw_line(); 

    if (argc < 2) { 
      std::cerr << "Usage: " << argv[0] << " <input_file>\n"; 
      return 1; 
    } // if (argc < 2)

    // extract image file name
    std::string input_file = argv[1]; 

    // Initialize FaceApp 
    const std::string detection_model_path = "/root/.insightface/models/buffalo_l/det_10g.onnx";
    const std::string recognition_model_path = "/root/.insightface/models/buffalo_l/w600k_r50.onnx";
    FaceApp face_app(
      /* const std::string& detection_model_path,   */ detection_model_path,
      /* const std::string& recognition_model_path, */ recognition_model_path,
      /* const bool device=false                    */ 1
    ); 

    if (argc == 2) { 

      // Extracting embeddings of an image input file
      if (isImage(input_file)) { 
        std::cout << "Processing an image input file ... ." << std::endl; 

        // Load image
        cv::Mat img = cv::imread(input_file);

        std::vector<std::vector<float>> embeddings = face_app.getEmbeddings(img); 
        vec::print(embeddings, "embeddings", 8, false); 
        std::cout << "embedding size: " << embeddings.size() << " " << embeddings[0].size() << std::endl;

        // Insert a new embedding
        std::string vec_str = vectorToString(embeddings[0]);
        std::string label = "person_001";

        std::string insert_sql =
            "INSERT INTO embeddings (label, vector) VALUES ('" +
            label + "', '" + vec_str + "');";

        if (sqlite3_exec(db, insert_sql.c_str(), nullptr, nullptr, &errMsg) != SQLITE_OK) {
            std::cerr << "Error inserting data: " << errMsg << "\n";
            sqlite3_free(errMsg);
        } else {
            std::cout << "Inserted new embedding for: " << label << "\n";
        }

        return 1; 
      } // if (isImage(input_file))

      // Extracting embeddings of a video input file
      else if (isVideo(input_file)) { 

        std::cout << "Processing a video input file ... ." << std::endl; 

        // Load Video
        std::cout << "input_file: " << input_file << std::endl; 
        cv::VideoCapture cap("/app/hekfa-tech/arc-face/person2.mp4");

        uint32_t i = 0; 
        while (1) {
          std::cout << "i: " << i << std::endl; 

          // create a frame
          cv::Mat frame;
          cap >> frame;
          if (frame.empty()) { 
            break;
          } // if (frame.empty())

          // main core
          std::vector<std::vector<float>> embeddings = face_app.getEmbeddings(frame); 

          cv::Mat image1 ;
          cv::resize(frame, image1, cv::Size( 896 , 416));//, INTER_LINEAR);

          i++; 
        } // while

        return 1; 
      } // else if (isVideo(input_file))

      else {
        std::cout << "Unknown or unsupported file format: " << input_file << std::endl; 
        return 1; 
      } // else after detecting image or video
    } // if (argc == 2)

    else if (argc == 3) { 

      std::string input_file2 = argv[2]; 

      // Extracting embeddings of an image input file
      if (isImage(input_file) && isImage(input_file2)) { 
        std::cout << "Compairing two image input files ... ." << std::endl; 

        // Load image
        cv::Mat img  = cv::imread(input_file );
        cv::Mat img2 = cv::imread(input_file2);

        std::vector<std::vector<float>> embeddings  = face_app.getEmbeddings(img ); 
        std::vector<std::vector<float>> embeddings2 = face_app.getEmbeddings(img2); 

        float sim = cosine_similarity(embeddings[0], embeddings2[0]); 
        std::cout << "Similarity: " << sim << std::endl; 

        return 1; 
      } // if (isImage(input_file))

    } // else if (argc == 3)


    // Close DB
    sqlite3_close(db);

    return 0;
}

