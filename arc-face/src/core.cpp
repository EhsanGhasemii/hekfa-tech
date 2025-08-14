#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <tuple>
#include <map>

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

int main() {
    const std::string detection_model_path = "/root/.insightface/models/buffalo_l/det_10g.onnx";

    // Assumptions from earlier context
    float input_mean = 127.5f;
    float input_std = 128.0f;

    int fmc = 3;
    std::vector<int> _feat_stride_fpn = {8, 16, 32};
    int _num_anchors = 2;
    bool use_kps = true;
    float nms_thresh = 0.4;
    float det_thresh = 0.5;


    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Detection");
    Ort::SessionOptions session_options;
    Ort::Session session(env, detection_model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // Inputs
    size_t num_inputs = session.GetInputCount();
    std::cout << "Number of inputs: " << num_inputs << std::endl;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    std::string input_name = input_name_ptr.get();
    std::cout << "Input name: " << input_name << std::endl;

    // Outputs
    size_t num_outputs = session.GetOutputCount();
    std::cout << "Number of outputs: " << num_outputs << std::endl;

    std::vector<std::string> output_names;
    for (size_t i = 0; i < num_outputs; ++i) {
        auto output_name_ptr = session.GetOutputNameAllocated(i, allocator);
        std::string output_name = output_name_ptr.get();
        output_names.emplace_back(output_name);
        std::cout << "Output " << i << " name: " << output_name << std::endl;
    }

    // Load image
    cv::Mat img = cv::imread("persons.jpg");
    if (img.empty()) {
        std::cerr << "Error loading image!" << std::endl;
        return -1;
    }

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

    // For demonstration:
    printf("new_height: %d\n", new_height); 
    printf("new_width: %d\n", new_width);
    printf("im_ratio: %.8f\n", im_ratio); 
    printf("model_ratio: %.8f\n", model_ratio);

    std::cout << "Original image size: " << img.cols << "x" << img.rows << std::endl;
    std::cout << "Resized image size: " << new_width << "x" << new_height << std::endl;
    std::cout << "Detection image size: " << det_img.cols << "x" << det_img.rows << std::endl;
    std::cout << "Detection scale: " << det_scale << std::endl;

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

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        reinterpret_cast<float*>(blob.data),
        input_tensor_size,
        input_tensor_shape.data(),
        input_tensor_shape.size()
    );

    const char* in_name = input_name.c_str();  // Use actual input name
    std::vector<const char*> out_names;
    for (const auto& name : output_names) {
        out_names.push_back(name.c_str());  // Convert std::string to const char*
    }

    // 256: Run session
    auto net_outs = session.Run(
        Ort::RunOptions{nullptr},
        &in_name,
        &input_tensor,
        1,
        out_names.data(),
        out_names.size()
    );

    // print output of the model
    for (size_t i = 0; i < net_outs.size(); ++i) {
        Ort::Value& output_tensor = net_outs[i];

        if (!output_tensor.IsTensor()) {
            std::cout << "Output " << i << " is not a tensor." << std::endl;
            continue;
        }

        // Get shape
        Ort::TensorTypeAndShapeInfo shape_info = output_tensor.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = shape_info.GetShape();
        size_t total_len = shape_info.GetElementCount();

        std::cout << "Output " << i << " shape: [";
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Assuming float output
        float* float_array = output_tensor.GetTensorMutableData<float>();
    } // iterater over i from 0 to size of net_outputs 

    std::map<std::tuple<int, int, int>, cv::Mat> center_cache;

    Ort::Value& first_output = net_outs[3];
    Ort::TensorTypeAndShapeInfo shape_info = first_output.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = shape_info.GetShape();
    int total_len = shape_info.GetElementCount();

    
    int rows = static_cast<int>(shape[0]);
    int cols = static_cast<int>(shape[1]);

    // Extract raw data
    float* data = first_output.GetTensorMutableData<float>();

    // Print elements
    std::cout << "First Output Tensor (raw values):" << std::endl;
    std::cout << "total len: " << total_len << std::endl; 
    std::cout << "rows: " << rows << std::endl; 
    std::cout << "cols: " << cols << std::endl; 
    std::cout << std::endl;

    // create data for bbox_preds
    std::vector<int> sizes = {12800, 3200, 800}; 
    std::vector<float> scores_list;
    std::vector<std::vector<float>> bboxes_list; 
    std::vector<std::pair<float, std::pair<std::vector<float>, std::vector<float>>>> pre_det;
    std::vector<std::vector<float>> kpss_list; 

    std::cout << "fmc: " << fmc << std::endl; 
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

      std::cout << "stride: " << stride << std::endl; 
      std::cout << "(Ar, c): " << "(" << rows << ", " << cols << ")" << std::endl; 

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

      std::cout << "(Br, c): " << "(" << rows << ", " << cols << ")" << std::endl; 

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

      std::cout << "(Cr, c): " << "(" << rows << ", " << cols << ")" << std::endl; 
      // =======================================================
      int width = input_width / stride; 
      int height = input_height / stride; 
      int K = height * width; 
      std::cout << "height: " << height << std::endl; 
      std::cout << "input_height: " << input_height << std::endl; 
      std::cout << "width: " << width << std::endl; 
      std::cout << "input_width: " << input_width << std::endl; 
      std::cout << "_num_anchors: " << _num_anchors << std::endl; 
      std::cout << "K: " << K << std::endl; 
      std::cout << "bbox_preds_len: " << bbox_preds_len << std::endl; 

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

//       vec::print(anchor_centers_x, height * width * _num_anchors);

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

      std::cout << "==========================" << std::endl;
    } 

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



    // =======================================================================

    std::string model_file = "/root/.insightface/models/buffalo_l/w600k_r50.onnx";
    input_mean = 127.5f;
    input_std  = 127.5f;

    // ==== Load ONNX Runtime session2 ====
    Ort::Env env2(ORT_LOGGING_LEVEL_WARNING, "arcface");
    Ort::SessionOptions session_options2;
    session_options2.SetIntraOpNumThreads(1);

    Ort::Session session2(env2, model_file.c_str(), session_options2);

    // ==== Get input info ====
    Ort::AllocatorWithDefaultOptions allocator2;
    auto input_name_c = session2.GetInputNameAllocated(0, allocator2);
    std::string input_name2 = input_name_c.get();
    std::cout << "input_name2: " << input_name2 << std::endl;

    // store in const char* array for Run
    const char* input_names2[] = { input_name2.c_str() };

    Ort::TypeInfo input_type_info = session2.GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> input_shape = tensor_info.GetShape();
    cv::Size input_size2((int)input_shape[3], (int)input_shape[2]); // width, height

    // ==== Get output info ====
    size_t num_outputs2 = session2.GetOutputCount();
    std::cout << "Number of outputs: " << num_outputs2 << std::endl;
    std::vector<std::string> output_names2;
    std::vector<const char*> output_names2_cstr; // store const char* for Run

    for (size_t i = 0; i < num_outputs2; ++i) {
        auto output_name_ptr = session2.GetOutputNameAllocated(i, allocator2);
        std::string output_name = output_name_ptr.get();
        output_names2.emplace_back(output_name);
        output_names2_cstr.push_back(output_names2.back().c_str()); // keep pointer valid
        std::cout << "Output " << i << " name: " << output_name << std::endl;
    }




    vec::draw_line(5); 
    std::cout << "input_size: " << input_size2.width << std::endl; 
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
      std::cout << "kps_mat: " << kps_mat << std::endl; 

      cv::Mat dst = arcface_dst * ratio;
      for (int i = 0; i < dst.rows; i++) {
          dst.at<float>(i, 0) += diff_x;
      }
      std::cout << "dst: " << dst << std::endl; 

//       cv::Mat M = cv::estimateAffinePartial2D(kps_mat, dst);
      cv::Mat M = (cv::Mat_<float>(2, 3) <<
          0.39182017f, 0.34702577f, -227.6744765f, 
          -0.34702577f, 0.39182017f, 192.44877767f
      );
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

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtDeviceAllocator, OrtMemTypeCPU);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        (float*)blob.data,
        blob.total(),
        input_shape.data(),
        input_shape.size()
    );

//     // 3. Run inference
    auto output_tensors = session2.Run(
        Ort::RunOptions{nullptr},
        input_names2,                               // const char* const*
        &input_tensor,                              // pointer to input tensor
        1,                                          // number of inputs
        output_names2_cstr.data(),                  // const char* const*
        output_names2_cstr.size()                   // number of outputs
    );

    // 4. Get the embedding result
    float* embedding_data = output_tensors.front().GetTensorMutableData<float>();

    // Optional: store embedding in a std::vector<float>
    size_t embedding_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> embedding(embedding_data, embedding_data + embedding_size);

    vec::print(embedding, "embedding", 8, true);
    std::cout << "embedding shape: " << embedding.size() << std::endl; 


      vec::draw_line(40, '-'); 
    } // for (const auto x: det)



    return 0;
}

