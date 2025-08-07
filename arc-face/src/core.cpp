#include <iostream>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <tuple>
#include <map>


// Converts distances to bounding boxes
cv::Mat distance2bbox(const cv::Mat& points, const cv::Mat& distance, const cv::Size& max_shape = cv::Size()) {
    CV_Assert(points.cols == 2 && distance.cols == 4);

    cv::Mat x1 = points.col(0) - distance.col(0);
    cv::Mat y1 = points.col(1) - distance.col(1);
    cv::Mat x2 = points.col(0) + distance.col(2);
    cv::Mat y2 = points.col(1) + distance.col(3);

    if (max_shape.width > 0 && max_shape.height > 0) {
        cv::min(cv::max(x1, 0), max_shape.width, x1);
        cv::min(cv::max(y1, 0), max_shape.height, y1);
        cv::min(cv::max(x2, 0), max_shape.width, x2);
        cv::min(cv::max(y2, 0), max_shape.height, y2);
    }

    std::vector<cv::Mat> bboxes = {x1, y1, x2, y2};
    cv::Mat result;
    cv::merge(bboxes, result);
    return result;
}

// Converts distances to keypoints
cv::Mat distance2kps(const cv::Mat& points, const cv::Mat& distance, const cv::Size& max_shape = cv::Size()) {
    CV_Assert(points.rows == distance.rows);
    CV_Assert(distance.cols % 2 == 0);

    std::vector<cv::Mat> preds;
    for (int i = 0; i < distance.cols; i += 2) {
        cv::Mat px = points.col(i % 2) + distance.col(i);
        cv::Mat py = points.col((i % 2 + 1) % 2) + distance.col(i + 1);

        if (max_shape.width > 0 && max_shape.height > 0) {
            cv::min(cv::max(px, 0), max_shape.width, px);
            cv::min(cv::max(py, 0), max_shape.height, py);
        }

        preds.push_back(px);
        preds.push_back(py);
    }

    cv::Mat result;
    cv::merge(preds, result);
    return result;
}

// Non-maximum suppression
std::vector<int> nms(const cv::Mat& dets, float thresh) {
    std::vector<float> x1, y1, x2, y2, scores;
    for (int i = 0; i < dets.rows; ++i) {
        x1.push_back(dets.at<float>(i, 0));
        y1.push_back(dets.at<float>(i, 1));
        x2.push_back(dets.at<float>(i, 2));
        y2.push_back(dets.at<float>(i, 3));
        scores.push_back(dets.at<float>(i, 4));
    }

    std::vector<float> areas;
    for (size_t i = 0; i < x1.size(); ++i) {
        areas.push_back((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1));
    }

    std::vector<int> order(x1.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) { return scores[i] > scores[j]; });

    std::vector<int> keep;
    while (!order.empty()) {
        int i = order[0];
        keep.push_back(i);

        std::vector<int> rem(order.begin() + 1, order.end());
        std::vector<int> new_order;

        for (int j : rem) {
            float xx1 = std::max(x1[i], x1[j]);
            float yy1 = std::max(y1[i], y1[j]);
            float xx2 = std::min(x2[i], x2[j]);
            float yy2 = std::min(y2[i], y2[j]);

            float w = std::max(0.0f, xx2 - xx1 + 1);
            float h = std::max(0.0f, yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (areas[i] + areas[j] - inter);

            if (ovr <= thresh) {
                new_order.push_back(j);
            }
        }

        order = new_order;
    }

    return keep;
}







int main() {
    const std::string detection_model_path = "/root/.insightface/models/buffalo_l/det_10g.onnx";

    // Assumptions from earlier context
    float input_mean = 127.5f;
    float input_std = 128.0f;

    int fmc = 3;
    std::vector<int> _feat_stride_fpn = {8, 16, 32};
    int _num_anchors = 2;
    bool use_kps = true;

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

    // Optional: Save or display det_img
    // cv::imwrite("det_img.jpg", det_img);
    // cv::imshow("Det Image", det_img);
    // cv::waitKey(0);


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

//         std::cout << "Output " << i << " values:" << std::endl;
//         for (size_t j = 0; j < total_len; ++j) {
//             std::cout << float_array[j] << " ";
//             if ((j + 1) % 10 == 0) std::cout << std::endl;
//         }
//         std::cout << std::endl;
    }


    // 251-253: Placeholder lists
    std::vector<cv::Mat> scores_list;
    std::vector<cv::Mat> bboxes_list;
    std::vector<cv::Mat> kpss_list;

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
//     for (int i = 0; i < total_len; ++i) {
//         std::cout << data[i] << " ";
//     }
    std::cout << "total len: " << total_len << std::endl; 
    std::cout << "rows: " << rows << std::endl; 
    std::cout << "cols: " << cols << std::endl; 
    std::cout << std::endl;


    for (uint32_t i = 0; i < _feat_stride_fpn.size(); ++i) { 
      std::cout << "kk: " << _feat_stride_fpn[i] << std::endl; 
    } 





    return 0;
}

