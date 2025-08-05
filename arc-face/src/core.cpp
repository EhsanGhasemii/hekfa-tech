#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Ort;

const int INPUT_WIDTH = 640;
const int INPUT_HEIGHT = 640;

// Preprocess image: Resize, BGR → RGB, Normalize, CHW
cv::Mat preprocess(const cv::Mat& img) {
    cv::Mat resized, rgb, float_img;
    cv::resize(img, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 127.5, -1.0);  // Normalize to [-1, 1]

    // Convert HWC to CHW
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);
    cv::Mat chw;
    cv::vconcat(channels, chw);  // C x H x W

    return chw;
}

int main() {

    // Load image
    std::string image_path = "persons.jpg";
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "❌ Failed to load image: " << image_path << std::endl;
        return -1;
    }

    // Preprocess
    cv::Mat input_tensor = preprocess(img);

    // Load model
    std::string model_path = "buffalo_l/det_10g.onnx";
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "detector");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // NOTE: Do NOT add CUDA or GPU execution provider to force CPU execution
    // session_options.AppendExecutionProvider_CUDA(0);  // <-- remove/comment out this line if exists

    Ort::Session session(env, model_path.c_str(), session_options);

    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name0_ptr = session.GetOutputNameAllocated(0, allocator);
    auto output_name1_ptr = session.GetOutputNameAllocated(1, allocator);
    auto output_name2_ptr = session.GetOutputNameAllocated(2, allocator);

    const char* input_name = input_name_ptr.get();
    const char* output_name0 = output_name0_ptr.get();
    const char* output_name1 = output_name1_ptr.get();
    const char* output_name2 = output_name2_ptr.get();

    // Build input tensor
    std::vector<int64_t> input_shape = {1, 3, INPUT_HEIGHT, INPUT_WIDTH};
    size_t tensor_size = 1 * 3 * INPUT_HEIGHT * INPUT_WIDTH;
    std::vector<float> input_data(tensor_size);
    std::memcpy(input_data.data(), input_tensor.data, tensor_size * sizeof(float));

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor_onnx = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), tensor_size, input_shape.data(), input_shape.size());

    // Run inference
    std::array<const char*, 1> input_names = {input_name};
    std::array<const char*, 3> output_names = {output_name0, output_name1, output_name2};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr},
                                      input_names.data(), &input_tensor_onnx, 1,
                                      output_names.data(), 3);

    // Get raw model outputs
    float* boxes = output_tensors[0].GetTensorMutableData<float>();
    float* landmarks = output_tensors[1].GetTensorMutableData<float>();  // not used here, but available
    float* scores = output_tensors[2].GetTensorMutableData<float>();

    auto boxes_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();  // [1, N, 4]
    size_t num_faces = boxes_shape[1];

    cout << "🔍 Total detected faces: " << num_faces << endl;

    int face_count = 0;

    for (size_t i = 0; i < num_faces; ++i) {
        float score = scores[i];
        if (score < 0.5f) continue;  // confidence threshold

        float x1 = boxes[i * 4 + 0] * img.cols / INPUT_WIDTH;
        float y1 = boxes[i * 4 + 1] * img.rows / INPUT_HEIGHT;
        float x2 = boxes[i * 4 + 2] * img.cols / INPUT_WIDTH;
        float y2 = boxes[i * 4 + 3] * img.rows / INPUT_HEIGHT;

        cout << "Face " << face_count << " → score: " << score
             << " | box: (" << x1 << ", " << y1 << ") - (" << x2 << ", " << y2 << ")\n";

        cv::rectangle(img, cv::Point(int(x1), int(y1)), cv::Point(int(x2), int(y2)),
                      cv::Scalar(0, 255, 0), 2);

        face_count++;
    }

    // Save output image
    cv::imwrite("output.jpg", img);
    cout << "✅ Output image saved as output.jpg" << endl;

    // Save detected face count to file
    std::ofstream ofs("faces.txt");
    ofs << face_count << std::endl;
    ofs.close();
    cout << "✅ Face count saved in faces.txt" << endl;

    return 0;
}

