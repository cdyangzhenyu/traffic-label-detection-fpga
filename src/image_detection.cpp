#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <csignal>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "paddle_api.h"
#include "json.hpp"
#include "time.h"

using namespace std;
using namespace cv;
using namespace paddle::lite_api;
using json = nlohmann::json;

std::shared_ptr<paddle::lite_api::PaddlePredictor> g_predictor;
static float THRESHOLD = 0.3;

void init(json& j) {
    std::string model_dir = j["model"];
    std::vector<Place> valid_places({
        Place{TARGET(kFPGA), PRECISION(kFP16), DATALAYOUT(kNHWC)},
        Place{TARGET(kHost), PRECISION(kFloat)},
        Place{TARGET(kARM), PRECISION(kFloat)},
    });

    paddle::lite_api::CxxConfig config;
    bool combined = true;

    if (combined) {
        config.set_model_file(model_dir + "/model");
        config.set_param_file(model_dir + "/params");
    } else {
        config.set_model_dir(model_dir);
    }

    config.set_valid_places(valid_places);
    auto predictor = paddle::lite_api::CreatePaddlePredictor(config);
    g_predictor = predictor;

    THRESHOLD = j["threshold"];
}

Mat read_image(json& value, float* data) {

    auto image = value["image"];
    Mat img = imread(image);
    std::string format = value["format"];
    std::transform(format.begin(), format.end(),format.begin(), ::toupper);

    int width = value["input_width"];
    int height = value["input_height"];
    std::vector<float> mean = value["mean"];
    std::vector<float> scale = value["scale"];

    Mat img2;
    resize(img, img2, Size(width, height));
    
    Mat sample_float;
    img2.convertTo(sample_float, CV_32FC3);

    int index = 0;
    for (int row = 0; row < sample_float.rows; ++row) {
        float* ptr = (float*)sample_float.ptr(row);
        for (int col = 0; col < sample_float.cols; col++) {
            float* uc_pixel = ptr;
            float b = uc_pixel[0];
            float g = uc_pixel[1];
            float r = uc_pixel[2];

            if (format == "RGB") {
                data[index] = (r - mean[0]) * scale[0];
                data[index + 1] = (g - mean[1]) * scale[1];
                data[index + 2] = (b - mean[2]) * scale[2];
            } else {
                data[index] = (b - mean[0]) * scale[0];
                data[index + 1] = (g - mean[1]) * scale[1];
                data[index + 2] = (r - mean[2]) * scale[2];
            }
            ptr += 3;
            index += 3;
        }
    }
    return img;
}

void drawRect(const Mat &mat, float *data, int len, bool yolo) {
  for (int i = 0; i < len; i++) {
    float index = data[0];
    float score = data[1];
    if (score > THRESHOLD) {
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
        if (yolo) {
            x1 = static_cast<int>(data[2]);
            y1 = static_cast<int>(data[3]);
            x2 = static_cast<int>(data[4]);
            y2 = static_cast<int>(data[5]);
        } else {
            x1 = static_cast<int>(data[2] * mat.cols);
            y1 = static_cast<int>(data[3] * mat.rows);
            x2 = static_cast<int>(data[4] * mat.cols);
            y2 = static_cast<int>(data[5] * mat.rows);
        }
        int width = x2 - x1;
        int height = y2 - y1;

        cv::Point pt1(x1, y1);
        cv::Point pt2(x2, y2);
        cv::rectangle(mat, pt1, pt2, cv::Scalar(102, 0, 255), 3);
        std::cout << "label:" << index << ",score:" << score << " loc:";
        std::cout << x1 << "," << y1 << "," << width << "," << height
                << std::endl;
    }
    data += 6;
  }
  imwrite("result.jpg", mat);
}

void predict(json& value) {
    int width = value["input_width"];
    int height = value["input_height"];

    auto input = g_predictor->GetInput(0);
    input->Resize({1, 3, height, width});
    auto* in_data = input->mutable_data<float>();

    Mat img = read_image(value, in_data);
    bool is_yolo = false;
    auto network_type = value["network_type"];
    if (network_type != nullptr && network_type == "YOLOV3") {
        is_yolo = true;
        auto img_shape = g_predictor->GetInput(1);
        img_shape->Resize({1, 2});
        auto* img_shape_data = img_shape->mutable_data<int32_t>();
        img_shape_data[0] = img.rows;
        img_shape_data[1] = img.cols;
    }
    clock_t start, ends;
    start = clock();
    g_predictor->Run();
    ends = clock();
    std::cout << "Running Time: " << (double)(ends - start)/1000<<" ms" << std::endl;
    
    auto output = g_predictor->GetOutput(0);
    float *data = output->mutable_data<float>();
    int size = output->shape()[0];

    auto image = value["image"];
    drawRect(img, data, size, is_yolo);
}

int main(int argc, char* argv[]){
    std::string path;
    if (argc > 1) {
        path = argv[1];
    } else {
        path = "../configs/config.json";
    }
    
    json j;
    std::ifstream is(path);
    is >> j;
    init(j);
    predict(j);
    return 0;
}

