#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <unistd.h>
#include <mutex>
#include <csignal>
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "camera.hpp"
#include "blocking_queue.h"
#include "paddle_api.h"
#include "time.h"

#include "json.hpp"

using namespace std;
using namespace cv;
using namespace paddle::lite_api;
using json = nlohmann::json;

const string name = "detection";

static Camera g_cap;
static BlockingQueue<Mat> g_image_queue;
static BlockingQueue<Mat> g_dis_queue;
static cv::Mat g_display_frame;
static std::mutex g_mtx;

static int width = 300;
static int height = 300;
static float THRESHOLD = 0.5f;

std::vector<float> mean_data;
std::vector<float> scale;
static std::string image_format = "BGR";
bool is_yolo = false;

//SSD解析后的结果
struct SSDResult{
    int type;
    float score;
    int x;
    int y;
    int width;
    int height;
};

std::shared_ptr<paddle::lite_api::PaddlePredictor> g_predictor;
std::vector<SSDResult> g_results;

void init(json& value) {
    std::string model_dir = value["model"];
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

    width = value["input_width"];
    height = value["input_width"];
    std::vector<float> mean = value["mean"];
    for (int i = 0; i < mean.size(); ++i) {
        mean_data.push_back(mean[i]);
    }

    std::vector<float> json_scale = value["scale"];
    for (int i = 0; i < json_scale.size(); ++i) {
        scale.push_back(json_scale[i]);
    }

    image_format = value["format"];
    THRESHOLD = value["threshold"];

    auto network_type = value["network_type"];
    if (network_type != nullptr && network_type == "YOLOV3") {
        is_yolo = true;
    }
}

int predict() {
    while (true) {
        cv::Mat mat = g_image_queue.Take();

        auto input = g_predictor->GetInput(0);
        input->Resize({1, 3, height, width});
        auto* data = input->mutable_data<float>();

        // std::cout<<"predicting..."<<std::endl;
        // g_mtx.lock();
        // 预处理。
        cv::Mat preprocessMat;
        mat.convertTo(preprocessMat, CV_32FC3);
        int index = 0;
        for (int row = 0; row < preprocessMat.rows; ++row) {
            float* ptr = (float*)preprocessMat.ptr(row);
            for (int col = 0; col < preprocessMat.cols; col++) {
                float* uc_pixel = ptr;
                float b = uc_pixel[0];
                float g = uc_pixel[1];
                float r = uc_pixel[2];
                // 减均值
                if (image_format == "RGB") {
                    data[index] = (r - mean_data[0]) * scale[0];
                    data[index + 1] = (g - mean_data[1]) * scale[1];
                    data[index + 2] = (b - mean_data[2]) * scale[2];
                } else {
                    data[index] = (b - mean_data[0]) * scale[0];
                    data[index + 1] = (g - mean_data[1]) * scale[1];
                    data[index + 2] = (r - mean_data[2]) * scale[2];
                }
                ptr += 3;
                index += 3;
            }
        }

        if (is_yolo) {
            auto img_shape = g_predictor->GetInput(1);
            img_shape->Resize({1, 2});
            auto* img_shape_data = img_shape->mutable_data<int32_t>();
            img_shape_data[0] = g_display_frame.rows;
            img_shape_data[1] = g_display_frame.cols;
        }

	clock_t start, end;
	start = clock();
        g_predictor->Run();
        end = clock();
	std::cout << "Predict time: " << (double)(end - start)/1000 << " ms" << std::endl;

        auto output = g_predictor->GetOutput(0);
        float *result_data = output->mutable_data<float>();
        int size = output->shape()[0];

        g_results.clear();
        int display_width = g_display_frame.cols;
        int display_height = g_display_frame.rows;

        for (int i = 0; i < size; i++) {
            float* data = result_data + i * 6;
            float score = data[1];
            if (score < THRESHOLD) {
                continue;
            }
            SSDResult r;
            r.type = (int)data[0];
            r.score = score;
            if (is_yolo) {
                r.x = data[2];
                r.y = data[3];
                r.width = data[4] - r.x;
                r.height = data[5] - r.y;
            } else {
                r.x = data[2] * display_width;
                r.y = data[3] * display_height;
                r.width = data[4] * display_width - r.x;
                r.height = data[5] * display_height - r.y;
            }
            
            g_results.push_back(r);
        }   
        // g_mtx.unlock();
    }
    return 0;
}

//相机回调，image为最新的一帧图片
int image_callback(cv::Mat& image) {
    cv::Mat mat;
    // g_mtx.lock();
    g_display_frame = image;
    // g_mtx.unlock();
    if (g_dis_queue.Size() < 1) {
        g_dis_queue.Put(image);
    }
    if (g_image_queue.Size() < 1) {
        cv::resize(image, mat, Size(width, height), INTER_AREA);
        g_image_queue.Put(mat);
    }
    return 0;    
}

void capture(std::string video_dev) {
    CameraConfig config;
    config.dev_name = video_dev;
    config.width = 1280;
    config.height = 720;
    g_cap.setConfig(config);
    g_cap.start(image_callback);
    g_cap.loop();
}

int cv_getpic_loop(std::string video_device) {
    cout << "cv getpic start" <<endl;
    cv::VideoCapture capture(video_device);
    if (!capture.isOpened()) {
        std::cout<<"can not open video device."<<std::endl;
        return 1;
    }

    capture.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    if (!capture.set(cv::CAP_PROP_FPS, 15)) {
        std::cout << "camera set fps failed" << std::endl;
    }
    //获取当前视频帧率
    double rate = capture.get(CV_CAP_PROP_FPS);
    std::cout<<"rate = "<<rate<<std::endl;

    cv::Mat frame;
    //每一帧之间的延时
    //与视频的帧率相对应
    int delay = 1000 / rate;
    while (1) {
        if(!capture.read(frame)) {
            std::cout<<"no video frame"<<std::endl;
            continue;
        }

        //此处为添加对视频的每一帧的操作方法
        image_callback(frame);
    }

    //关闭视频，手动调用析构函数（非必须）
    capture.release();
    return 0;
}
void signal_handler( int signum ) {
    g_cap.release();
    exit(signum);  
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
    signal(SIGINT, signal_handler); 

    // 为相机开启一个新的线程
    // std::thread task_capture(capture);
    // 预测线程
    std::thread task_predict(predict);
    std::thread task_cvcap(cv_getpic_loop, j["video_device"]);
    
    cv::namedWindow(name, CV_WINDOW_NORMAL);
    cv::setWindowProperty(name, CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    // moveWindow(name, 20, 20);
    usleep(2000); // 等待窗口初始化
    while (true) {
        cv::Mat mat = g_dis_queue.Take();
        /* if (g_display_frame.cols == 0) {
            // std::cout<< "FIXME: shouldn't be 0 of image's cols" << std::endl;
            usleep(40000);
            continue;
        } */
        // g_mtx.lock();
        for (int i = 0; i < g_results.size(); ++i) {
            SSDResult r = g_results[i];
            cv::Rect rect(r.x, r.y,r.width,r.height);
            // cv::rectangle(g_display_frame, rect, Scalar(0,0,224), 2);
            cv::rectangle(mat, rect, Scalar(0,0,224), 2);
        }
        imshow(name, mat);
        // imshow(name, g_display_frame);
        // g_mtx.unlock();
        cv::waitKey(10);
    }
    // task_capture.join();
    task_cvcap.join();
    task_predict.join();
    return 0;
}

