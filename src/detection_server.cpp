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
#include "httplib.h"

using namespace std;
using namespace httplib;
using namespace cv;
using namespace paddle::lite_api;
using json = nlohmann::json;

std::shared_ptr<paddle::lite_api::PaddlePredictor> g_predictor;
static float THRESHOLD = 0.3;
std::vector<string> labels;

void read_labels(json& j) {
    auto label_path = j["labels"];
    if (label_path == nullptr) {
        return;
    }
    std::ifstream file(label_path);
    if (file.is_open()) {
        std::string line;
        while (getline(file, line)) {
            labels.push_back(line);
        }
        file.close();
    }
}

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
    read_labels(j);
}

Mat read_image(json& value, float* data, cv::Mat &img) {

    //auto image = value["image"];
    //Mat img = imread(image);
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

std::string drawRect(const Mat &mat, float *data, int len, bool yolo) {
  std::string out_put = "";
  int num = 0;
  json out;
  json res_all;
  for (int i = 0; i < len; i++) {
    json res;
    float index = data[0];
    float score = data[1];
    if (score > THRESHOLD) {
        num += 1;
        int x1 = 0;
        int y1 = 0;
        int x2 = 0;
        int y2 = 0;
        if (yolo) {
            x1 = static_cast<int>(data[2]);
            y1 = static_cast<int>(data[3]);
            x2 = static_cast<int>(data[4]);
            y2 = static_cast<int>(data[5]);
        } 
	else {
            x1 = static_cast<int>(data[2] * mat.cols);
            y1 = static_cast<int>(data[3] * mat.rows);
            x2 = static_cast<int>(data[4] * mat.cols);
            y2 = static_cast<int>(data[5] * mat.rows);
        }
        int width = x2 - x1;
        int height = y2 - y1;

        //cv::Point pt1(x1, y1);
        //cv::Point pt2(x2, y2);
	//cv::Point pt3(x1-10, y1-10);
        //cv::rectangle(mat, pt1, pt2, cv::Scalar(102, 0, 255), 3);
	int class_id = static_cast<int>(index);
	std::string class_name = labels[class_id];
	//cv::putText(mat, class_name, pt3, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 255));
        std::cout << "label:" << index << ",score:" << score << " loc:";
        std::cout << x1 << "," << y1 << "," << width << "," << height
                << std::endl;
        
	res["score"] = score;
	res["class_name"] = class_name;
	res["loc"] = {x1, y1, x2, y2};
	res_all.push_back(res);
	//out_put += "{'score':"+std::to_string(score)+",'class_name': '" + class_name + "', 'loc': [" + std::to_string(x1) + ", " + std::to_string(y1) + ", " + std::to_string(x2)  + ", " + std::to_string(y2) + "]},";
    }
    
    data += 6;
    
  }
  out["len"] = num;
  out["result"] = res_all;
  out_put = out.dump();
  //out_put = "{'len':"+std::to_string(num)+",'result': ["+out_put+"]}"; 
  //imwrite("result.jpg", mat);
  return out_put;
}


std::string predict(json &value, cv::Mat &image) {
    int width = value["input_width"];
    int height = value["input_height"];
    clock_t start, start_run;
    start = clock();
    auto input = g_predictor->GetInput(0);
    input->Resize({1, 3, height, width});
    auto* in_data = input->mutable_data<float>();

    Mat img = read_image(value, in_data, image);
    std::cout << "Read image Time: " << (double)(clock() - start)/1000<<" ms" << std::endl;
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
    start_run = clock();   
    g_predictor->Run();
    std::cout << "Running Time: " << (double)(clock() - start_run)/1000<<" ms" << std::endl;
    
    auto output = g_predictor->GetOutput(0);
    float *data = output->mutable_data<float>();
    int size = output->shape()[0];
    std::string out_put = drawRect(img, data, size, is_yolo);
    return out_put;
}

std::string dump_headers(const Headers &headers) {
  std::string s;
  char buf[BUFSIZ];

  for (auto it = headers.begin(); it != headers.end(); ++it) {
    const auto &x = *it;
    snprintf(buf, sizeof(buf), "%s: %s\n", x.first.c_str(), x.second.c_str());
    s += buf;
  }

  return s;
}


std::string log(const Request &req, const Response &res) {
  std::string s;
  char buf[BUFSIZ];

  s += "================================\n";

  snprintf(buf, sizeof(buf), "%s %s %s", req.method.c_str(),
           req.version.c_str(), req.path.c_str());
  s += buf;

  std::string query;
  for (auto it = req.params.begin(); it != req.params.end(); ++it) {
    const auto &x = *it;
    snprintf(buf, sizeof(buf), "%c%s=%s",
             (it == req.params.begin()) ? '?' : '&', x.first.c_str(),
             x.second.c_str());
    query += buf;
  }
  snprintf(buf, sizeof(buf), "%s\n", query.c_str());
  s += buf;

  s += dump_headers(req.headers);

  s += "--------------------------------\n";

  snprintf(buf, sizeof(buf), "%d %s\n", res.status, res.version.c_str());
  s += buf;
  s += dump_headers(res.headers);
  s += "\n";

  if (!res.body.empty()) { s += res.body; }

  s += "\n";

  return s;
}

const char *html = R"(

<form id="formElem">
  <input type="file" name="image_file" onchange="uploadImg(this) " accept="image/*">
  <input type="submit">
</form>
<div style="float:top;border:1px dashed;background:#F0F8FF">
<pre>REST API Request: curl -F image_file=@test.jpg http://{ip}:8080/predict</pre>
<pre>Response:{"len":2,"result":[{"class_name":"red","loc":[677,174,866,569],"score":0.972812831401825},{"class_name":"red","loc":[25,163,215,582],"score":0.9500260949134827}]}
</div>
<img id="predictImg" width="768px" style="float:left"/>
<pre id="result" style="float:left;margin-left:150px;margin-top:100px"></pre>
<script src="https://cdn.staticfile.org/jquery/1.10.2/jquery.min.js"></script>
<script>
  formElem.onsubmit = async (e) => {
    e.preventDefault();
    let res = await fetch('/predict', {
      method: 'POST',
      body: new FormData(formElem)
    });
    var result = await res.text();
    result =  eval('(' + result + ')');
    $("#result").text(JSON.stringify(result, null, 4));
    console.log(result);
  };
  function uploadImg(obj) {
    var file = obj.files[0];
    var reader = new FileReader();
    reader.onload = function (e) {
      var img = document.getElementById("predictImg");
      img.src = e.target.result;
    }
    reader.readAsDataURL(file)
  }

</script>

)";

int main(int argc, char* argv[]){
    std::string path;
    path = "config.json";
    json j;
    std::ifstream is(path);
    is >> j;
    init(j);
    auto image = "test.jpg";
    Mat img = imread(image);
    predict(j, img);

    Server svr;
    svr.Get("/", [](const Request & /*req*/, Response &res) {
      res.set_content(html, "text/html");
    });
    svr.Post("/predict", [](const Request &req, Response &res) {
      clock_t start;
      start = clock();
      auto image_file = req.get_file_value("image_file");
      cv::Mat img_decode;
      std::vector<uchar> data(image_file.content.begin(), image_file.content.end());
      img_decode = cv::imdecode(data, CV_LOAD_IMAGE_COLOR);
      std::cout << "Read image to cv time: " << (double)(clock() - start)/1000<<" ms" << std::endl;
      //{
      //  ofstream ofs("debug.jpg", ios::binary);
      //  ofs << image_file.content;
      //}
      json k;
      std::ifstream is("config.json");
      is >> k;
      std::string out_put = predict(k, img_decode);
      res.set_content(out_put, "text/plain");
      std::cout << "Running All Time: " << (double)(clock() - start)/1000<<" ms" << std::endl;
    });
    svr.set_logger([](const Request &req, const Response &res) {
      printf("%s", log(req, res).c_str());
    });
    svr.listen("0.0.0.0", 8080);
    return 0;
}
