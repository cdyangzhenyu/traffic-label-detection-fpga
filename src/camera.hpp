#ifndef CAMERA_HPP
#define CAMERA_HPP

#include <mutex>
#include <opencv2/opencv.hpp>

typedef int (*callback)(cv::Mat&);

struct CameraConfig {
    std::string dev_name;
    int width;
    int height;
};

class Camera {
public:
    Camera();
    void setConfig(CameraConfig config) {
        this->config = config;
        this->dev_name = config.dev_name;
    }
    void loop(void);
    void start(callback call);
    void release(void);
    ~Camera();
private:
    callback call;
    std::mutex _mtx;
    CameraConfig config;
    int width;
    int height;

    int fd;
    std::string dev_name;
    void errno_exit(const char *s);
    int xioctl(int fh, int request, void *arg);
    void process_image(void *p, int size);
    int read_frame(void);
    void stop_capturing(void);
    void start_capturing(void);
    void uninit_device(void);
    void init_mmap(void);
    void init_device(void);
    void close_device(void);
    void open_device(void);
};

#endif /* CAMERA_HPP */