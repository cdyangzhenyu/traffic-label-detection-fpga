## 物体检测server

### 说明

本项目基于Paddle-Lite FPGA模型检测框架封装的HTTP服务，可对外提供简单的web页面和REST API接口。

和分类不同，物体检测除了能知道物体的类型，还能检测出物体所在的位置坐标。物体检测也分了两个示例，一个是在图片上检测物体，并绘制出坐标信息。还有通过摄像头采集视频，检测在屏幕上绘制坐标信息。

### 编译

```
cd build
cmake ..
make
```

### 服务启动

参考/bin目录

### 模型替换

根据要求替换以下文件即可
```
label_list.txt
model
params
test.jpg
config.json
```

### 要求

模型需要使用paddle-detection的yolov3训练
