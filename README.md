## 物体检测server

### 说明

本项目基于Paddle-Lite FPGA模型检测框架封装的HTTP服务，可对外提供简单的web页面和REST API接口。

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
