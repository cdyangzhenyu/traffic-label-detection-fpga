## 物体检测示例使用说明

### 说明

和分类不同，物体检测除了能知道物体的类型，还能检测出物体所在的位置坐标。物体检测也分了两个示例，一个是在图片上检测物体，并绘制出坐标信息。还有通过摄像头采集视频，检测在屏幕上绘制坐标信息。

工程目录结构:

```
├── CMakeLists.txt // cmake 工程配置文件。
├── include 
├── configs // 配置文件目录
│   ├── yolov3
│   └── ssd
├── lib
├── models // 模型文件目录
│   ├── yolov3
│   └── ssd
│── src
│   ├── json.hpp // json 解析库
│   └── image_detect.cpp // 图片推理示例
└── README.md
```

下面是配置文件示例。

```JSON
{
	"model":"../models/yolov3",
	"combined_model":true,
    "video_device":"/dev/video0",
	"input_width":608,
	"input_height":608,
	"image":"../models/yolov3/1.jpg",
	"mean":[123.675, 116.28, 103.53],
    "scale":[0.0171248, 0.017507, 0.0174292],
	"format":"RGB",
	"threshold":0.3,
	"network_type":"YOLOV3"
}

```

| key            | value                                      |
| -------------- | ------------------------------------------ |
| model          | 模型目录存放的位置                         |
| combined_model | 是否为融合的模型，只有两个文件的是融合模型 |
| video_device   | usb摄像头识别到的设备节点                  |
| input_width    | 输入网络的图片尺寸 输入图片会缩放到该大小  |
| input_height   | 输入网络的图片尺寸                         |
| image          | 进行检测的图片输入                         |
| mean           | 平均值                                     |
| scale          | 输入网络前预算处理为 ( x - mean ) * scale  |
| format         | 网络所需要的格式，OpenCV默认是BGR          |
| threshold      | 信心值阈值                                 |
| network_type   |                                            |

其它的目标检测网络也可以通过添加/修改 配置文件实现，无须修改代码。

### 使用步骤

1.加载驱动,系统启动后加载一次即可（已启动的可忽略本）

```
insmod /home/root/workspace/driver/fpgadrv.ko
```

2.编译示例，EdgeBoard边缘计算盒上具有编译能力，进入到PaddleLiteSample/detection示例的build目录下进行编译

```
cd /home/root/workspace/PaddleLiteSample/detection
// 如果没有build目录，创建一个
mkdir build
cd build
rm -rf *
// 调用cmake 创建 Makefile 
cmake ..

// 编译工程。
make
	
```

编译结束后会在build 目录生成如下几个文件。
image\_detection 读取本地图片推理示例。
video\_detection 读取摄像头数据进行推理，要连接USB Camera才能使用。如需显示结果还要连接DP显示器或者HDMI显示器或者VGA显示器；

3.执行示例<br>

```
读取一张图片，做推理，将结果保存为图片输出result.jpg
./image_detection ../configs/yolov3/screw.json 

读取摄像头数据，做推理，将结果通过DP接口显示到桌面显示屏上
./video_detection ../configs/yolov3/screw.json     
```