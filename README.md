# 视觉检测系统

**版本：2.0** | **发布日期：2026-04-04**

## 项目简介

本系统是一个基于 C++、OpenCV 和 YOLOv11 (ONNX Runtime) 的双路相机视觉检测程序，用于工业场景下的实时目标识别与定位。系统采用模块化架构，支持数字相机与豆子相机双路并行处理，具备强容错能力和稳定的实时性能。

## 系统架构与技术要点

### 模块化设计

系统代码按功能划分为以下模块，职责清晰，便于维护与扩展：

- **`src/core/`**：逻辑与配置
  - `main.cpp`：程序入口与主循环
  - `Config.h`：全局配置
  - `FrameQueue.h/.cpp`：线程安全帧队列
  - `CoordinateCalculator.h/.cpp`：坐标计算与防抖

- **`src/vision/`**：视觉算法模块
  - `CameraManager.h/.cpp`：双相机并行
  - `Detector.h/.cpp`：YOLO推理
  - `Tracker.h/.cpp`：目标跟踪
  - `ContourProcessor.h/.cpp`：轮廓处理与纸张区域提取
  - `AutoLabeler.h/.cpp`：自动标注工具
- **`src/io/`**：输入输出与通信
  - `SerialPort.h/.cpp`：串口通信封装
  - `DataTransmitter.h/.cpp`：数据打包与发送
### 多线程与并发控制

系统特点：

1. **独立线程**：每个相机拥有独立的处理线程，通过 `CameraManager` 统一调度。
2. **最新帧覆盖**：采用 `std::mutex` 保护帧缓存，读取时直接覆盖旧帧，彻底解决帧积压导致的延迟问题。
3. **异步推理**：YOLO 检测任务通过 `std::async` 异步执行，避免阻塞相机采集线程。
4. **错峰启动**：豆子相机线程延迟 2 秒启动，避免双相机同时初始化时硬件资源抢占。

### 底层设备控制

针对 USB 工业相机的特性，系统实现了精细的底层控制：

1. **MSMF 后端**：使用 OpenCV 的 `cv::CAP_MSMF` 后端（Windows Media Foundation），提供稳定的高速采集。
2. **防拖影曝光设置**：
   - 关闭自动曝光：`cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25)`
   - 固定短曝光时间：`cap.set(cv::CAP_PROP_EXPOSURE, exposure)`（曝光值为负，数值越小快门越快）
   - 该设置根除运动模糊，但要求充足的环境光照或外部补光
3. **缓冲区优化**：`cap.set(cv::CAP_PROP_BUFFERSIZE, 1)` 限制底层缓冲区大小，减少延迟。
4. **暖机机制**：相机打开后连续读取 30 帧等待画面稳定，确保初始化可靠性。

## 编译与运行环境

### 系统要求

- **操作系统**：Windows 10/11，Linux（需调整相机索引与串口路径）
- **编译器**：支持 C++17 的编译器（MSVC 2019+ / GCC 9+）
- **CMake**：3.16+

### 依赖库

1. **OpenCV 4.x**：图像采集、处理与显示
2. **ONNX Runtime 1.14+**：YOLOv11 模型推理
3. **Windows SDK**（仅 Windows）：串口通信

### 编译步骤

1. 配置环境变量：
   - 设置 `OpenCV_DIR` 指向 OpenCV 安装目录的 `build` 文件夹
   - 添加 ONNX Runtime 头文件与库路径

2. 使用项目提供的构建脚本：
   ```bash
   # Windows
   .\build.bat

   # Linux/Mac（需自行编写 build.sh）
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . --config Release
   ```

3. 输出文件位于 `build/bin/` 目录，可执行文件为 `yolov11_async.exe`（Windows）或 `yolov11_async`（Linux）。

### 模型文件准备

将训练好的 YOLOv11 ONNX 模型放置在 `models/` 目录下，并在 `Config.h` 中配置正确的 `MODEL_PATH` 路径。类别标签文件（可选）通过 `CLASSES_FILE` 指定。

## 参数配置指南

所有可调参数均在 `src/core/Config.h` 中定义，修改后需重新编译生效。

### 硬件参数（相机配置）

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `DIGIT_CAMERA_ID` | `int` | 0 | 数字相机设备索引（Windows 通常为 0,1,2...） |
| `BEAN_CAMERA_ID` | `int` | 1 | 豆子相机设备索引（避免与数字相机共用 USB 控制器） |
| `DIGIT_EXPOSURE` | `double` | -3.0 | 数字相机曝光值（负值，数值越小快门越快） |
| `BEAN_EXPOSURE` | `double` | -4.0 | 豆子相机曝光值（建议比数字相机稍亮） |
| `DIGIT_SKIP_FRAMES` | `int` | 0 | 数字相机跳帧数（0=实时，1=每2帧处理1帧） |
| `BEAN_SKIP_FRAMES` | `int` | 1 | 豆子相机跳帧数（降低 CPU 负载） |
| `DIGIT_SOURCE_ID` | `int` | 0x01 | 数字相机数据源 ID（串口协议标识） |
| `BEAN_SOURCE_ID` | `int` | 0x02 | 豆子相机数据源 ID（串口协议标识） |

**曝光参数说明**：
- 曝光值范围为负值（如 -10.0 ~ 0.0），数值越大进光量越大，但运动拖影越明显。
- 数值越小（如 -6.0）表示快门时间极短，可完全消除运动模糊，但需要更强的环境光照。
- **重要**：在低曝光设置下，必须配合外部物理补光（如 LED 环形灯），否则图像将过暗无法识别。

### 算法参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `MODEL_PATH` | `string` | （示例路径） | YOLO ONNX 模型文件路径 |
| `CLASSES_FILE` | `string` | （示例路径） | 类别标签文件路径（每行一个类别名称） |
| `NETWORK_WIDTH` | `int` | 640 | 网络输入宽度（必须为 32 的倍数） |
| `NETWORK_HEIGHT` | `int` | 640 | 网络输入高度（必须为 32 的倍数） |
| `CONFIDENCE_THRESHOLD` | `float` | 0.5 | 检测置信度阈值（0.0~1.0） |
| `NMS_THRESHOLD` | `float` | 0.4 | 非极大值抑制阈值（0.0~1.0） |
| `NUM_CLASSES` | `int` | 5 | 模型类别总数 |

### 传统视觉参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `THRESHOLD` | `int` | 200 | 二值化阈值（0-255，用于提取白色纸张区域） |
| `MIN_AREA` | `int` | 1500 | 最小轮廓面积（像素数，过滤噪声） |
| `LOST_BUFFER_FRAMES` | `int` | 5 | 目标丢失缓冲帧数（防抖动） |
| `FILTER_ALPHA` | `float` | 0.3 | 低通滤波器系数（0.0~1.0，用于坐标平滑） |

### 系统参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `INPUT_WIDTH` | `int` | 640 | 输入图像宽度（相机采集分辨率） |
| `INPUT_HEIGHT` | `int` | 480 | 输入图像高度（建议保持 4:3 比例） |
| `SCREEN_CENTER_X` | `int` | 320 | 屏幕中心 X 坐标（自动计算） |
| `SCREEN_CENTER_Y` | `int` | 240 | 屏幕中心 Y 坐标（自动计算） |
| `SERIAL_PORT` | `string` | "COM3" | 串口端口号（Windows: COM3, Linux: /dev/ttyUSB0） |
| `SERIAL_BAUD` | `int` | 115200 | 串口波特率 |

## 目录结构

```
.
├── CMakeLists.txt                # CMake 构建配置
├── build.bat                     # Windows 构建脚本
├── README.md                     # 本文档
├── src/                          # 源代码目录
│   ├── core/                     # 核心逻辑
│   │   ├── main.cpp              # 程序入口
│   │   ├── Config.h              # 全局配置中心
│   │   ├── FrameQueue.h/.cpp     # 线程安全帧队列
│   │   └── CoordinateCalculator.h/.cpp  # 坐标计算器
│   ├── vision/                   # 视觉算法模块
│   │   ├── CameraManager.h/.cpp  # 双相机
│   │   ├── Detector.h/.cpp       # YOLO 推理
│   │   ├── Tracker.h/.cpp        # 目标跟踪
│   │   ├── ContourProcessor.h/.cpp  # 轮廓处理
│   │   ├── AutoLabeler.h/.cpp    # 自动标注工具
│   └── io/                       # 输入输出模块
│       ├── SerialPort.h/.cpp     # 串口通信
│       └── DataTransmitter.h/.cpp # 数据发射器
├── models/                       # 模型文件目录
│   └── classes.txt               # 类别标签文件
└── dataset/                      # 数据集目录
```

## 运行说明

1. 连接两台USB相机，确认设备索引
2. 根据实际硬件调整 `Config.h` 中的相机索引、曝光值、串口参数。
3. 编译项目并运行可执行文件。
4. 程序启动后将显示两个独立窗口：
   - **Digit Camera**：数字相机实时画面与识别结果
   - **Bean Camera**：豆子相机实时画面与识别结果
5. 键盘控制：
   - `ESC`：退出程序
   - `1`：启用/禁用数字相机
   - `2`：启用/禁用豆子相机
   - `d`：显示/隐藏调试信息
   - `t`：截取当前帧

## 注意事项

1. **光照要求**：短曝光设置需要充足的环境光照，建议使用高亮度 LED 环形灯补光。
2. **USB 带宽**：双相机应连接至不同的 USB 控制器（通常为不同颜色的 USB 接口），避免带宽抢占。
3. **模型兼容性**：YOLOv11 ONNX 模型必须使用与训练时相同的输入尺寸（默认 640x640）。
4. **串口通信**：确保下位机波特率与 `SERIAL_BAUD` 一致，数据包格式符合协议要求。
5. **故障容错**：单个相机初始化失败不会导致程序崩溃，系统将继续运行可用分支。
