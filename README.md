YOLOv11 异步目标检测系统
基于 C++ 和 OpenCV DNN 模块，使用 YOLOv11 模型的实时目标检测与跟踪系统。专为数字（1-5）和豆子（3种）的检测与位置跟踪而设计。
功能特点 (Features)
 * 多线程生产者-消费者架构：分离图像采集与推理过程，最大化帧率并消除延迟。
 * 异步推理：使用 OpenCV DNN 模块在 CPU 上进行 YOLOv11 推理。
 * 目标跟踪与偏移量计算：计算目标相对于屏幕中心（640x480）的 X/Y 像素偏移量。
 * 低通滤波：平滑坐标输出，减少画面及底盘抖动。
 * 目标丢失缓冲：在检测失败时维持数帧的输出，防止坐标突然跳变导致机械臂或底盘抽搐。
 * 串口通信：实现 11 字节强校验协议，向下位机（单片机）发送带符号的偏移量、类别 ID 和心跳包。
 * 自动标注：当检测置信度超过设定阈值（如 0.85）时，自动保存图像并生成 YOLO 格式的标签文件，方便后续收集数据集。
环境要求 (Requirements)
 * OpenCV 4.x（需包含 DNN 模块）
 * C++17 编译器
 * CMake 3.16+
 * Windows 系统：串口库（包含在 Windows SDK 中）
 * Linux 系统：termios 库
编译说明 (Build Instructions)
 * 克隆或解压源代码。
 * 安装 OpenCV，并确保 CMake 能够找到它。
 * 创建 build 目录并运行 CMake：
mkdir build
cd build
cmake ..
cmake --build . --config Release

 * 编译生成的可执行文件将存放在 bin/ 目录中。
模型准备 (Model Preparation)
 * 将你的 YOLOv11 模型导出为 ONNX 格式（假设输入尺寸为 640x640）。
 * 将 ONNX 模型文件放置在 models/yolov11.onnx。
 * 创建类别名称文件 models/classes.txt，每行一个类别（共 8 个类别：数字 1-5 和三种豆子）。示例：
1
2
3
4
5
bean_type1
bean_type2
bean_type3

配置 (Configuration)
编辑 src/Config.h 来调整系统参数：
 * MODEL_PATH：ONNX 模型路径。
 * CLASSES_FILE：类别名称文件路径。
 * NETWORK_WIDTH, NETWORK_HEIGHT：模型输入尺寸（应与你的模型结构一致）。
 * INPUT_WIDTH, INPUT_HEIGHT：摄像头画面尺寸（默认 640x480）。
 * CONFIDENCE_THRESHOLD, NMS_THRESHOLD：检测置信度与非极大值抑制（NMS）阈值。
 * AUTO_LABEL_THRESH：触发自动标注的置信度阈值。
 * SERIAL_PORT：串口名称（Windows 下为 COM3，Linux 下为 /dev/ttyUSB0）。
 * SERIAL_BAUD：串口波特率。
 * FILTER_ALPHA：低通滤波系数（值越小越平滑，但响应延迟越大）。
 * LOST_BUFFER_FRAMES：丢失目标后维持坐标输出的帧数。
使用方法 (Usage)
运行可执行文件：
./bin/yolov11_async

 * 按 ESC 键退出程序。
 * 系统默认调用索引为 0 的摄像头进行画面采集。如需更改摄像头索引，请修改 main.cpp。
 * 检测到的目标会在屏幕上绘制出边界框和中心点。
 * X/Y 偏移量会实时显示在屏幕上，并通过串口发送。
 * 高置信度检测会触发自动标注；图像和标签会被保存在 dataset/images/ 和 dataset/labels/ 目录中。
串口通信协议 (Serial Protocol)
系统会发送 11 字节的数据包，数据帧格式如下：
| 字节 (Byte) | 内容 (Content) | 描述 (Description) |
|---|---|---|
| 0 | 0xAA | 起始帧头 |
| 1-2 | int16_t (LE) | X 轴偏移量（有符号，像素值，小端序） |
| 3-4 | int16_t (LE) | Y 轴偏移量（有符号，像素值，小端序） |
| 5 | uint8_t | 类别 ID (0-255) |
| 6 | uint8_t | 心跳计数器（每帧自增，用于检测卡死） |
| 7-9 | 0x00 | 保留位（全零填充） |
| 10 | uint8_t | 校验和（第 0-9 字节累加和对 256 取模） |
自动标注 (Auto-Labeling)
当检测置信度超过 AUTO_LABEL_THRESH 时，系统会保存当前帧及其检测结果。图像保存为 JPEG 格式，标签保存为 YOLO 格式的文本文件（归一化坐标）。在程序运行期间，你的数据集会不断自动增量增长。
注意事项 (Notes)
 * 请确保摄像头输出分辨率设定为 640x480；程序会在送入网络前自动完成 Resize 操作。
 * 如果串口无法打开，程序将打印警告并继续运行，但不会发送硬件数据（软调模式）。
 * 跟踪算法默认选择最靠近屏幕中心的检测目标作为主跟踪对象（过滤背景干扰）。
 * 低通滤波器采用一阶指数平滑算法：filtered = alpha * new + (1-alpha) * filtered。
故障排除 (Troubleshooting)
 * 模型加载失败：检查 ONNX 文件路径是否正确，以及模型算子是否兼容 OpenCV DNN 模块。
 * 没有检测结果：尝试调低置信度阈值，验证模型输出维度是否符合预期格式。
 * CPU 占用率过高：减小帧队列大小（Queue Size），或考虑降低摄像头的物理采样帧率。
 * 串口不工作：验证串口名称和波特率是否匹配；如果运行在 Linux 环境下，请检查串口读写权限（sudo chmod 666 /dev/ttyUSB0）。
许可证 (License)
本项目按“原样”提供，仅供教育、比赛及研究目的使用。