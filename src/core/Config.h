#ifndef CONFIG_H
#define CONFIG_H

#include <string>
#include <vector>

namespace Config {
    
    //  数字相机设备索引（通常为0，代表第一个USB相机）
    //根据实际硬件连接顺序调整，Windows系统一般为0,1,2...
    inline constexpr int DIGIT_CAMERA_ID = 0;

    //  豆子相机设备索引（通常为1，代表第二个USB相机）
    //确保与数字相机不在同一个USB
    inline constexpr int BEAN_CAMERA_ID = 1;

    //  数字相机曝光值（负值表示快门时间更短，数值越大进光量越大）
    //-10.0 ~ 0.0，室内光线建议-4.0 ~ -2.0，室外强光可设-6.0以下
    inline constexpr double DIGIT_EXPOSURE = 1.0;

    //  豆子相机曝光值
    //豆子识别需要更均匀光照，建议比数字相机稍亮
    inline constexpr double BEAN_EXPOSURE = -4.0;

    //  数字相机跳帧数（0表示不跳帧，实时处理）
    //数字识别要求高实时性，通常设为0；如CPU负载过高可设为1
    inline constexpr int DIGIT_SKIP_FRAMES = 0;

    //  豆子相机跳帧数
    //豆子运动较慢，可适当降帧；1表示每2帧处理1帧，2表示每3帧处理1帧
    inline constexpr int BEAN_SKIP_FRAMES = 1;

    //  数字相机数据
    //用于串口数据包区分数据来源，必须与下位机协议一致
    inline constexpr int DIGIT_SOURCE_ID = 0x01;

    //  豆子相机数据源ID）
    inline constexpr int BEAN_SOURCE_ID = 0x02;


    //  YOLO模型路径
    inline const std::string MODEL_PATH = "C:/Users/Administrator/runs/detect/train9/weights/best.onnx";

    //  类别标签文件路径
    inline const std::string CLASSES_FILE = "E:/yolov11/datasets/labels/train/classes.txt";

    //  网络输入宽度（YOLO标准尺寸，必须为32的倍数）
    //与训练时尺寸一致可获得最佳精度
    inline constexpr int NETWORK_WIDTH = 640;

    //  网络输入高度（YOLO标准尺寸，必须为32的倍数）
    inline constexpr int NETWORK_HEIGHT = 640;

    //  目标检测置信度阈值（0.0~1.0，值越高漏检越多但误检越少）
    //工业场景建议0.5~0.7，追求高召回率可设0.3~0.4
    inline constexpr float CONFIDENCE_THRESHOLD = 0.5f;

    //  非极大值抑制阈值（0.0~1.0，值越高保留的重复框越多）
    //通常设0.4~0.5，目标密集场景可降至0.3，稀疏场景可增至0.6
    inline constexpr float NMS_THRESHOLD = 0.4f;

    //  模型类别总数
    //数字1-5 + 豆子类型，共5类
    inline constexpr int NUM_CLASSES = 5;

    //  自动标注置信度阈值
    //高于此阈值的检测结果才会保存为标注数据，建议0.85以上保证质量
    inline constexpr float AUTO_LABEL_THRESH = 0.85f;

    //  二值化阈值（0-255，用于提取白色纸张区域）
    //值越高越只保留亮部，室内日光灯建议180~220，室外可提至230
    inline constexpr int THRESHOLD = 200;

    //  最小轮廓面积（过滤噪声小轮廓）
    //根据目标实际大小调整，A4纸在640x480下约1500~3000像素
    inline constexpr int MIN_AREA = 1500;

    //  目标丢失缓冲帧数
    //值越大输出越稳定但响应延迟越大，典型5~10帧
    inline constexpr int LOST_BUFFER_FRAMES = 1;

    //  低通滤波器系数（0.0~1.0，用于坐标平滑）
    //越小越平滑但延迟越大，0.3适合中等速度目标，快速目标可提至0.5
    inline constexpr float FILTER_ALPHA = 0.5f;

    //  输入图像宽度（相机采集分辨率，影响处理速度和精度）
    inline constexpr int INPUT_WIDTH = 640;

    //  输入图像高度（建议保持4:3或16:9标准比例）
    //480为VGA标准高度，与宽度配合形成640x480标准分辨率
    inline constexpr int INPUT_HEIGHT = 480;

    //  屏幕中心X坐标
    //公式：INPUT_WIDTH / 2
    inline constexpr int SCREEN_CENTER_X = INPUT_WIDTH / 2;

    //  屏幕中心Y坐标
    //公式：INPUT_HEIGHT / 2
    inline constexpr int SCREEN_CENTER_Y = INPUT_HEIGHT / 2;

    //  串口端口号（Windows为COM3、COM4等，Linux为/dev/ttyUSB0）
    //波特率必须与下位机一致
    inline const std::string SERIAL_PORT = "COM3";

    //  串口波特率
    inline constexpr int SERIAL_BAUD = 115200;
}

#endif // CONFIG_H