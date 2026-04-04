#ifndef DATA_TRANSMITTER_H
#define DATA_TRANSMITTER_H

#include <limits>
#include <cstdint>
#include <string>
#include <vector>
#include <memory>

// 前向声明
class SerialPort;

/**
 * @brief 数据源类型枚举
 */
enum DataSource {
    SOURCE_DIGIT = 0x01,   // 来自数字相机
    SOURCE_BEAN  = 0x02    // 来自豆子相机
};

/**
 * @brief 传输数据包结构体
 */
struct DataPacket {
    uint8_t sourceId;      // 数据源ID (0x01:数字相机, 0x02:豆子相机)
    uint8_t classId;       // 类别ID (数字:1-5, 豆子:1-3, 0:未发现)
    int16_t dx;            // X方向偏差（有符号）
    int16_t dy;            // Y方向偏差（有符号）

    DataPacket() : sourceId(SOURCE_DIGIT), classId(0), dx(0), dy(0) {}
};

/**
 * @brief 工业级串口协议与重定向器
 *
 * 封装全新的11字节协议帧，支持双数据源，具有校验和计算。
 * 如果串口未连接，将数据包以易读格式打印到终端。
 */
class DataTransmitter {
public:
    DataTransmitter();
    ~DataTransmitter();

    /**
     * @brief 初始化串口
     * @param portName 串口名称（如"COM3"）
     * @param baudRate 波特率
     * @return true 串口打开成功，false 串口打开失败（将使用终端输出）
     */
    bool initSerial(const std::string& portName, int baudRate = 115200);

    /**
     * @brief 发送数据包
     * @param packet 数据包结构体
     * @return true 发送成功（或已打印到终端），false 发送失败
     */
    bool sendPacket(const DataPacket& packet);

    /**
     * @brief 发送原始数据（高级用法）
     * @param sourceId 数据源ID
     * @param classId 类别ID
     * @param dx X偏差
     * @param dy Y偏差
     * @return true 发送成功，false 发送失败
     */
    bool sendData(uint8_t sourceId, uint8_t classId, int16_t dx, int16_t dy);

    /**
     * @brief 检查串口是否连接
     * @return true 串口已连接，false 串口未连接（使用终端输出）
     */
    bool isSerialConnected() const;

    /**
     * @brief 关闭串口
     */
    void closeSerial();

    /**
     * @brief 设置调试模式
     * @param enabled 是否启用调试模式（即使串口连接也打印到终端）
     */
    void setDebugMode(bool enabled);

private:
    /**
     * @brief 构建11字节协议帧
     * @param sourceId 数据源ID
     * @param classId 类别ID
     * @param dx X偏差
     * @param dy Y偏差
     * @param buffer 输出缓冲区（必须至少11字节）
     */
    void buildProtocolFrame(uint8_t sourceId, uint8_t classId, int16_t dx, int16_t dy, uint8_t* buffer);

    /**
     * @brief 计算校验和（byte[2]到byte[7]的累加和）
     * @param data 数据指针
     * @param length 数据长度
     * @return 校验和
     */
    uint8_t calculateChecksum(const uint8_t* data, size_t length) const;

    /**
     * @brief 打印数据包到终端（易读格式）
     * @param packet 数据包
     * @param hexData 十六进制数据
     */
    void printToTerminal(const DataPacket& packet, const std::vector<uint8_t>& hexData) const;

    /**
     * @brief 获取数据源名称
     * @param sourceId 数据源ID
     * @return 数据源名称字符串
     */
    std::string getSourceName(uint8_t sourceId) const;

    /**
     * @brief 获取类别名称
     * @param sourceId 数据源ID
     * @param classId 类别ID
     * @return 类别名称字符串
     */
    std::string getClassName(uint8_t sourceId, uint8_t classId) const;

private:
    // 串口对象
    std::unique_ptr<SerialPort> m_serialPort;

    // 配置
    bool m_serialConnected;
    bool m_debugMode;

    // 协议常量
    static const uint8_t HEADER1 = 0x55;
    static const uint8_t HEADER2 = 0xAA;
    static const uint8_t FOOTER1 = 0x0D;  // \r
    static const uint8_t FOOTER2 = 0x0A;  // \n
};

#endif // DATA_TRANSMITTER_H