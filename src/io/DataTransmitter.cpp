#include "DataTransmitter.h"
#include "SerialPort.h"
#include <iostream>
#include <iomanip>
#include <sstream>

// 构造函数
DataTransmitter::DataTransmitter()
    : m_serialConnected(false)
    , m_debugMode(false) {
}

// 析构函数
DataTransmitter::~DataTransmitter() {
    closeSerial();
}

// 初始化串口
bool DataTransmitter::initSerial(const std::string& portName, int baudRate) {
    m_serialPort = std::make_unique<SerialPort>();

    if (m_serialPort->open(portName, baudRate)) {
        m_serialConnected = true;
        std::cout << "串口已连接: " << portName << " @" << baudRate << "bps" << std::endl;
        return true;
    } else {
        m_serialConnected = false;
        std::cout << "串口连接失败: " << portName << "，将使用终端输出" << std::endl;
        return false;
    }
}

// 发送数据包
bool DataTransmitter::sendPacket(const DataPacket& packet) {
    return sendData(packet.sourceId, packet.classId, packet.dx, packet.dy);
}

// 发送原始数据
bool DataTransmitter::sendData(uint8_t sourceId, uint8_t classId, int16_t dx, int16_t dy) {
    // 构建协议帧
    uint8_t buffer[11];
    buildProtocolFrame(sourceId, classId, dx, dy, buffer);

    // 转换为vector用于打印
    std::vector<uint8_t> hexData(buffer, buffer + 11);

    // 创建数据包用于打印
    DataPacket packet;
    packet.sourceId = sourceId;
    packet.classId = classId;
    packet.dx = dx;
    packet.dy = dy;

    // 如果串口已连接，则发送
    if (m_serialConnected && m_serialPort && m_serialPort->isOpen()) {
        int bytesSent = m_serialPort->send(buffer, 11);
        if (bytesSent == 11) {
            // 如果调试模式启用，也打印到终端
            if (m_debugMode) {
                printToTerminal(packet, hexData);
            }
            return true;
        } else {
            std::cerr << "串口发送失败，发送了 " << bytesSent << "/11 字节" << std::endl;
            printToTerminal(packet, hexData);  // 失败时也打印
            return false;
        }
    } else {
        // 串口未连接，打印到终端
        printToTerminal(packet, hexData);
        return true;  // 视为成功（因为已经打印）
    }
}

// 构建11字节协议帧
void DataTransmitter::buildProtocolFrame(uint8_t sourceId, uint8_t classId, int16_t dx, int16_t dy, uint8_t* buffer) {
    // 包头
    buffer[0] = HEADER1;
    buffer[1] = HEADER2;

    // 数据源ID
    buffer[2] = sourceId;

    // 类别ID
    buffer[3] = classId;

    // X偏差（低字节在前，小端序）
    buffer[4] = static_cast<uint8_t>(dx & 0xFF);        // 低八位
    buffer[5] = static_cast<uint8_t>((dx >> 8) & 0xFF); // 高八位

    // Y偏差
    buffer[6] = static_cast<uint8_t>(dy & 0xFF);        // 低八位
    buffer[7] = static_cast<uint8_t>((dy >> 8) & 0xFF); // 高八位

    // 校验和（byte[2]到byte[7]）
    buffer[8] = calculateChecksum(&buffer[2], 6);

    // 包尾
    buffer[9] = FOOTER1;  // \r
    buffer[10] = FOOTER2; // \n
}

// 计算校验和
uint8_t DataTransmitter::calculateChecksum(const uint8_t* data, size_t length) const {
    uint8_t sum = 0;
    for (size_t i = 0; i < length; ++i) {
        sum += data[i];
    }
    return sum;
}

// 打印数据包到终端（仅在调试模式下使用）
void DataTransmitter::printToTerminal(const DataPacket& packet, const std::vector<uint8_t>& hexData) const {
    // 不打印，避免阻塞
    // 如需调试，取消下面的注释
    /*
    std::cout << "dx:" << packet.dx << " dy:" << packet.dy << std::endl;
    */
}

// 获取数据源名称
std::string DataTransmitter::getSourceName(uint8_t sourceId) const {
    switch (sourceId) {
        case SOURCE_DIGIT: return "数字相机";
        case SOURCE_BEAN:  return "豆子相机";
        default:           return "未知源(" + std::to_string(static_cast<int>(sourceId)) + ")";
    }
}

// 获取类别名称
std::string DataTransmitter::getClassName(uint8_t sourceId, uint8_t classId) const {
    if (classId == 0) {
        return "未发现";
    }

    if (sourceId == SOURCE_DIGIT) {
        // 数字相机：1-5 对应数字1-5
        if (classId >= 1 && classId <= 5) {
            return "数字" + std::to_string(static_cast<int>(classId));
        }
    } else if (sourceId == SOURCE_BEAN) {
        // 豆子相机：1-3 对应黄/绿/白
        switch (classId) {
            case 1: return "黄豆";
            case 2: return "绿豆";
            case 3: return "白豆";
            default: break;
        }
    }

    return "类别" + std::to_string(static_cast<int>(classId));
}

// 检查串口是否连接
bool DataTransmitter::isSerialConnected() const {
    return m_serialConnected && m_serialPort && m_serialPort->isOpen();
}

// 关闭串口
void DataTransmitter::closeSerial() {
    if (m_serialPort) {
        m_serialPort->close();
        m_serialConnected = false;
    }
}

// 设置调试模式
void DataTransmitter::setDebugMode(bool enabled) {
    m_debugMode = enabled;
    std::cout << "串口调试模式 " << (enabled ? "启用" : "禁用") << std::endl;
}