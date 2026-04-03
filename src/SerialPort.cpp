#include "SerialPort.h"
#include <iostream>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#endif

SerialPort::SerialPort() : m_handle(nullptr), m_isOpen(false) {}

SerialPort::~SerialPort() {
    close();
}

bool SerialPort::open(const std::string& portName, int baudRate) {
    if (isOpen()) {
        close();
    }

#ifdef _WIN32
    // Windows implementation
    std::string fullPortName = portName;
    if (fullPortName.find("\\\\.\\") == std::string::npos) {
        fullPortName = "\\\\.\\" + fullPortName;
    }
    HANDLE hSerial = CreateFileA(fullPortName.c_str(),
                                 GENERIC_READ | GENERIC_WRITE,
                                 0,
                                 NULL,
                                 OPEN_EXISTING,
                                 FILE_ATTRIBUTE_NORMAL,
                                 NULL);
    if (hSerial == INVALID_HANDLE_VALUE) {
        std::cerr << "Failed to open serial port " << portName << std::endl;
        return false;
    }

    DCB dcbSerialParams = { 0 };
    dcbSerialParams.DCBlength = sizeof(dcbSerialParams);
    if (!GetCommState(hSerial, &dcbSerialParams)) {
        CloseHandle(hSerial);
        return false;
    }
    dcbSerialParams.BaudRate = baudRate;
    dcbSerialParams.ByteSize = 8;
    dcbSerialParams.StopBits = ONESTOPBIT;
    dcbSerialParams.Parity = NOPARITY;
    if (!SetCommState(hSerial, &dcbSerialParams)) {
        CloseHandle(hSerial);
        return false;
    }

    // Set timeouts
    COMMTIMEOUTS timeouts = { 0 };
    timeouts.ReadIntervalTimeout = 50;
    timeouts.ReadTotalTimeoutConstant = 50;
    timeouts.ReadTotalTimeoutMultiplier = 10;
    timeouts.WriteTotalTimeoutConstant = 50;
    timeouts.WriteTotalTimeoutMultiplier = 10;
    if (!SetCommTimeouts(hSerial, &timeouts)) {
        CloseHandle(hSerial);
        return false;
    }

    m_handle = hSerial;
    m_isOpen = true;
    std::cout << "Serial port " << portName << " opened at " << baudRate << " baud." << std::endl;
    return true;
#else
    // Linux implementation
    int fd = ::open(portName.c_str(), O_RDWR | O_NOCTTY | O_NDELAY);
    if (fd == -1) {
        std::cerr << "Failed to open serial port " << portName << std::endl;
        return false;
    }

    struct termios options;
    tcgetattr(fd, &options);
    cfsetispeed(&options, baudRate);
    cfsetospeed(&options, baudRate);
    options.c_cflag |= (CLOCAL | CREAD);
    options.c_cflag &= ~PARENB;
    options.c_cflag &= ~CSTOPB;
    options.c_cflag &= ~CSIZE;
    options.c_cflag |= CS8;
    options.c_lflag &= ~(ICANON | ECHO | ECHOE | ISIG);
    options.c_iflag &= ~(IXON | IXOFF | IXANY);
    options.c_oflag &= ~OPOST;
    options.c_cc[VMIN] = 0;
    options.c_cc[VTIME] = 10;
    tcsetattr(fd, TCSANOW, &options);

    m_handle = reinterpret_cast<void*>(fd);
    m_isOpen = true;
    std::cout << "Serial port " << portName << " opened at " << baudRate << " baud." << std::endl;
    return true;
#endif
}

void SerialPort::close() {
    if (!isOpen()) {
        return;
    }
#ifdef _WIN32
    CloseHandle(reinterpret_cast<HANDLE>(m_handle));
#else
    ::close(reinterpret_cast<int>(m_handle));
#endif
    m_handle = nullptr;
    m_isOpen = false;
}

bool SerialPort::isOpen() const {
    return m_isOpen;
}

int SerialPort::send(const uint8_t* data, size_t length) {
    if (!isOpen()) {
        return -1;
    }
#ifdef _WIN32
    DWORD bytesWritten;
    if (!WriteFile(reinterpret_cast<HANDLE>(m_handle), data, length, &bytesWritten, NULL)) {
        return -1;
    }
    return static_cast<int>(bytesWritten);
#else
    int bytesWritten = ::write(reinterpret_cast<int>(m_handle), data, length);
    return bytesWritten;
#endif
}

bool SerialPort::sendTrackedObject(int dx, int dy, int classId, uint8_t heartbeat) {
    // 11-byte protocol:
    // Byte 0:   Start marker 0xAA
    // Byte 1-2: X offset (int16_t, little-endian)
    // Byte 3-4: Y offset (int16_t, little-endian)
    // Byte 5:   Class ID (0-255)
    // Byte 6:   Heartbeat counter (0-255)
    // Byte 7-10: CRC32 (optional, but we'll use simple checksum for now)
    // For simplicity, we use simple checksum: sum of bytes 0-6 modulo 256.
    // Actually we need 11 bytes total, so we can fill remaining with zeros or crc32.
    // Let's define: bytes 7-10 = 0x00 (reserved).
    // Byte 10: checksum (sum of bytes 0-9) & 0xFF
    // This makes 11 bytes.

    uint8_t buffer[11];
    buffer[0] = 0xAA; // start
    int16_t dx16 = static_cast<int16_t>(dx);
    int16_t dy16 = static_cast<int16_t>(dy);
    buffer[1] = static_cast<uint8_t>(dx16 & 0xFF);
    buffer[2] = static_cast<uint8_t>((dx16 >> 8) & 0xFF);
    buffer[3] = static_cast<uint8_t>(dy16 & 0xFF);
    buffer[4] = static_cast<uint8_t>((dy16 >> 8) & 0xFF);
    buffer[5] = static_cast<uint8_t>(classId & 0xFF);
    buffer[6] = heartbeat;
    buffer[7] = 0x00; // reserved
    buffer[8] = 0x00; // reserved
    buffer[9] = 0x00; // reserved
    // Calculate checksum
    uint8_t sum = 0;
    for (int i = 0; i < 10; ++i) {
        sum += buffer[i];
    }
    buffer[10] = sum;

    int sent = send(buffer, sizeof(buffer));
    return sent == sizeof(buffer);
}