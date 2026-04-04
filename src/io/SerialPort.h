#ifndef SERIAL_PORT_H
#define SERIAL_PORT_H

#include <string>
#include <cstdint>

class SerialPort {
public:
    SerialPort();
    ~SerialPort();

    /**
     * @brief Open serial port.
     * @param portName e.g. "COM3" on Windows, "/dev/ttyUSB0" on Linux.
     * @param baudRate Baud rate.
     * @return true if successful.
     */
    bool open(const std::string& portName, int baudRate = 115200);

    /**
     * @brief Close serial port.
     */
    void close(); 

    /**
     * @brief Check if port is open.
     */
    bool isOpen() const;

    /**
     * @brief Send data buffer.
     * @param data Pointer to data.
     * @param length Number of bytes.
     * @return Number of bytes sent, or -1 on error.
     */
    int send(const uint8_t* data, size_t length);

    /**
     * @brief Send tracked object data via 11-byte protocol.
     * @param dx X offset (signed).
     * @param dy Y offset (signed).
     * @param classId Class ID (0-255).
     * @param heartbeat Heartbeat counter (0-255).
     * @return true if sent successfully.
     */
    bool sendTrackedObject(int dx, int dy, int classId, uint8_t heartbeat);

private:
    void* m_handle; // Platform-specific handle (HANDLE on Windows, int on Linux)
    bool m_isOpen;
};

#endif // SERIAL_PORT_H