import serial
import serial.tools.list_ports
import time
import struct

class Connection:
    def __init__(self, port=None, gyro=False):
        if port is None:
            print("Serial Ports Available: " + str([port[0] for port in serial.tools.list_ports.comports()]))

            found = False
            for port in list(serial.tools.list_ports.comports()):
                # Test port to see if they are a nybble device
                self.conn = serial.Serial(port[0], 115200, timeout=1)
                line = self.read_line()
                if line == b'\x00':
                    lines = self.conn.readlines()
                    if b'Nybble\r\n' in lines:
                        print("Found Nybble on port: " + port[0])
                        found = True
                        break
        else:
            self.conn = serial.Serial(port, 115200, timeout=1)
            found = False
            line = self.read_line()
            if line == b'\x00':
                lines = self.conn.readlines()
                if b'Nybble\r\n' in lines:
                    print("Found Nybble on port: " + port[0])
                    found = True

        if not self.conn or not found:
            raise Exception("No Nybble found")
        
        time.sleep(3)
        
        self.gyro = gyro
        if not self.gyro:
            self.send_command('g', get_confirmation=False)
            print(self.read_line())
            time.sleep(1)
        
    def alarm(self):
        self.send_command('u2 10', get_confirmation=False)
        print(self.read_line())
        print(self.read_line())
        
        time.sleep(3)
        
    def close(self):
        self.send_command('u', get_confirmation=False)
        self.send_command('d', get_confirmation=False)
        self.conn.close()
        
    def get_joint_angles(self):
        # self.send_command('j')
        # self.read_line()
        # return [int(a.strip()) for a in self.read_line().decode().split(',') if a.strip() != ""]
        
        # The above should not be used because for some reason it also moves the joints in our use case
        
        return [0] * 16
    
    def set_joint_angles(self, angles):
        # Angles are encoded in binary
        angles = list(map(int, angles))
        payload = 'l'.encode() + struct.pack('b' * len(angles), *angles) + '~'.encode()
        
        self.clear_input()
            
        print("Sending command " + str(payload))
        self.conn.write(payload)
        print(self.read_line())
        
        time.sleep(0.5)
    
    def get_IMU(self):
        if not self.gyro:
            return [0, 0, 0, 0, 0, 0]
        self.send_command('v')
        IMU_data = self.read_line().decode()
        return [float(a.strip()) for a in IMU_data.split(':')[1].split('\t') if a.strip() != ""][:-1]
    
    def read_line(self):
        line = self.conn.readline()
        while "Low power" in line.decode():
            print("Skipping low power message")
            line = self.conn.readline()
        return line
    
    def clear_input(self):
        #self.conn.reset_input_buffer()
        #self.conn.readlines()
        
        num_to_read = self.conn.in_waiting
        
        if num_to_read > 0:
            print(self.conn.read(self.conn.in_waiting))
            print("Had to read " + str(num_to_read) + " bytes before sending command")
            time.sleep(0.1)
    
    def send_command(self, command, get_confirmation=True):
        self.clear_input()

        print("Sending command " + command)
        self.conn.write(command.encode())
        if get_confirmation:
            line = self.read_line()
            if line != command.encode() + b'\r\n':
                print(line)
                print("Above line was not confirmation of command!")
                print("SLOWING DOWN!")
                time.sleep(1)
                self.send_command(command, get_confirmation)
            #print("Got confirmation")

# Test
if __name__ == "__main__":
    conn = Connection()
    
    conn.alarm()
    
    print(conn.get_joint_angles())
    print(conn.get_IMU())
    print(conn.get_joint_angles())
    print(conn.get_IMU())
    print(conn.get_joint_angles())
    print(conn.get_IMU())
    conn.set_joint_angles([20, 0, 0, 0, 0, 0, 0, 0, 45, 45, 45, 45, 36, 36, 36, 36])
    
    print(conn.get_joint_angles())
    print(conn.get_IMU())
    
    conn.close()