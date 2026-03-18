#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'Antoine Weill--Duflos'
__version__ = '1.0.0'
__date__ = '2021/03/17'
__description__ = 'Python API definitions of the Haply hAPI'

import serial
import struct
import math
import sys
from typing import List
import array


class Actuator:
    __actuator = 0
    __direction = 0
    __torque = 0
    __actuatorPort = 0

    def __init__(self, actuator=0, direction=0, port=0):
        """Create an Actuator

        Args:
            actuator (int, optional): actuator index. Defaults to 0.
            direction (int, optional): direction. Defaults to 0.
            port (int, optional): motor port position for actuator. Defaults to 0.
        """
        self.__actuator = actuator
        self.__direction = direction
        self.__actuatorPort = port

    def set_actuator(self, actuator):
        """Set actuator index parameter of sensor

        Args:
            actuator ([type]): [index]
        """
        self.__actuator = actuator

    def set_direction(self, direction):
        """Set actuator rotation direction

        Args:
            direction ([type]): direction of rotation
        """
        self.__direction = direction

    def set_port(self, port):
        """sets motor port position

        Args:
            port ([type]): port index
        """
        self.__actuatorPort = port

    def set_torque(self, torque):
        """sets torque variable to the given torque value

        Args:
            torque ([type]): new torque value for update
        """
        self.__torque = torque

    def get_actuator(self):
        """ get actuator index

        Returns:
            [type]: actuator index
        """
        return self.__actuator

    def get_direction(self):
        return self.__direction

    def get_port(self):
        return self.__actuatorPort

    def get_torque(self):
        return self.__torque


class Board:
    """Define Board system

    Returns:
        [type]: [description]
    """
    __port: serial.Serial
    __deviceID = 0

    def __init__(self, app, port, baud):
        """Initialise Bard

        Args:
            app (string): name of the app
            port (string): com port
            baud (int): rate
        """
        self.__applet = app
        self.__port = serial.Serial(port, baud)
        self.__reset_board()

    def floatToBits(self,f):
        s = struct.pack('>f', f)
        return struct.unpack('>I', s)[0]

    def bitsToFloat(self, b):
        s = struct.pack('>I', b)
        return struct.unpack('>f', s)[0]

    def float_to_bytes(self, float_data):
        segments = bytearray(4)
        #convert float to int32
        int32_data = self.floatToBits(float_data)
        segments[3] = (int32_data >> 24) & 0xFF
        segments[2] = (int32_data >> 16) & 0xFF
        segments[1] = (int32_data >> 8) & 0xFF
        segments[0] = int32_data & 0xFF
        return segments

    def bytes_to_float(self, data):
        temp = 0
        temp = (temp | (data[3] & 0xFF)) << 8
        temp = (temp | (data[2] & 0xFF)) << 8
        temp = (temp | (data[1] & 0xFF)) << 8
        temp = (temp | (data[0] & 0xFF))

        val = self.bitsToFloat(temp)
        return val



    def transmit(self, communicationType, deviceID, bData, fData):
        outData = bytearray(2 + len(bData) + 4*len(fData))
        segments = bytearray(4)

        outData[0] = communicationType
        outData[1] = deviceID
        self.__deviceID = deviceID
        outData[2:2+len(bData)] = bData
        j = 2+len(bData)
        # TODO: replace with a single pack
        for i in range(0, len(fData)):
            # store the floats into the bytearray one by one
            segments = self.float_to_bytes(fData[i])#bytearray(struct.pack('!f', fData[i]))
            outData[j:j+4] = segments
            j = j+4
        wrote = self.__port.write(outData)

    def receive(self, communicationType, deviceID, expected):
        inData = bytearray(1+4*expected)
        data = [None]*expected
        inData = self.__port.read(1+4*expected)
        if(inData[0] != deviceID):
            sys.stderr.write("Error, another device expects this data!\n")
        buf = inData[1:expected*4+1]
        for i in range(0, expected):
            data[i] = self.bytes_to_float(buf[i*4:i*4+4])
        #data = struct.unpack('!'+str(expected)+'f', buf)
        #print(expected)
        #print(data)
        return data

    def data_available(self):
        available = False
        if(self.__port.in_waiting > 0):
            available = True
        return available

    def __reset_board(self):
        communicationType = 0
        deviceID = 0
        bData = bytearray(0)
        fData = bytearray(0)
        self.transmit(communicationType, deviceID, bData, fData)


class Mechanisms:
    def forwardKinematics(self):
        pass

    def torqueCalculation(self):
        pass

    def forceCalculation(self):
        pass

    def positionControl(self):
        pass

    def inverseKinematics(self):
        pass

    def set_mechanism_parameters(self, parameters):
        pass

    def set_sensor_data(self, data):
        pass

    def get_coordinate(self):
        pass

    def get_torque(self):
        pass

    def get_angle(self):
        pass


class Pwm:
    __pin = 0
    __value = 0

    def __init__(self, pin=0, pulsewidth=0):
        self.__pin = pin
        if(pulsewidth > 100.0):
            self.__value = 255
        else:
            self.__value = int(pulsewidth*255/100)

    def set_pin(self, pin):
        self.__pin = pin

    def set_pulse(self, percent):
        if(percent > 100.0):
            self.__value = 255
        elif(percent < 0):
            self.__value = 0
        else:
            self.__value = int(percent*255/100)

    def get_pin(self):
        return self.__pin

    def get_value(self):
        return self.__value

    def get_pulse(self):
        percent = self.__value*100/255
        return percent


class Sensor:
    __encoder = 0
    __direction = 0
    __encoder_offset = 0
    __encoder_resolution = 0
    __value = 0
    __port = 0

    def __init__(self, encoder=0, direction=0, offset=0, resolution=0, port=0):
        self.__encoder = encoder
        self.__direction = direction
        self.__encoder_offset = offset
        self.__encoder_resolution = resolution
        self.__port = port

    def set_encoder(self, encoder):
        self.__encoder = encoder

    def set_direction(self, direction):
        self.__direction = direction

    def set_offset(self, offset):
        self.__encoder_offset = offset

    def set_resolution(self, resolution):
        self.__encoder_resolution = resolution

    def set_port(self, port):
        self.__port = port

    def set_value(self, value):
        self.__value = value

    def get_encoder(self):
        return self.__encoder

    def get_direction(self):
        return self.__direction

    def get_offset(self):
        return self.__encoder_offset

    def get_resolution(self):
        return self.__encoder_resolution

    def get_port(self):
        return self.__port

    def get_value(self):
        return self.__value


class Device:
    __deviceLink = 0
    __deviceID = 0
    __mechanism = 0
    __communicationType = 0
    __actuatorsActive = 0
    __motors: List[Actuator] = []
    __encodersActive = 0
    __encoders: List[Sensor] = []
    __sensorsActive = 0
    __sensors: List[Sensor] = []
    __pwmsActive = 0
    __pwms: List[Pwm] = []
    __actuatorPositions = bytearray([0, 0, 0, 0])
    __encoderPositions = bytearray([0, 0, 0, 0])

    def __init__(self, deviceID, deviceLink):
        self.__deviceID = deviceID
        self.__deviceLink = deviceLink

    def add_actuator(self, actuator, rotation, port):
        error = False
        if(port < 1 or port > 4):
            sys.stderr.write("error: encoder port index out of bounds\n")
            error = True
        if(actuator < 1 or actuator > 4):
            sys.stderr.write("error: encoder index out of bounds\n")
            error = True
        j = 0
        # TODO: figure what is really hapenning here ?
        for i in range(self.__actuatorsActive):
            if(self.__motors[j].get_actuator() < actuator):
                j += 1
            elif(self.__motors[j].get_actuator() == actuator):
                sys.stderr.write("error: actuator "+actuator +
                                 " has already been set")
                error = True
        if(not error):
            temp: List[Actuator] = [None] * (self.__actuatorsActive + 1)
            temp[0:len(self.__motors)] = self.__motors[0:len(self.__motors)]
            if(j < self.__actuatorsActive):
                temp[j+1:] = self.__motors[j:]
            temp[j] = Actuator(actuator, rotation, port)
            self.__actuator_assignment(actuator, port)
            self.__motors = temp
            self.__actuatorsActive += 1

    def add_encoder(self, encoder, rotation, offset, resolution, port):
        error = False
        if(port < 1 or port > 4):
            sys.stderr.write("error: encoder port index out of bounds")
            error = True
        if(encoder < 1 or encoder > 4):
            sys.stderr.write("error: encoder index out of bound!")
            error = True
        j = 0
        for i in range(self.__encodersActive):
            if(self.__encoders[i].get_encoder() < encoder):
                j += 1
            if(self.__encoders[i].get_encoder() == encoder):
                sys.stderr.write("error: encoder " +
                                 encoder + " has already been set")
                error = True
        if(not error):
            temp: List[Sensor] = [None] * (self.__encodersActive + 1)
            temp[0:len(self.__encoders)] = self.__encoders
            if(j < self.__encodersActive):
                temp[j+1:] = self.__encoders[j:]
            temp[j] = Sensor(encoder, rotation, offset, resolution, port)
            self.__encoder_assignment(encoder, port)
            self.__encoders = temp
            self.__encodersActive += 1

    def add_analog_sensor(self, pin):
        error = False
        port = pin[0]
        number = pin[1:]
        value = int(number)
        value = value + 54

        for i in range(self.__sensorsActive):
            if(value == self.__sensors[i].get_port()):
                sys.stderr.write("error: Analog pin A" +
                                 (value - 54) + " has already been set")
                error = True

        if(port != 'A' or value < 54 or value > 65):
            sys.stderr.write("error: outside analog pin range")
            error = True

        if(not error):
            temp = self.__sensors
            temp[self.__sensorsActive] = Sensor()
            temp[self.__sensorsActive].set_port(value)
            self.__sensors = temp
            self.__sensorsActive += 1

    def add_pwm_pin(self, pin):
        error = False
        for i in range(self.__pwmsActive):
            if(pin == self.__pwms[i].get_pin()):
                sys.stderr.write("error: pwm pin " + pin +
                                 " has already been set")
                error = True
        if(pin < 0 or pin > 13):
            sys.stderr.write("error: outside pwm pin range")
            error = True
        if(pin == 0 or pin == 1):
            sys.stdout.write(
                "warning: 0 and 1 are not pwm pints on Haply M3 or Haply original")
        if(not error):
            temp = self.__pwms
            temp[self.__pwmsActive] = Pwm()
            temp[self.__pwmsActive].set_pin(pin)
            self.__pwms = temp
            self.__pwmsActive += 1

    def set_mechanism(self, mechanism):
        self.__mechanism = mechanism

    def device_set_parameters(self):
        self.__communicationType = 1
        encoderParameters = []
        encoderParams = bytearray()
        motorParams = bytearray()
        sensorParams = bytearray()
        pwmParams = bytearray()

        if(self.__encodersActive > 0):
            encoderParams = bytearray(self.__encodersActive + 1)
            control = 0
            for i in range(len(self.__encoders)):
                if(self.__encoders[i].get_encoder() != (i+1)):
                    sys.stderr.write("warning, improper encoder indexing")
                    self.__encoders[i].set_encoder(i+1)
                    self.__encoderPositions[self.__encoders[i].get_port(
                    ) - 1] = self.__encoders[i].get_encoder()
            for i in range(len(self.__encoderPositions)):
                control = control >> 1
                if(self.__encoderPositions[i] > 0):
                    control = control | 0x0008
            encoderParams[0] = control
            encoderParameters = [None]*(2*self.__encodersActive)
            j = 0
            for i in range(len(self.__encoderPositions)):
                if(self.__encoderPositions[i] > 0):
                    encoderParameters[2 *
                                      j] = self.__encoders[self.__encoderPositions[i]-1].get_offset()
                    encoderParameters[2*j +
                                      1] = self.__encoders[self.__encoderPositions[i]-1].get_resolution()
                    j += 1
                    encoderParams[j] = self.__encoders[self.__encoderPositions[i] -
                                                       1].get_direction()
        else:
            encoderParams = bytearray(1)
            encoderParams[0] = 0
            encoderParameters = []
        if(self.__actuatorsActive > 0):
            motorParams = bytearray(self.__actuatorsActive + 1)
            control = 0
            for i in range(len(self.__motors)):
                if(self.__motors[i].get_actuator() != (i+1)):
                    sys.stderr.write("warning, improper actuator indexing, "+ str(self.__motors[i].get_actuator()) + " instead of " + str(i+1) + "\n")
                    self.__motors[i].set_actuator(i+1)
                    self.__actuatorPositions[self.__motors[i].get_port(
                    ) - 1] = self.__motors[i].get_actuator()
            for i in range(len(self.__actuatorPositions)):
                control = control >> 1
                if(self.__actuatorPositions[i] > 0):
                    control = control | 0x0008
            motorParams[0] = control
            j = 1
            for i in range(len(self.__actuatorPositions)):
                if(self.__actuatorPositions[i] > 0):
                    motorParams[j] = self.__motors[self.__actuatorPositions[i] -
                                            1].get_direction()
                    j += 1
        else:
            motorParams = bytearray(1)
            motorParams[0] = 0
        
        if(self.__sensorsActive > 1):
            sensorParams = bytearray(self.__sensorsActive + 1)
            sensorParams[0] = self.__sensorsActive
            for i in range(self.__sensorsActive):
                sensorParams[i+1] = self.__sensors[i].get_port()
            sensorParams = bytearray(array.array(sensorParams.typecode, sorted(sensorParams)))
            for i in range(self.__sensorsActive):
                self.__sensors[i].set_port(sensorParams[i+1])
        else:
            sensorParams = bytearray(1)
            sensorParams[0] = 0
        if(self.__pwmsActive > 0):
            temp = bytearray(self.__pwmsActive)
            pwmParams = bytearray(self.__pwmsActive + 1)
            pwmParams[0] = self.__pwmsActive

            for i in range(self.__pwmsActive):
                temp[i] = self.__pwms[i].get_pin()
            temp = bytearray(array.array(temp.typecode, sorted(temp)))

            for i in range(self.__pwmsActive):
                self.__pwms[i].set_pin(temp[i])
                pwmParams[i+1] = self.__pwms[i].get_pin()
        else:
            pwmParams = bytearray(1)
            pwmParams[0] = 0
        encMtrSenPwm = bytearray(
            len(motorParams) + len(encoderParams) + len(sensorParams) + len(pwmParams))
        encMtrSenPwm[0:len(motorParams)] = motorParams
        encMtrSenPwm[len(motorParams):len(motorParams) +
                     len(encoderParams)] = encoderParams
        encMtrSenPwm[len(motorParams)+len(encoderParams):len(motorParams) +
                     len(encoderParams)+len(sensorParams)] = sensorParams
        encMtrSenPwm[len(motorParams)+len(encoderParams) +
                     len(sensorParams):] = pwmParams

        self.__deviceLink.transmit(
            self.__communicationType, self.__deviceID, encMtrSenPwm, encoderParameters)

    def __actuator_assignment(self, actuator, port):
        if(self.__actuatorPositions[port - 1] > 0):
            sys.stderr.println("warning, double check actuator port usage")
        self.__actuatorPositions[port-1] = actuator

    def __encoder_assignment(self, encoder, port):
        if(self.__encoderPositions[port - 1] > 0):
            sys.stderr.println("warning, double check encoder port usage")
        self.__encoderPositions[port - 1] = encoder

    def device_read_data(self):
        self.__communicationType = 2
        dataCount = 0
        device_data = self.__deviceLink.receive(
            self.__communicationType, self.__deviceID, self.__sensorsActive + self.__encodersActive)

        for i in range(self.__sensorsActive):
            self.__sensors[i].set_value(device_data[dataCount])
            dataCount += 1

        for i in range(len(self.__encoderPositions)):
            if(self.__encoderPositions[i] > 0):
                self.__encoders[self.__encoderPositions[i] -
                                1].set_value(device_data[dataCount])
                dataCount += 1

    def device_read_request(self):
        self.__communicationType = 2
        pulses = bytearray(self.__pwmsActive)
        encoderRequest = []*self.__actuatorsActive

        for i in range(len(self.__pwms)):
            pulses[i] = self.__pwms[i].get_values()

        j = 0
        for i in range(len(self.__actuatorPositions)):
            if(self.__actuatorPositions[i] > 0):
                encoderRequest[j] = 0
                j += 1

        self.__deviceLink.transmit(
            self.__communicationType, self.__deviceID, pulses, encoderRequest)

    def device_write_torques(self):
        self.__communicationType = 2
        pulses = bytearray(self.__pwmsActive)
        deviceTorques = [None]*self.__actuatorsActive

        for i in range(len(self.__pwms)):
            pulses[i] = self.__pwms[i].get_value()

        j = 0
        for i in range(len(self.__actuatorPositions)):
            if(self.__actuatorPositions[i] > 0):
                deviceTorques[j] = self.__motors[self.__actuatorPositions[i]-1].get_torque()
                j += 1
                
        self.__deviceLink.transmit(
            self.__communicationType, self.__deviceID, pulses, deviceTorques)

    def set_pwm_pulse(self, pin, pulse):
        for i in range(len(self.__pwms)):
            if(self.__pwms[i].get_pin() == pin):
                self.__pwms[i].set_pulse(pulse)

    def get_pwm_pulse(self, pin):
        pulse = 0
        for i in range(len(self.__pwms)):
            if(self.__pwms[i].get_pin() == pin):
                pulse = self.__pwms[i].get_pulse()
        return pulse

    def get_device_angles(self):
        angles = [None]*self.__encodersActive
        for i in range(self.__encodersActive):
            angles[i] = self.__encoders[i].get_value()
        return angles

    def get_sensor_data(self):
        data = [None]*self.__sensorsActive
        for i in range(self.__sensorsActive):
            data[i] = self.__sensors[i].get_value()
        return data

    def get_device_position(self, angles):
        self.__mechanism.forwardKinematics(angles)
        endEffectorPosition = self.__mechanism.get_coordinate()
        return endEffectorPosition

    def set_device_torques(self, forces):
        self.__mechanism.torqueCalculation(forces)
        torques = self.__mechanism.get_torque()
        for i in range(self.__actuatorsActive):
            self.__motors[i].set_torque(torques[i])
        return torques


class Pantograph(Mechanisms):
    __l = __L = __d = 0
    __th1 = __th2 = 0
    __tau1 = __tau2 = 0
    __f_x = __f_y = 0
    __q_x = __q_y = 0
    __x_E = __y_E = 0
    __pi = math.pi
    __J11 = __J12 = __J21 = __J22 = 0
    __gain = 1

    def __init__(self, version=2):
        self.__l = 0.07
        self.__L = 0.09
        if version == 3:
            self.__d = 0.038
        else:
            self.__d = 0
    
    def forwardKinematics(self, angles):
        l1 = self.__l
        l2 = self.__l
        L1 = self.__L
        L2 = self.__L

        self.__th1 = self.__pi / 180 * angles[0]
        self.__th2 = self.__pi / 180 * angles[1]

        c1 = math.cos(self.__th1)
        c2 = math.cos(self.__th2)
        s1 = math.sin(self.__th1)
        s2 = math.sin(self.__th2)

        xA = l1 * c1
        yA = l1 * s1
        xB = self.__d + l2 * c2
        yB = l2 * s2
        R = math.pow(xA, 2) + math.pow(yA, 2)
        S = math.pow(xB, 2) + math.pow(yB, 2)
        hx = xB-xA
        hy = yB-yA
        hh = math.pow(hx,2) + math.pow(hy,2)
        hm = math.sqrt(hh)
        if (hm == 0):
            cB = 0
            h1x = 0
            h1y = 0
        else:
            cB = - (math.pow(L2,2)-math.pow(L1,2)-hh)/(2*L1*hm)
            h1x = L1*cB * hx/hm
            h1y = L1*cB * hy/hm
        h1h1 = math.pow(h1x,2) + math.pow(h1y,2)
        h1m = math.sqrt(h1h1)
        sB = math.sqrt(1-pow(cB,2))
        if (h1m == 0):
            lx = 0
            ly = 0
        else:
            lx = -L1*sB*h1y/h1m
            ly = L1*sB*h1x/h1m

        x_P = xA + h1x + lx
        y_P = yA + h1y + ly

        phi1 = math.acos((x_P-l1*c1)/L1)
        phi2 = math.acos((x_P- self.__d-l2*c2)/L2)

        c11 = math.cos(phi1)
        s11 = math.sin(phi1)
        c22 = math.cos(phi2)
        s22 = math.sin(phi2)

        dn = L1 * (c11 * s22 - c22 * s11)
        if(dn == 0):
            eta = 0
            nu = 0
        else:
            eta = (-L1 * c11 * s22 + L1 * c22 * s11 - c1 * l1 * s22 + c22 * l1 * s1) / dn
            nu = l2 * (c2 * s22 - c22 * s2)/dn

        self.__J11 = -L1 * eta * s11 - L1 * s11 - l1 * s1
        self.__J12 = L1 * c11 * eta + L1 * c11 + c1 * l1
        self.__J21 = -L1 * s11 * nu
        self.__J22 = L1 * c11 * nu

        self.__x_E = x_P
        self.__y_E = y_P
    
    def torqueCalculation(self, force):
        self.__f_x = force[0]
        self.__f_y = force[1]

        self.__tau1 = self.__J11*self.__f_x + self.__J12*self.__f_y
        self.__tau2 = self.__J21*self.__f_x + self.__J22*self.__f_y

        self.__tau1*= self.__gain
        self.__tau2*= self.__gain

    def op_velocityCalculation(self, q):
        op_vels = [0.0,0.0]
        self.__q_x = q[0]
        self.__q_y = q[1]

        op_vels[0] = self.__J11*self.__q_x + self.__J21*self.__q_y
        op_vels[1] = self.__J12*self.__q_x + self.__J22*self.__q_y

        op_vels[0]*= self.__gain
        op_vels[1]*= self.__gain

        return op_vels

    def forceCalculation(self):
        pass

    def positionControl(self):
        pass

    def inverseKinematics(self):
        pass
    
    def set_mechanism_parameters(self, parameters):
        self.__l = parameters[0]
        self.__L = parameters[1]
        self.__d = parameters[2]

    def set_sensor_data(self, data):
        pass

    def get_coordinate(self):
        return [self.__x_E, self.__y_E]
    
    def get_torque(self):
        return [self.__tau1, self.__tau2]

    def get_angle(self):
        return [self.__th1, self.__th2]