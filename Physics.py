# -*- coding: utf-8 -*-

import numpy as np
import math
import matplotlib.pyplot as plt
from HaplyHAPI import Board, Device, Mechanisms, Pantograph
import sys, serial, glob
from serial.tools import list_ports
import time


import serial.tools.list_ports


class Physics:
    def __init__(self,reverse_motor_order=False,hardware_version=3):
        #return True if a device is found, False if no device is found
        CW = 0
        CCW = 1
        haplyBoard = Board
        device = Device
        SimpleActuatorMech = Mechanisms
        pantograph = Pantograph
        
        #########Open the connection with the arduino board#########
        self.port = self.serial_ports()   ##port contains the communication port or False if no device
        if hardware_version==3:
            self.l1 = 0.07
            self.l2 = 0.09
            self.d = 0.038
        else:
            self.l1 = 0.07
            self.l2 = 0.09
            self.d = 0.0
        
        if self.port:
            print("Board found on port %s"%self.port[0])
            self.haplyBoard = Board("test", self.port[0], 0)
            self.device = Device(5, self.haplyBoard)
            self.pantograph = Pantograph(hardware_version)
            self.device.set_mechanism(self.pantograph)
            if hardware_version == 3:
                if reverse_motor_order: #sometimes the motor wires for version 3 are connected in reverse
                    self.device.add_actuator(2, CCW, 2)
                    self.device.add_actuator(1, CCW, 1)
                    self.device.add_encoder(2, CCW, 82.7, 4880, 2) #angle a1
                    self.device.add_encoder(1, CCW, 97.3, 4880, 1) #angle a2
                else:
                    self.device.add_actuator(1, CCW, 2)
                    self.device.add_actuator(2, CCW, 1)
                    #self.device.add_encoder(1, CCW, 97.3, 4880, 2) #fully extended starting position
                    #self.device.add_encoder(2, CCW, 82.7, 4880, 1)
                    self.device.add_encoder(1, CCW, 168, 4880, 2) #fully retracted starting position
                    self.device.add_encoder(2, CCW, 12, 4880, 1)
            else: #not tested with hardware version 2
                self.device.add_actuator(1, CCW, 2)
                self.device.add_actuator(2, CW, 1)
                self.device.add_encoder(1, CCW, 241, 10752, 2)
                self.device.add_encoder(2, CW, -61, 10752, 1)
            
            self.device.device_set_parameters()
            self.device_present = True
            
            #THE DEVICE MUST HAVE THE TORQUE WRITTEN BEFORE IT CAN PROVIDE DATA!!!!!!!
            #This section prevents the program from not having available data for 1 to 2 initial frames
            start_time = time.time()
            while True:
                if not self.haplyBoard.data_available():
                    #port present, but no data available. Setting initial torques
                    self.device.set_device_torques(np.zeros(2))
                    self.device.device_write_torques()
                    time.sleep(0.001) #pause for 1 millisecond
                    if time.time()-start_time>5.0: #this is taking longer than 5 seconds...
                        raise ValueError("Haply board present, but not providing data!")
                else:
                    #data now available! Proceed.
                    print("[PHYSICS]: Haply found and data available. Ready to run!")
                    break
        else:
            print("[PHYSICS]: No compatible device found.")
            self.device_present = False
    
    def is_device_connected(self):
        return self.device_present
    
    def get_device_pos(self):
        #Get pantograph joint positions. Only works if a device is connected!
        if self.device_present and self.port and self.haplyBoard.data_available():    ##If Haply is present
            #get device angles
            self.device.device_read_data()
            motorAngle = self.device.get_device_angles()
            
            #forward kinematics to get position
            device_position = self.device.get_device_position(motorAngle)
        else:
            print("debug vals:",self.device_present,self.port,self.haplyBoard.data_available())
            raise ValueError("[PHYSICS] Cannot get device position if no device is connected!")
        #get other device positions
        pA0 = (0.0,0.0)
        pB0 = (self.d,0.0)
        a1 = math.radians(motorAngle[0])
        a2 = math.radians(motorAngle[1])
        pA = ( self.l1*math.cos(a1),self.l1*math.sin(a1) )
        pB = ( self.l1*math.cos(a2)+self.d, self.l1*math.sin(a2) )
        return pA0,pB0,pA,pB,device_position
    
    def update_force(self,f):
        #Send forces to the device. Only works if a device is connected!
        if self.device_present and self.port:
            #update and send torques
            f[1] = -f[1] #graphical y axis is reversed
            self.device.set_device_torques( f ) #forces in cartesian coordinates. Calculates the needed motor torques.
            self.device.device_write_torques()
            time.sleep(0.001) #pause for 1 millisecond
        elif not self.device_present:
            print("debug vals:",self.device_present,self.port)
            raise ValueError("[PHYSICS] Cannot set device force if no device is connected!")
        
    def serial_ports(self):
        #Detect and Connect Physical device
        """ Lists serial port names """
        ports = list(serial.tools.list_ports.comports())
        result = []
        for p in ports:
            try:
                port = p.device
                s = serial.Serial(port)
                s.close()
                if p.description[0:12] == "Arduino Zero":
                    result.append(port)
                    print(p.description[0:12])
            except (OSError, serial.SerialException):
                pass
        return result
        
    def derive_device_pos(self,pe,recursive_call=0):
        #given the endpoint location pe, find the locations of the intermediate points
        #pe: endpoint location
        pA0 = (0.0,0.0) #pA: origin location
        pB0 = (pA0[0]+self.d,pA0[1]) #pB is assumed to be at pA_x+d !!!!
        dA0 = math.sqrt( (pe[0]-pA0[0])**2+(pe[1]-pA0[1])**2 ) #distance from point A0 to the endpoint
        dB0 = math.sqrt( (pe[0]-pB0[0])**2+(pe[1]-pB0[1])**2 ) #distance from point B0 to the endpoint
        uVA0 = ( (pe[0]-pA0[0])/dA0, (pe[1]-pA0[1])/dA0 ) #unit vector from A0 to the endpoint
        uVB0 = ( (pe[0]-pB0[0])/dB0, (pe[1]-pB0[1])/dB0 ) #unit vector from B0 to the endpoint
        #check for invalid positions
        distance_margin = 0.0005 #m
        max_arm_length = self.l1+self.l2-distance_margin
        min_dist = self.l2-self.l1+distance_margin
        if dA0>max_arm_length or dB0>max_arm_length: #pantograph overextended
            #for this limited setting, being outside the reach of the pantograph would always mean at least one arm has two segments that are colinear
            #thus find the base point with the longest distance, and imagine a line from the given pE to that base point
            #at the distance l1+l2 along this line from the base point is the maximum extension of the pantograph
            #(subtract a little bit of distance to accomodate for floating point and pixel errors)
            if dA0>dB0:
                pe = ( pA0[0]+uVA0[0]*max_arm_length, pA0[1]+uVA0[1]*max_arm_length )
            else:
                pe = ( pB0[0]+uVB0[0]*max_arm_length, pB0[1]+uVB0[1]*max_arm_length )
        elif pe[1]<pA0[1]+min_dist: #pantograph too close to the base
            #the endpoint is so close to base joints that the inverse kinematics starts having issues.
            #limit motion by simply restricting y
            pe[1] = pA0[1]+min_dist
        
        #find valid angles
        try:
            dA0 = math.sqrt( (pe[0]-pA0[0])**2+(pe[1]-pA0[1])**2 ) #distance from point A0 to the endpoint
            theta_dA0 = math.atan2(pe[1]-pA0[1],pe[0]-pA0[0]) #angle to the line connecting point A to the endpoint
            theta_cA = math.acos( (self.l1**2+dA0**2-self.l2**2)/(2*self.l1*dA0) )
            theta_A0 = theta_dA0 + theta_cA

            dB0 = math.sqrt( (pe[0]-pB0[0])**2+(pe[1]-pB0[1])**2 ) #distance from point B0 to the endpoint
            theta_dB0 = math.atan2(pe[1]-pB0[1],pe[0]-pB0[0]) #angle to the line connecting point B to the endpoint
            theta_cB = math.acos( (self.l1**2+dB0**2-self.l2**2)/(2*self.l1*dB0) )        
            theta_B0 = theta_dB0 - theta_cB
        except Exception as e:
            theta_A0 = 0.0
            theta_B0 = 0.0
            print("[Physics] Unclassified pantograph domain error")
        
        pA = ( self.l1*math.cos(theta_A0)+pA0[0],self.l1*math.sin(theta_A0)+pA0[1] ) #intermediate point A
        pB = ( self.l1*math.cos(theta_B0)+pB0[0],self.l1*math.sin(theta_B0)+pB0[1] ) #intermediate point B
        
        #pA0,pB0,pA,pB,pE
        return pA0,pB0,pA,pB,pe
    
    def close(self):
        if self.device_present and self.port:
            #reset the force to 0, otherwise it will stay nonzero
            self.device.set_device_torques( [0,0] )
            self.device.device_write_torques()
            time.sleep(0.001) #pause for 1 millisecond