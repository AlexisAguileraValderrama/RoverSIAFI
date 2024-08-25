#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray, Int16MultiArray

import numpy as np

from geometry_msgs.msg import Twist

from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

import cv2

import time

class ControllerNodes(Node):
    def __init__(self):

        self.vel_state = np.array([0,0])

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )
        
        super(ControllerNodes,self).__init__(f'velocity_controller')

        self.max_speed = 0.5
        self.max_steps = 4

        self.declare_parameter('namespace', '')
        namespace = self.get_parameter('namespace').get_parameter_value().string_value

        # Create a publisher for '/output_topic'
        self.publisher_cmd_vel = self.create_publisher(Twist, f'{namespace}/cmd_vel', 10)
        self.publisher_control_state = self.create_publisher(Float32MultiArray, f'{namespace}/controller_state', qos_profile)

    def ReadControl(self):
        
        k = cv2.waitKey(1)

        if k == ord('w'):
            self.actions = [1,0]
        elif k == ord('s'):
            self.actions = [-1,0]
        elif k == ord('a'):
            self.actions = [0,1]
        elif k == ord('d'):
            self.actions = [0,-1]
        else: self.actions = [0,0]

        self.actions = np.array(self.actions)

    def listener_callback(self):

        while True:

            self.ReadControl()

            self.vel_state = np.clip(self.vel_state + ((self.actions/self.max_steps)*(self.max_speed)),-self.max_speed,self.max_speed)

            msg = Twist()
            msg.linear.x = self.vel_state[0]  # Set desired linear velocity (e.g., 0.5 m/s)
            msg.angular.z = self.vel_state[1]  # Set desired angular velocity (e.g., 0.1 rad/s)
            self.publisher_cmd_vel.publish(msg)

            new_msg = Float32MultiArray()
            new_msg.data = self.vel_state
            self.publisher_control_state.publish(new_msg)
            print(f"vel array {self.vel_state}")
            time.sleep(0.1)

if __name__ == '__main__':
    image = cv2.imread('/home/serapf/Downloads/miku.jpg')
    
    
    # using cv2.imshow() to display the image
    cv2.imshow('Display', image)

    rclpy.init()
    controller_nodes = ControllerNodes()
    controller_nodes.listener_callback()
    




