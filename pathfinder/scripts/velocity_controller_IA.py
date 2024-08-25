#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray, Int16MultiArray

import numpy as np

from geometry_msgs.msg import Twist

from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

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

        self.subscription = self.create_subscription(
            Int16MultiArray,
            f'IA_actions',
            self.listener_callback,
            10
        )
        self.subscription

        # Create a publisher for '/output_topic'
        self.publisher_cmd_vel = self.create_publisher(Twist, f'cmd_vel', 10)
        self.publisher_control_state = self.create_publisher(Float32MultiArray, f'controller_state', qos_profile)
        
        new_msg = Float32MultiArray()
        new_msg.data = self.vel_state
        self.publisher_control_state.publish(new_msg)

    def listener_callback(self, msg):
        ##self.get_logger().info(f'Received message: "{len(point_cloud.data)}"')
        actions = np.array(msg.data)

        if actions[0] != -2:

            self.vel_state = np.clip(self.vel_state + ((actions)*(self.max_speed/self.max_steps)),-self.max_speed,self.max_speed)

            msg = Twist()
            msg.linear.x = self.vel_state[0]  # Set desired linear velocity (e.g., 0.5 m/s)
            msg.angular.z = self.vel_state[1]  # Set desired angular velocity (e.g., 0.1 rad/s)
            self.publisher_cmd_vel.publish(msg)

            new_msg = Float32MultiArray()
            new_msg.data = self.vel_state
            self.publisher_control_state.publish(new_msg)
        else:
            msg = Twist()
            msg.linear.x = 0.0  # Set desired linear velocity (e.g., 0.5 m/s)
            msg.angular.z = 0.0  # Set desired angular velocity (e.g., 0.1 rad/s)
            self.publisher_cmd_vel.publish(msg)

if __name__ == '__main__':
    rclpy.init()
    controller_nodes = ControllerNodes()
    try:
        rclpy.spin(controller_nodes)
    except rclpy.executors.ExternalShutdownException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        controller_nodes.destroy_node()




