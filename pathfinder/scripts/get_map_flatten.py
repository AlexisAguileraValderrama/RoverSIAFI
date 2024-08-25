#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt

import sensor_msgs_py.point_cloud2 as pc2

import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F

from std_msgs.msg import Float32MultiArray
from operator import itemgetter

from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

from parser_tuple import parse_tuple
from scipy import ndimage
import cv2

import rclpy.executors

import time

import threading

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Define a 2D convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, stride=1, padding=1)
        # Define a max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Define another 2D convolutional layer
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=10, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        # Apply the first convolutional layer followed by ReLU activation
        x = self.conv1(x)
        x = torch.relu(x)
        # Apply max pooling
        x = self.pool(x)
        # Apply the second convolutional layer followed by ReLU activation
        x = self.conv2(x)
        x = torch.relu(x)
        # Apply max pooling
        x = self.pool(x)
        return x

class ExampleSubscriber(Node):
    def __init__(self):
        super(ExampleSubscriber,self).__init__(f'flatten_map_node')

        self.get_logger().info(f'creando flatten map node')

        self.declare_parameter('map_size', '10,10,1')
        print(self.get_parameter('map_size').get_parameter_value().string_value)
        self.map_size = parse_tuple(self.get_parameter('map_size').get_parameter_value().string_value)

        self.declare_parameter('vox_size', 200)
        self.vox_size = self.get_parameter('vox_size').get_parameter_value().integer_value

        self.declare_parameter('depth_image_dim', '100,100')
        self.depth_image_dim = parse_tuple(self.get_parameter('depth_image_dim').get_parameter_value().string_value)

        self.declare_parameter('x_loc', 0.0)
        self.x_loc = self.get_parameter('x_loc').get_parameter_value().double_value

        self.declare_parameter('y_loc', 0.0)
        self.y_loc = self.get_parameter('y_loc').get_parameter_value().double_value

        self.declare_parameter('z_loc', 0.0)
        self.z_loc = self.get_parameter('z_loc').get_parameter_value().double_value

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.subscription = self.create_subscription(
            PointCloud2,
            f'octomap_point_cloud_centers',
            self.listener_callback,
            qos_profile
        )
        self.subscription  # prevent unused variable warning

        # Create a publisher for '/output_topic'
        self.publisher_flatten_map = self.create_publisher(Float32MultiArray, f'flatten_map', qos_profile)
        self.publisher_spawn_point = self.create_publisher(Float32MultiArray, f'spawn_point', qos_profile)
        self.publisher_goal_point = self.create_publisher(Float32MultiArray, f'goal_point', qos_profile)

        self.get_logger().info(f'creado ')

        self.scaled_image_3d = None

    def run_window(self):
        cv2.namedWindow(f"{self.get_namespace()}_map",cv2.WINDOW_NORMAL)
        while True:
            if self.scaled_image_3d is not None:
                imS = cv2.resize(self.scaled_image_3d, (300, 300))   
                cv2.imshow(f"{self.get_namespace()}_map", imS)
                cv2.waitKey(1000)
            else:
                time.sleep(1)



    def listener_callback(self, point_cloud):
        self.get_logger().info(f'Received message: "{len(point_cloud.data)}"')

        points_list = []
        for point in pc2.read_points(point_cloud, skip_nans=False):
            points_list.append([point[0], point[1], point[2]])

        points_list = sorted(points_list, key=itemgetter(2))
        
        zs = [point[2] for point in points_list]

        print(max(zs))
        print(min(zs))

        mesh_size = self.map_size[0]
        res = mesh_size / self.vox_size

        dimension = int(mesh_size/res)

        m = 1 / res
        b = mesh_size / (2*res)

        # Populate the occupancy grid ######################################################
        depth_image = numpy.zeros(shape=(dimension,dimension))

        for point in points_list:
            index_x = int((m*point[0] + b))
            index_y = int((m*point[1] + b))
            depth_image[index_x][index_y] = point[2]

        # plt.imshow(depth_image)
        # plt.show()

        norm_depth_image = numpy.clip(depth_image/(self.map_size[2]/2),-1,1)

        # plt.imshow(norm_depth_image)
        # plt.show()

        self.scaled_image_3d = cv2.resize(norm_depth_image, dsize=self.depth_image_dim, interpolation=cv2.INTER_NEAREST)

        # plt.imshow(self.scaled_image_3d)
        # plt.show()

        ##########################2D Convolution###################################3

        # Convert to PyTorch tensor
        tensor = torch.from_numpy(self.scaled_image_3d)
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch dimension

        tensor = tensor.to(torch.float64)

        # print(tensor.shape)

        # # Initialize the feature extractor
        # feature_extractor = SimpleCNN()

        # # Pass your data through the model
        # with torch.no_grad():  # No need to compute gradients
        #     features = feature_extractor(tensor.float())

        # print(features.shape)  # Check the shape of the extracted features

        flatten = torch.flatten(tensor).cpu().detach().numpy()
        # print(flatten.shape)
        new_msg = Float32MultiArray()
        new_msg.data = flatten
        self.get_logger().info(f'Received message: {flatten}')
        self.get_logger().info(f'Received message: {max(flatten)}')
        self.publisher_flatten_map.publish(new_msg)

        ###################Get spawn and goal point##########################

        flat_areas = flat_map(norm_depth_image,0.01)
        kernel = numpy.ones((6, 6), numpy.uint8)
        err = cv2.erode(flat_areas.astype(numpy.uint8), kernel,iterations=2)
        
        coordinates = numpy.column_stack(numpy.where(err == 1))
        
        random_index = numpy.random.randint(0, len(coordinates))
        random_pixel_coordinates = tuple(coordinates[random_index])

        spawn_point = numpy.array([
                       (random_pixel_coordinates[0] - b)/m,
                       (random_pixel_coordinates[1] - b)/m,
                       depth_image[random_pixel_coordinates]+0.15],
                       dtype=numpy.float16)

        dist = 0
        while dist < self.map_size[0]*0.05:
            random_index = numpy.random.randint(0, len(coordinates))
            random_pixel_coordinates = tuple(coordinates[random_index])

            goal_point = numpy.array([
                    (random_pixel_coordinates[0] - b)/m,
                    (random_pixel_coordinates[1] - b)/m,
                    depth_image[random_pixel_coordinates] + 0.15],
                    dtype=numpy.float16)

            dist = numpy.linalg.norm(goal_point-spawn_point)

        print(goal_point)

        # plt.imshow(err)
        # plt.show()

        print(spawn_point)

        new_msg = Float32MultiArray()
        new_msg.data = spawn_point
        self.publisher_spawn_point.publish(new_msg)

        new_msg = Float32MultiArray()
        new_msg.data = goal_point
        self.publisher_goal_point.publish(new_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ExampleSubscriber()

    # Create a thread
    thread = threading.Thread(target=node.run_window)
    # Start the thread
    thread.start()

    try:
        rclpy.spin(node)
    except rclpy.executors.ExternalShutdownException:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()

# Calculate the gradient (rate of change in height)
def flat_map(height_map, flatness_threshold):

    gradient_x = numpy.gradient(height_map, axis=0)
    gradient_y = numpy.gradient(height_map, axis=1)

    # Calculate the magnitude of the gradient
    gradient_magnitude = numpy.sqrt(gradient_x**2 + gradient_y**2)

    # Detect flat areas
    flat_areas = gradient_magnitude < flatness_threshold

    return flat_areas

if __name__ == '__main__':
    main()
