#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node

import subprocess
from ament_index_python.packages import get_package_share_directory

class ExampleSubscriber(Node):
    def __init__(self):
        super(ExampleSubscriber, self).__init__('run_script_node')
        self.get_logger().info('Node has been started.')

        self.declare_parameter('world_name', 'default_argument')
        world_name = self.get_parameter('world_name').get_parameter_value().string_value

        self.declare_parameter('script_path', './')
        script_path = self.get_parameter('script_path').get_parameter_value().string_value

        self.declare_parameter('vox_size', './')
        vox_size = self.get_parameter('vox_size').get_parameter_value().string_value

        self.get_logger().info(f'Script_path argument: {script_path}')
        self.get_logger().info(f'world_name argument: {world_name}')
        self.get_logger().info(f'vox_size argument: {vox_size}')

        while True:
            try:
                result = subprocess.run(
                    [script_path, world_name, vox_size], 
                    check=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=5,errors='replace'
                )
                print("Script Output:\n", result.stdout)
                print("Script Errors:\n", result.stderr)
                break  # Exit the loop if the script runs successfully
            except subprocess.TimeoutExpired:
                print(f"Script exceeded the timeout of {5} seconds. Retrying...")
            except subprocess.CalledProcessError as e:
                print("Error occurred while running the script:", e)
                break  # Exit the loop if there is a non-timeout error
            
    def listener_callback(self):
        self.get_logger().info(f'Received message: ')

def main(args=None):
    rclpy.init(args=args)
    node = ExampleSubscriber()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
