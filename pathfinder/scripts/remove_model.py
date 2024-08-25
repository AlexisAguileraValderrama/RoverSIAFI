#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import DeleteEntity

class ModelRemover(Node):
    def __init__(self):
        super(ModelRemover,self).__init__('model_remover')
        self.client = self.create_client(DeleteEntity, 'delete_entity')
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        
        self.declare_parameter('model_name', 'default_argument')
        model_name = self.get_parameter('model_name').get_parameter_value().string_value
        print(model_name)

        self.request = DeleteEntity.Request()
        self.request.name = model_name  # Replace with the name of your model

    def send_request(self):
        success = False
        retry_count = 0
        max_retries = 5

        while not success and retry_count < max_retries:
            future = self.client.call_async(self.request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=3)
            if future.result() is not None:
                success = future.result().success
                self.get_logger().info('Response: %r' % future.result().success)
            else:
                self.get_logger().error('Service call failed %r' % future.exception())

            if not success:
                retry_count += 1
                self.get_logger().info(f'Retry {retry_count}/{max_retries}')

        if not success:
            self.get_logger().error('Failed to set entity state after several attempts')

def main(args=None):
    rclpy.init(args=args)
    model_remover = ModelRemover()
    model_remover.send_request()

if __name__ == '__main__':
    main()
