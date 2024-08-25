import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

import xacro

from launch.actions import IncludeLaunchDescription

def generate_launch_description():

    # Codigo para inicializar Gazebo con el mundo empty_world.py
    
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')
    pkg_custom_world = get_package_share_directory('pathfinder')

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py'),
        ),
        launch_arguments={'world': os.path.join(pkg_custom_world, 'worlds', 'empty_world.world')}.items()  # Replace 'your_world_file.world' with your custom world file
    )

    launch_description = LaunchDescription([
        DeclareLaunchArgument(
          'use_sim_time',
          default_value='true',
          description='Use simulation/Gazebo clock'),
        gazebo,
    ])

    return launch_description