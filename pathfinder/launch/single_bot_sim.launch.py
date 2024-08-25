import os
import xacro

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnExecutionComplete
from launch_ros.actions import Node

from launch.actions import ExecuteProcess


def generate_launch_description():

    pkg_pathfinder = get_package_share_directory('pathfinder')

    # Process the URDF xacro file
    xacro_file = os.path.join(pkg_pathfinder,'description','robot.urdf.xacro')
    urdf = xacro.process_file(xacro_file)

    rviz_config_file = os.path.join(pkg_pathfinder,'config','view_octomap_complete.rviz')
    octomap_map_file = os.path.join(pkg_pathfinder,'gen_worlds_model','world.bt')
    
    #Include the robot_state_publisher to publish transforms
    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': urdf.toxml()}]
    )

    # Include the Gazebo launch file, provided by the gazebo_ros package
    world_file_path = os.path.join(pkg_pathfinder, 'worlds', 'empty_world.world') 
    gazebo = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
                    launch_arguments={'world': world_file_path}.items()
    )

    # octomap_map = Node(
    #         package='octomap_server',
    #         executable='octomap_server_node',
    #         name='octomap_server_node',
    #         output='screen',
    #         parameters=[
    #             {'octomap_path': octomap_map_file},
    #             {'frame_id': 'odom'},
    #         ],
    #     )
    
    # flatten_map = Node(
    #         package='pathfinder',
    #         executable='get_map_flatten.py',
    #         name='map_flatten_node',
    #         output='screen',
    #     )

    # Run the spawner node from the gazebo_ros package to spawn the robot in the simulation
    spawn_entity = Node(
                package='gazebo_ros', 
                executable='spawn_entity.py',
                output='screen',
                arguments=['-topic', 'robot_description',   # The the robot description is published by the rsp node on the /robot_description topic
                           '-entity', 'pathfinder',
                           '-z', '1'],        # The name of the entity to spawn (doesn't matter if you only have one robot)
    )

    rviz =  ExecuteProcess(
            cmd=['rviz2', '-d', rviz_config_file],
            output='screen'
        )

    # Launch everything!
    return LaunchDescription([
        DeclareLaunchArgument(
          'use_sim_time',
          default_value='false',
          description='Use simulation/Gazebo clock'
        ),
        rsp,
        rviz,
        gazebo,
        spawn_entity,
        # octomap_map,
        # flatten_map
    ])
