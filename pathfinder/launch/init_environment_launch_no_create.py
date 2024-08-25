import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from launch.actions import RegisterEventHandler, ExecuteProcess
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration

import xacro

from launch.actions import IncludeLaunchDescription, GroupAction
from launch.substitutions import LaunchConfiguration, TextSubstitution

def generate_launch_description():
    # Define arguments
    
    use_sim_time = LaunchConfiguration('use_sim_time')

    launch_description = LaunchDescription([
        DeclareLaunchArgument(
            'world_name',default_value='default_value1',description='A parameter for the node'
        ),
        DeclareLaunchArgument(
            'vox_size',default_value='default_value1',description='A parameter for the node'
        ),
        DeclareLaunchArgument(
            'x_loc',default_value='default_value2',description='Another parameter for the node'
        ),
        DeclareLaunchArgument(
            'y_loc',default_value='default_value2',description='Another parameter for the node'
        ),
        DeclareLaunchArgument(
            'z_loc',default_value='default_value2',description='Another parameter for the node'
        ),
        DeclareLaunchArgument(
            'namespace',default_value='default_value2',description='Another parameter for the node'
        ),
        DeclareLaunchArgument(
            'octomap_bt_path',default_value='default_value2',description='Another parameter for the node'
        ),
        DeclareLaunchArgument(
            'octomap_frame_id',default_value='default_value2',description='Another parameter for the node'
        ),
        DeclareLaunchArgument(
            'depth_image_dim',default_value='default_value2',description='Another parameter for the node'
        ),
        DeclareLaunchArgument(
            'map_size',default_value='default_value2',description='Another parameter for the node'
        ),
    ])

    pkg_pathfinder = get_package_share_directory('pathfinder')

    load_world_node = Node(
            package='pathfinder',  # Replace with your package name
            executable='load_world_gazebo.py',      # Replace with your executable name
            name='load_world_gazebo',
            #output='screen',
            parameters=[
                {'world_name': LaunchConfiguration('world_name')},
                {'world_path': os.path.join(pkg_pathfinder, 'gen_worlds_model')},
                {'x': LaunchConfiguration('x_loc')},
                {'y': LaunchConfiguration('y_loc')},
                {'z': LaunchConfiguration('z_loc')}
            ]
        )
    
    octomap_world = Node(
            package='octomap_server',  # Replace with your package name
            executable='octomap_server_node',      # Replace with your executable name
            name='load_octomap_world',
            namespace=LaunchConfiguration('namespace'),
            #output='screen',
            parameters=[
                {'octomap_path': LaunchConfiguration('octomap_bt_path')},
                {'frame_id:': LaunchConfiguration('octomap_frame_id')},
                # {'x': str(x_loc)},
                # {'y': str(y_loc)},
                # {'z': str(z_loc)}
            ]
        )
    
    flatten_map = Node(
            package='pathfinder',  # Replace with your package name
            executable='get_map_flatten.py',      # Replace with your executable name
            name='get_map_flatten',
            #output='screen',
            parameters=[
                {'namespace': LaunchConfiguration('namespace')},
                {'map_size': LaunchConfiguration('map_size')},
                {'vox_size': LaunchConfiguration('vox_size')},
                {'depth_image_dim': LaunchConfiguration('depth_image_dim')},
                {'x_loc': LaunchConfiguration('x_loc')},
                {'y_loc': LaunchConfiguration('y_loc')},
                {'z_loc': LaunchConfiguration('z_loc')}
            ]
        )

    # Get the URDF xacro file path
    # xacro_file = xacro.process_file(os.path.join(pkg_pathfinder, 'description/', 'robot.urdf.xacro'))
    # urdf_path = os.path.join(pkg_pathfinder, 'description/', 'full_pathfinder.urdf')

    # urdf_file = open(urdf_path, "w")
    # urdf_file.write(xacro_file.toxml())
    # urdf_file.close()    

    # group_cmds = GroupAction([
    #     IncludeLaunchDescription(
    #         PythonLaunchDescriptionSource(os.path.join(pkg_pathfinder, 'launch', 'generic_spawn_launch.py')),
    #         launch_arguments={
    #                         'use_sim_time': use_sim_time,
    #                         'robot_name': robot_name,
    #                         'robot_namespace': robot_name,
    #                         'tf_remapping': '/'+robot_name+'/tf',
    #                         'static_tf_remap': '/'+robot_name+'/tf_static',
    #                         'scan_topic': '/'+robot_name+'/scan',
    #                         'map_topic': '/'+robot_name+'/map',
    #                         'odom_topic': '/'+robot_name+'/odom',
    #                         'urdf': open(urdf_path).read(),
    #                         'urdf_path': urdf_path,
    #                         'x': LaunchConfiguration('x_loc'),
    #                         'y': LaunchConfiguration('y_loc'),
    #                         'z': LaunchConfiguration('z_loc')
    #                         }.items()), ])

    event_handler = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=load_world_node,
            on_exit=[octomap_world, flatten_map]
        )
    )

    launch_description.add_action(load_world_node)
    launch_description.add_action(event_handler)

    return launch_description