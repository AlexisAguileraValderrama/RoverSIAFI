from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.registration import EnvSpec

from stable_baselines3.common.type_aliases import GymStepReturn

from std_msgs.msg import Int16MultiArray, Float32MultiArray, String

from sensor_msgs.msg import  JointState
from geometry_msgs.msg import PoseArray, Point
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import EntityState

import time

import datetime
from numpy.linalg import norm

import pickle

import rclpy
from rclpy.node import Node

import os
from ament_index_python.packages import get_package_share_directory

from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy

import threading
import subprocess
import signal

import xacro

from scipy.spatial.transform import Rotation as R


class DRL_SubsPub(Node):

    def __init__(self, name_space):
        super(DRL_SubsPub,self).__init__(f'{name_space}_IA_node')

        self.flatten_map = None
        self.goal_point = None
        self.robot_odom = None
        self.spawn_point = None
        self.controller_state = None

        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.sub_flatten_map = self.create_subscription(
            Float32MultiArray, 
            f'{name_space}/flatten_map',
            self.listener_flatten_map,
            qos_profile)
        self.sub_flatten_map  # prevent unused variable warning

        self.sub_goal_point = self.create_subscription(
            Float32MultiArray, 
            f'{name_space}/goal_point',
            self.listener_goal_point,
            qos_profile)
        self.sub_goal_point  # prevent unused variable warning

        self.sub_goal_point = self.create_subscription(
            Float32MultiArray, 
            f'{name_space}/spawn_point',
            self.listener_spawn_point,
            qos_profile)
        self.sub_goal_point  # prevent unused variable warning

        self.sub_odometry = self.create_subscription(
            Odometry, 
            f'{name_space}/odom',
            self.listener_odometry,
            10)
        self.sub_odometry  # prevent unused variable warning

        self.sub_controller_state = self.create_subscription(
            Float32MultiArray, 
            f'{name_space}/controller_state',
            self.listener_controller_node,
            qos_profile)
        self.sub_controller_state  # prevent unused variable warning

        # Create a publisher for '/output_topic'
        self.publisher_IA_actions = self.create_publisher(Int16MultiArray, f'{name_space}/IA_actions',10)
        self.publisher_fast_relocate = self.create_publisher(EntityState, f'/gazebo/mgp/fast_relocate',10)
        self.publisher_fast_remove = self.create_publisher(String, f'/gazebo/mgp/fast_remove',10)
    
    def reset_buffers(self):
        self.flatten_map = None
        self.goal_point = None
        self.spawn_point = None
        # self.robot_odom = None
        # self.controller_state = None

    def listener_flatten_map(self, msg):
        self.flatten_map = msg.data
    
    def listener_goal_point(self, msg):
        self.goal_point = msg.data

    def listener_spawn_point(self, msg):
        self.spawn_point = msg.data

    def listener_controller_node(self, msg):
        self.controller_state = msg.data
    
    def listener_odometry(self, msg):
        self.robot_odom = msg
        

class RoverNavEnvMulti(Env):

    spec = EnvSpec("RoverNavEnvMulti-v0", "no-entry-point")
    state: np.ndarray

    def is_z_axis_flipped(self,quaternion, threshold):
        q_x, q_y, q_z, q_w = quaternion
        
        # Calculate the z-component of the transformed z-axis
        z_component = q_w**2 - q_x**2 - q_y**2 + q_z**2
        
        # Check if the z-component is negative
        return z_component <= threshold, z_component

    def status_publisher_function(self):
        while not self.detener:
            msg_ia_status = Int16MultiArray()
            msg_ia_status.data = [self.ia_status]
            self.pub_ia_status.publish(msg_ia_status)
            time.sleep(0.005)

    def recieve_odom(self,data):
      if self.in_step == False:
        self.cube_info = data

    def __init__( 
            self,
            brain_name = '',
            multi_num = None,
            continous_space = True,
            options = None
    ):
        super().__init__()
        self.multi_num = multi_num
        self.continous_space = continous_space
        self.options = options

        self.options.map_size = np.array(self.options.map_size)
        self.options.grid_loc = np.array(self.options.grid_loc)

        self.detener = False

        self.timer = 0
        self.time_limit = 15

        self.succes_count = 0
        self.failed_count = 0

        #Wuachar esto XD
        self.accel_time = 4

        self.world_count = 0

        self.step_count = 0
        self.step_count_total = 0

        self.in_step = False
        self.total_reward = 0
        self.last_reward = 0

        self.ia_status = 0

        self.last_time = 0

        rclpy.init()
        self.IA_nodes = DRL_SubsPub(name_space=f'pathfinder_{self.options.agente}')

        # # # Run ROS 2 spin in a separate thread
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.IA_nodes,))
        self.spin_thread.start()

        # self.status_thread = Thread(target=self.status_publisher_function)
        # self.status_thread.start()

        self.ros_process_buffer = {}

        self.pkg_pathfinder = get_package_share_directory('pathfinder')

        print("estamos aca\n")

        self.inif_environment()

        print("aLGO PASO \n")

        #############################################################################

        dim_odom = 3 + 4 + 3 + 3 #3 pose 3 orientation 3 vel 3 ang vel
        dim_goal_point = 3 # 3 coordinates
        dim_map = 0#len(self.IA_nodes.flatten_map)
        dim_controller_state = 2
        dim_distance = 1
        dim_timer = 1
        self.total_dim_input = dim_odom + dim_goal_point + dim_map + dim_controller_state + dim_distance + dim_timer
        self.IA_nodes.get_logger().info(f'Input dim: {self.total_dim_input}')

        #Actions
        if continous_space:
            self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(2,), dtype=np.float32)
        else:
           self.action_space = spaces.MultiDiscrete([2,2])

        self.observation_space = spaces.Box(
                            low=-2,
                            high=2,
                            shape=(self.total_dim_input,),
                            dtype=np.float32,
                        )
        
    def kill_all_ros_process(self):
        print("Killing all process")
        for name, process in self.ros_process_buffer.items():
            if process is not None:
                print(f"Killing process {name}")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait()

    def seed(self, seed: int) -> None:
        self.obs_space.seed(seed)

    def _get_obs(self):

        #############Odom#################### absolute

        self.robot_coor_abs = np.array([self.IA_nodes.robot_odom.pose.pose.position.x,
                            self.IA_nodes.robot_odom.pose.pose.position.y,
                            self.IA_nodes.robot_odom.pose.pose.position.z])

        self.robot_coor_rel = (self.robot_coor_abs - np.array(self.options.grid_loc))/(np.array(self.options.map_size)/2)
        
        self.robot_orien = np.array([self.IA_nodes.robot_odom.pose.pose.orientation.x,
                            self.IA_nodes.robot_odom.pose.pose.orientation.y,
                            self.IA_nodes.robot_odom.pose.pose.orientation.z,
                            self.IA_nodes.robot_odom.pose.pose.orientation.w])
        
        self.robot_vel = np.array([self.IA_nodes.robot_odom.twist.twist.linear.x,
                            self.IA_nodes.robot_odom.twist.twist.linear.y,
                            self.IA_nodes.robot_odom.twist.twist.linear.z])
        
        self.robot_ang_vel = np.array([self.IA_nodes.robot_odom.twist.twist.angular.x,
                    self.IA_nodes.robot_odom.twist.twist.angular.y,
                    self.IA_nodes.robot_odom.twist.twist.angular.z])

        #############Goal point############## relative

        #self.goal_point = np.array(self.IA_nodes.goal_point)/(np.array(self.options.map_size)/2)

        #############build observation####################

        dist = np.linalg.norm(self.goal_point-self.robot_coor_rel)
        
        self.current_fact_time = self.timer / self.time_limit

        observation = np.concatenate((np.array([]),#self.IA_nodes.flatten_map),
                                      self.goal_point,
                                    self.robot_coor_rel, #3
                                    self.robot_orien,   #4
                                    self.robot_vel,     #3
                                    self.robot_ang_vel, #3
                                    dist,
                                    self.current_fact_time,
                                    np.array(self.IA_nodes.controller_state)), axis=None) #2
        
        # print("-----------------------------------------------")
        # print(observation)
        # print("-----------------------------------------------")

        return observation

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Union[float, np.ndarray]], Dict]:
        
        self.in_step = True
        if seed is not None:
            self.obs_space.seed(seed)

        ################Reacomodar mapas e iniciar nuevo octomap#################

        # os.killpg(os.getpgid(self.octomap_process.pid), signal.SIGTERM)
        # self.octomap_process.wait()

        # move_world_process = self.relocate_model(f'world_{str(self.options.agente)}_{self.world_count}',
        #                                 [self.options.grid_loc[0],
        #                                 self.options.grid_loc[1],
        #                                 self.options.grid_loc[2]-4],
        #                                 [0,0,0,1])
        # move_world_process.wait()

        # move_world_process = self.relocate_model(f'world_{str(self.options.agente)}_{self.world_count+1}',
        #                                 [self.options.grid_loc[0],
        #                                 self.options.grid_loc[1],
        #                                 self.options.grid_loc[2]],
        #                                 [0,0,0,1])
        # move_world_process.wait()

        # self.world_count += 1

        # self.IA_nodes.reset_buffers()

        # bt_path = os.path.join(self.pkg_pathfinder, 'gen_worlds_model', 'bt',f'world_{self.options.agente}_{self.world_count}.bt')
        # self.octomap_process = self.run_node(package='octomap_server',
        #                                      script_name='octomap_server_node',
        #                                      node_name='octomap_server_node',
        #                                      namespace=f'/pathfinder_{self.options.agente}',
        #                                      run_arguments=[
        #                                                     f'octomap_path:={bt_path}',
        #                                                     f'frame_id:=frame_id',
        #                                                 ])
        
        ################Reacomodar robot##############

        # print("Getting spawn_point nodes")
        # while self.IA_nodes.spawn_point is None:
        #     print("Waiting for spawn_points")
        #     time.sleep(0.5)

        self.IA_nodes.spawn_point = np.random.uniform(low=-(self.options.map_size[0]-1.2)/2, high=(self.options.map_size[0]-1.2)/2, size=(3,))
        self.IA_nodes.spawn_point[2] = 0
        self.goal_point = np.random.uniform(low=-0.8, high=0.8, size=(3,))
        self.goal_point[2] = 0.2

        # self.robot_process = self.relocate_model('pathfinder_'+str(self.options.agente),
        #                                 [self.IA_nodes.spawn_point[0] + self.options.grid_loc[0],
        #                                 self.IA_nodes.spawn_point[1] + self.options.grid_loc[1],
        #                                 self.IA_nodes.spawn_point[2] + self.options.grid_loc[2]],
        #                                 [0,0,0,1])
        # self.robot_process.wait()
        self.fast_relocate_model(f'pathfinder_{str(self.options.agente)}',
                                            [self.IA_nodes.spawn_point[0] + self.options.grid_loc[0],
                                            self.IA_nodes.spawn_point[1] + self.options.grid_loc[1],
                                            self.IA_nodes.spawn_point[2] + self.options.grid_loc[2]],
                                            [0,0,0,1],3,0.1)
        self.fast_relocate_model(f'marker_{self.options.agente}',
                    self.goal_point*(self.options.map_size/2) + self.options.grid_loc,
                    [0,0,0,1],3,0.1)
        # posm = self.goal_point*(self.options.map_size/2) + self.options.grid_loc
        # posm[2] = 0.5
        # self.relocate_model(f'marker_{self.options.agente}',
        #             posm,
        #             [0,0,0,1]).wait()

        ################################################
        ##################Elimina el viejo y crea el nuevo (sin detenerse)

        # self.remove_world = self.remove_model(f'world_{str(self.options.agente)}_{self.world_count-1}')
        # #self.remove_world.wait()

        # self.IA_nodes.get_logger().info(f'Mapa eliminado')

        # self.map_process = self.create_a_world(f"world_{self.options.agente}_{self.world_count+1}",[0,0,-2])

        ###########################

        self.step_count = 0
        self.timer = 0

        obs = self._get_obs()
        self.in_step = False

        #print(f'flat image: {len(self.IA_nodes.flatten_map)}')
        print(f'goal: {len(self.goal_point)}')
        print(f'robot_pos: {len(self.robot_coor_rel)}')
        print(f'robot_orien: {len(self.robot_orien)}')
        print(f'robit_vel: {len(self.robot_vel)}')
        print(f'robot_ang: {len(self.robot_ang_vel)}')
        print(f'robot_ang: {len(self.IA_nodes.controller_state)}')
        print(f'Total supuesto: {self.total_dim_input}')

        #input("terminando reset xDD")

        return obs,{}

        # if self.accel_time > 1.5:
        #     return obs, {}
        # else:
        #    return obs
    
    
    def step(self, action) -> GymStepReturn:
        """
        Step into the env.

        :param action:
        :return:
        """

        current_time = datetime.datetime.now()
        if self.last_time == 0:
            self.last_time = current_time

        delta_time = (current_time - self.last_time).microseconds/1000000

        self.timer = self.timer + delta_time * self.accel_time

        self.ia_status = 1
        self.in_step = True

        if self.continous_space:
           discrete_actions = np.round(action,0).astype(int)
        else:
           discrete_actions = action - [1,1]

        message = Int16MultiArray()
        message.data = discrete_actions
        self.IA_nodes.publisher_IA_actions.publish(message)
        
        
        observation = self._get_obs()
 
        reward = self.compute_reward()

        terminated = False

        dist = np.linalg.norm(self.goal_point-self.robot_coor_rel)

        if dist < 0.12:
            terminated = True
            reward = reward + 50 * (-1.25*self.current_fact_time + 1.75)
            self.succes_count += 1

        if self.timer > self.time_limit:
            print("Time out")
            terminated = True
            reward = reward - 25*dist
            self.failed_count += 1
        
        if np.prod(abs(self.robot_coor_rel[:2]) < 0.95) == 0:
            print("Out of bounds")
            message = Int16MultiArray()
            message.data = [-2,-2]
            self.IA_nodes.publisher_IA_actions.publish(message)
            # self.robot_process = self.relocate_model('pathfinder_'+str(self.options.agente),
            #                                 [self.IA_nodes.spawn_point[0] + self.options.grid_loc[0],
            #                                 self.IA_nodes.spawn_point[1] + self.options.grid_loc[1],
            #                                 self.IA_nodes.spawn_point[2] + self.options.grid_loc[2]],
            #                                 [0,0,0,1])
            # self.robot_process.wait()
            self.fast_relocate_model(f'pathfinder_{str(self.options.agente)}',
                                             [self.IA_nodes.spawn_point[0] + self.options.grid_loc[0],
                                             self.IA_nodes.spawn_point[1] + self.options.grid_loc[1],
                                             self.IA_nodes.spawn_point[2] + self.options.grid_loc[2]],
                                             [0,0,0,1],3,0.1)
            reward = reward - 5
            self.failed_count += 1

        is_flipped, flip_value = self.is_z_axis_flipped(self.robot_orien, 0.5)
        if is_flipped:
            print("Flipped")
            message = Int16MultiArray()
            message.data = [-2,-2]
            self.IA_nodes.publisher_IA_actions.publish(message)
            # self.robot_process = self.relocate_model('pathfinder_'+str(self.options.agente),
            #                                 [self.IA_nodes.spawn_point[0] + self.options.grid_loc[0],
            #                                 self.IA_nodes.spawn_point[1] + self.options.grid_loc[1],
            #                                 self.IA_nodes.spawn_point[2] + self.options.grid_loc[2]],
            #                                 [0,0,0,1])
            # self.robot_process.wait()
            self.fast_relocate_model(f'pathfinder_{str(self.options.agente)}',
                                             [self.IA_nodes.spawn_point[0] + self.options.grid_loc[0],
                                             self.IA_nodes.spawn_point[1] + self.options.grid_loc[1],
                                             self.IA_nodes.spawn_point[2] + self.options.grid_loc[2]],
                                             [0,0,0,1],3,0.1)
            reward = reward - 5
            self.failed_count += 1

        if terminated:
            message = Int16MultiArray()
            message.data = [-2,-2]
            self.IA_nodes.publisher_IA_actions.publish(message)

        truncated = False

        self.step_count += 1
        self.step_count_total += 1

        info = {"is_success": terminated}

        print(f'----------------------------------------------------------------- \n' + \
              f'Timer:  {self.timer} \n' + \
              f'Fact Timer:  {self.current_fact_time} \n' + \
              f'Step:  {self.step_count} \n' + \
              f'Total Step:  {self.step_count_total} \n' + \
              f'Robot position: {self.robot_coor_rel} \n' + \
              f'Robot orientation: {self.robot_orien} \n' + \
              f'Robot velocity: {self.robot_vel} \n' + \
              f'Robot velocity ang: {self.robot_ang_vel} \n' + \
              #f'Observation: {observation["observation"]} \n' + \
              f'goal_point: {self.goal_point} \n' + \
              f'distance: {dist} \n' + \
              f'angle: {self.dot_product} \n' + \
              f'Controller state: {self.IA_nodes.controller_state} \n' + \
              f'reward in step: {reward} \n' + \
              f'reward dist: {self.reward_dist} \n' + \
              f'reward ang: {self.reward_dir} \n' + \
              f'reward total: {self.total_reward} \n' + \
              f'Counter success: {self.succes_count} \n' + \
              f'Counter failes: {self.failed_count} \n' + \
              f'disc Actions: {discrete_actions} \n' + \
              f'Actions: {action} \n'  )

        self.in_step = False

        #self.ia_status = 0 if terminated else 1

        self.total_reward += reward
        self.last_reward = reward

        if delta_time == 0:
            print("esperando comparacion")
        else:
            print(1.0/delta_time)

        self.last_time = current_time
        
        time.sleep(0.1/self.accel_time)

        return observation, reward, terminated, truncated, info

    def compute_reward(self):

        dist = np.linalg.norm(self.goal_point-self.robot_coor_rel)

        self.reward_dist = 0.5 - dist

        # Step 1: Convert Quaternion to Rotation Matrix
        rotation = R.from_quat(self.robot_orien)
        rotation_matrix = rotation.as_matrix()

        # Step 2: Extract the X-axis Direction
        x_axis_direction = rotation_matrix[:, 0]

        # Step 3: Compute the Vector to the Point
        vector_to_point = np.array(self.goal_point)

        print(f"point vec: {vector_to_point}")
        print(f"orient vec: {x_axis_direction}")

        # Step 4: Calculate the Dot Product
        self.dot_product = np.dot(x_axis_direction, vector_to_point)
        self.reward_dir = self.dot_product/3.0

        reward = self.reward_dist
        return reward

    def render(self) -> Optional[np.ndarray]:  # type: ignore[override]
        if self.render_mode == "rgb_array":
            return self.state.copy()
        print(self.state)
        return None
    
    def write_report(self, path_name):
       
        report = {"agent_id" : self.options.agente,
                 "reward":self.total_reward,
                 "last_reward":self.last_reward,
                 "success":self.succes_count}
        
        file = open(path_name,'wb')
        pickle.dump(report,file)
        file.close()

    def full_reset(self, kill_status_thread):
        self.reset()

        self.ia_status = -1

        time.sleep(8)

        if kill_status_thread:
            self.detener = True


    def close(self) -> None:
        self.remove_model(f"pathfinder_{self.options.agente}").wait()
        self.remove_model(f'marker_{self.options.agente}').wait()
        # self.fast_remove_model(f"pathfinder_{self.options.agente}",tries=3, time_delay=0.4)
        # self.fast_remove_model(f'marker_{self.options.agente}',tries=3, time_delay=0.4)
        time.sleep(1)
        # os.killpg(os.getpgid(self.robot_process.pid), signal.SIGTERM)
        # self.robot_process.wait()

    def run_node(self,package ,script_name, run_arguments, node_name, namespace):

        # Construct the command
        command = [
            'ros2', 'run', package, script_name, '--ros-args',
            '-r', f'__node:={node_name}', '-r', f'__ns:={namespace}'
        ]
        for arg in run_arguments:
            command.extend(['-p',arg])

        # Start the launch file in a subprocess
        process = subprocess.Popen(command, preexec_fn=os.setpgrp)

        return process 
    
    def spawn_model(self, xml, pos):

        launch_arguments = [
            f'xml:={xml}',
            f'x_pos:={float(pos[0])}',
            f'y_pos:={float(pos[1])}',
            f'z_pos:={float(pos[2])}',
        ]

        # Construct the command
        command = [
            'ros2', 'run', 'pathfinder', 'spawn_model.py', '--ros-args',
        ]
        for arg in launch_arguments:
            command.extend(['-p',arg])

        # Start the launch file in a subprocess
        process = subprocess.Popen(command)

        return process 

    def remove_model(self, model_name):
        launch_arguments = [
            f'model_name:={model_name}',
        ]

        # Construct the command
        command = [
            'ros2', 'run', 'pathfinder', 'remove_model.py', '--ros-args', '-p'
        ] + launch_arguments

        # Start the launch file in a subprocess
        process = subprocess.Popen(command)

        return process
    
    def relocate_model(self, model_name, pos, quaternion):

        launch_arguments = [
            f'model_name:={model_name}',
            f'x_pos:={float(pos[0])}',
            f'y_pos:={float(pos[1])}',
            f'z_pos:={float(pos[2])}',
            f'x_orien:={float(quaternion[0])}',
            f'y_orien:={float(quaternion[1])}',
            f'z_orien:={float(quaternion[2])}',
            f'w_orien:={float(quaternion[3])}',
        ]

        # Construct the command
        command = [
            'ros2', 'run', 'pathfinder', 'relocate_model.py', '--ros-args',
        ]
        for arg in launch_arguments:
            command.extend(['-p',arg])

        # Start the launch file in a subprocess
        process = subprocess.Popen(command)

        return process 
    
    def fast_relocate_model(self, model_name, pos, quaternion, tries, time_delay):

        for i in range(tries):
            entity_state = EntityState()
            entity_state.name = model_name
            entity_state.pose.position.x = float(pos[0])
            entity_state.pose.position.y = float(pos[1])
            entity_state.pose.position.z = float(pos[2])
            entity_state.pose.orientation.x = float(quaternion[0])
            entity_state.pose.orientation.y = float(quaternion[1])
            entity_state.pose.orientation.z = float(quaternion[2])
            entity_state.pose.orientation.w = float(quaternion[3])

            self.IA_nodes.publisher_fast_relocate.publish(entity_state)

            time.sleep(time_delay)

    def fast_remove_model(self, model_name, tries, time_delay):
        for i in range(tries):
            msg = String()
            msg.data = model_name

            self.IA_nodes.publisher_fast_remove.publish(msg)

            time.sleep(time_delay)

    def inif_environment(self):

        # self.IA_nodes.get_logger().info("Creando mundoooooo")
        # self.map_process = self.create_a_world(f"world_{self.options.agente}_{self.world_count}",[0,0,0])
        # self.map_process.wait()

        # self.flatten_map_process = self.run_node(package='pathfinder',
        #                         script_name='get_map_flatten.py',
        #                         node_name='flatten_map',
        #                         namespace=f'/pathfinder_{self.options.agente}',
        #                         run_arguments=[
        #                                         f'map_size:={str(self.options.map_size)[1:-1]}',
        #                                         f'vox_size:={self.options.vox_size}',
        #                                         f'depth_image_dim:={str(self.options.depth_image_dim)[1:-1]}',
        #                                         f'x_loc:={float(self.options.grid_loc[0])}',
        #                                         f'y_loc:={float(self.options.grid_loc[1])}' ,
        #                                         f'z_loc:={float(self.options.grid_loc[2])}'
        #                                     ])

        # bt_path = os.path.join(self.pkg_pathfinder, 'gen_worlds_model', 'bt',f'world_{self.options.agente}_{self.world_count}.bt')
        # self.octomap_process = self.run_node(package='octomap_server',
        #                                      script_name='octomap_server_node',
        #                                      node_name='octomap_server_node',
        #                                      namespace=f'/pathfinder_{self.options.agente}',
        #                                      run_arguments=[
        #                                                     f'octomap_path:={bt_path}',
        #                                                     f'frame_id:=frame_id',
        #                                                 ])

        # print("Getting spawn_point nodes")
        # while self.IA_nodes.spawn_point is None:
        #     print("Waiting for spawn_points")
        #     time.sleep(0.5)

        # print("Getting flatten_map nodes")
        # while self.IA_nodes.flatten_map is None:
        #     print("Waiting for flatten_map")
        #     time.sleep(0.5)

        self.IA_nodes.spawn_point = np.random.uniform(low=-(self.options.map_size[0]-1.2)/2, high=(self.options.map_size[0]-1.2)/2, size=(3,))
        self.IA_nodes.spawn_point[2] = 0
        self.goal_point = np.random.uniform(low=-0.8, high=0.8, size=(3,))
        self.goal_point[2] = 0.2

        self.robot_process = self.init_robot_process()

        print('Creando marker')
        self.create_a_marker(self.options.agente).wait()
        self.fast_relocate_model(f'marker_{self.options.agente}',
                    self.goal_point*(self.options.map_size/2) + self.options.grid_loc,
                    [0,0,0,1],3,0.1)

        print("Getting odom nodes")
        while self.IA_nodes.robot_odom is None:
            print("Waiting for odom nodes")
            time.sleep(0.5)
        
        print("Getting controller state")
        while self.IA_nodes.controller_state is None:
            print("Waiting for controller state")
            time.sleep(0.5)

        self.map_process = self.create_a_world(f"world_{self.options.agente}_{self.world_count+1}",[0,0,-2])

    def create_a_marker(self, marker_num):
        xml = f"""
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="marker_{marker_num}">
    <pose>0 0 0 0 0 0</pose>
    <link name="link">
      <inertial>
        <mass>1.0</mass>
        <inertia>
          <ixx>0.083</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>0.083</iyy>
          <iyz>0.0</iyz>
          <izz>0.083</izz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry>
          <box>
            <size>0.25 0.25 0.25</size>
          </box>
        </geometry>
        <surface>
          <contact>
            <ode/>
          </contact>
          <bounce/>
          <friction>
            <ode/>
          </friction>
        </surface>
      </collision>
      <visual name="visual">
        <geometry>
          <box>
            <size>0.25 0.25 0.25</size>
          </box>
        </geometry>
        <material>
          <script>
            <uri>file://media/materials/scripts/gazebo.material</uri>
            <name>Gazebo/Red</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>

        """

        return self.spawn_model(xml,[0,0,0.5])

    def create_a_world(self, world_name, offset):

        xml = f"""
        <sdf version="1.6">
            <model name="{world_name}">
                <static>true</static>
                <link name="link">
                    <visual name="visual">
                        <geometry>
                            <mesh>
                                <uri>{os.path.join(self.pkg_pathfinder, 'gen_worlds_model')}/dae/{world_name}.dae</uri>
                            </mesh>
                        </geometry>
                    </visual>
                    <collision name="collision">
                        <geometry>
                            <mesh>
                                <uri>{os.path.join(self.pkg_pathfinder, 'gen_worlds_model')}/dae/{world_name}.dae</uri>
                            </mesh>
                        </geometry>
                    </collision>
                </link>
            </model>
        </sdf>
        """
        
        launch_arguments = [
            f'xml:={xml}',
            f'x_loc:={float(self.options.grid_loc[0] + offset[0])}',
            f'y_loc:={float(self.options.grid_loc[1] + offset[1])}',
            f'z_loc:={float(self.options.grid_loc[2] + offset[2])}',
            f'namespace:=pathfinder_{str(self.options.agente)}',
            f'map_size:={str(self.options.map_size)[1:-1]}'
        ]
        self.IA_nodes.get_logger().info(f'Iniciando creacion de mundo')
        # Construct the command
        command = [
            'ros2', 'launch', 'pathfinder', 'create_load_gazebo_map_launch.py'
        ] + launch_arguments

        # Start the launch file in a subprocess
        process = subprocess.Popen(command, preexec_fn=os.setpgrp)

        return process 
    
    def init_robot_process(self):
        pkg_pathfinder = get_package_share_directory('pathfinder')

        xacro_file = xacro.process_file(os.path.join(pkg_pathfinder, 'description/', 'robot.urdf.xacro'))
        urdf_path = os.path.join(pkg_pathfinder, 'description/', 'full_pathfinder.urdf')

        urdf_file = open(urdf_path, "w")
        urdf_file.write(xacro_file.toxml())
        urdf_file.close()

        robot_name = 'pathfinder_'+str(self.options.agente)
        launch_arguments = [
                        f'use_sim_time:=True',
                        f'robot_name:={robot_name}',
                        f'robot_namespace:={robot_name}',
                        f'tf_remapping:=/{robot_name}/tf',
                        f'static_tf_remap:=/{robot_name}/tf_static',
                        f'scan_topic:=/{robot_name}/scan',
                        f'map_topic:=/{robot_name}/map',
                        f'odom_topic:=/{robot_name}/odom',
                        # 'slam_params': slam_params_path,
                        f'frame_prefix:={robot_name}/',
                        f'urdf:={open(urdf_path).read()}',
                        f'urdf_path:={urdf_path}',
                        f'x:={self.IA_nodes.spawn_point[0] + self.options.grid_loc[0]}',
                        f'y:={self.IA_nodes.spawn_point[1] + self.options.grid_loc[1]}',
                        f'z:={self.IA_nodes.spawn_point[2] + self.options.grid_loc[2]}'
        ]

        # Construct the command
        command = [
            'ros2', 'launch', 'pathfinder', 'generic_spawn_launch.py'
        ] + launch_arguments
        self.IA_nodes.get_logger().info(f'Iniciando creacion de robot')
        # Start the launch file in a subprocess
        process = subprocess.Popen(command)

        return process 
