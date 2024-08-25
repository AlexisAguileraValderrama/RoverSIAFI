from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from gymnasium import Env, spaces
from gymnasium.envs.registration import EnvSpec

from stable_baselines3.common.type_aliases import GymStepReturn

from std_msgs.msg import Int16MultiArray, Float32MultiArray, String
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import EntityState, ModelStates

import time

import datetime
from numpy.linalg import norm
import random

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
import xml.etree.ElementTree as ET

from scipy.spatial.transform import Rotation as R

from AutoEncoderMaker import Autoencoder

# Clase que define a los publishers y listeners del 
class DRL_SubsPub(Node):

    def __init__(self, name_space):
        super(DRL_SubsPub,self).__init__(f'{name_space}_IA_node')

        # Buffers para guardar 
        self.flatten_map = None       # Define el vector flatten representando el mapa (se define en get_map_flatten) 
        self.goal_point = None        # Define el punto meta (se define en get_map_flatten)
        self.robot_odom = None        # Odometria del robot 
        self.spawn_point = None       # Punto donde va a spawnear el robot (se define en get_map_flatten)
        self.controller_state = None  # Estado del control joystick del robot (se define en velocity_controller_IA) 

        self.model_states = None      # Para obtener los nombres de las entidades existentes

        # Protocolo de comunicación entre los nodos
        qos_profile = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        # Listeners para cada buffer

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


        self.sub_model_states = self.create_subscription(
            ModelStates,
            '/model_states',
            self.listener_model_states,
            10
        )
        self.sub_model_states  # prevent unused variable warning

        # Publisher para publicar las acciones del agente, van para velocity_controller_IA
        self.publisher_IA_actions = self.create_publisher(Int16MultiArray, f'{name_space}/IA_actions',10)

        # Crea publishers para comunicarse con el plugin de gazebo
        self.publisher_fast_relocate = self.create_publisher(EntityState, f'/gazebo/mgp/fast_relocate',10)
        self.publisher_fast_remove = self.create_publisher(String, f'/gazebo/mgp/fast_remove',10)
        self.publisher_fast_spawn = self.create_publisher(String, f'/gazebo/mgp/fast_spawn',10)
    
    # Elimina toda la información relativo al mapa actual para esperar la información del siguiente
    def reset_buffers(self):
        self.flatten_map = None
        self.goal_point = None
        self.spawn_point = None

    # Funciones callback para llenar los buffers
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

    def listener_model_states(self, msg):
        self.model_states = msg
        

class RoverNavEnvMulti(Env):

    spec = EnvSpec("RoverNavEnvMulti-v0", "no-entry-point")
    state: np.ndarray

    # Funcion para 
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

        # Variables para que cada itento tenga un episodio de tiempo
        self.timer = 0
        self.time_limit = 20

        #COntadores de éxito y fracaso
        self.succes_count = 0
        self.failed_count = 0

        # Variable para acelerar el pensamiento del agente (no de la simulación)
        self.accel_time = 4

        # para tener en cuenta cuantas acciones ha realizado el agente
        self.step_count = 0
        self.step_count_total = 0

        self.in_step = False
        self.total_reward = 0
        self.last_reward = 0

        self.last_time = 0

        rclpy.init()
        self.IA_nodes = DRL_SubsPub(name_space=f'pathfinder_{self.options.agente}')

        # # # Run ROS 2 spin in a separate thread
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.IA_nodes,))
        self.spin_thread.start()

        self.pkg_pathfinder = get_package_share_directory('pathfinder')

        print("estamos aca\n")

        self.init_environment()

        # input("Pidos")

        ##############Definicion de dimension de entrada ######################################

        dim_odom = 3 + 4 + 3 + 3                 # 3 pose (xyz) 4 orientation (xyzw) 3 vel lineal (xyz) 3 vel lineal (xyz)
        dim_goal_point = 3                       # 3 coordinates x y z
        dim_map = len(self.IA_nodes.flatten_map) # Depende de map_dim y de las convolusiones de get_map_falatten
        dim_controller_state = 2                 # [vel_lineal, vel_angular]
        dim_distance = 1                         # distancia entre goal point y robot
        dim_timer = 1                            # factor de tiempo ([0-1])

        ################ Experimental!!!!!! ####################################################
        # Un autoencoder que aumenta la dimension de los datos ajenos a la imagen (dim map) para que estos
        # datos tengan mas representación en la entrada de la imagen 
        # Dejar esto conlleva que se entrene al inicio rapidamente un autoencoder
        self.autoencoder_input_dim = dim_odom + dim_goal_point + dim_controller_state + dim_distance + dim_timer
        self.autoencoder_output_dim = self.autoencoder_input_dim * 15

        self.auto_encoder = Autoencoder(self.autoencoder_input_dim, self.autoencoder_output_dim)
        self.auto_encoder.fast_train() 
        #########################################################################################

        self.total_dim_input = dim_map + self.autoencoder_output_dim
        self.IA_nodes.get_logger().info(f'Input dim: {self.total_dim_input}')

        # Se define las acciones de tipo continuo y de dos dimensiones [vel_lineal, vel_ang]
        # ATENCIÓN: NO SE USAN DIRECTAMENTE LAS VELOCIDADES, LOS VALORES SE MAPEAN DE UN DOMINIO CONTINUO A UNO DISCRETO
        # EJ: [0.65,-0.23] -> [1,0]
        # EJ: [0.12, -0.99] -> [0,-1]
        # Implica 1 -> aumentar velocidad, 0 -> no aumentar, -1 -> aumentar negativamente
        # Estas acciones son procesadas en velocity controller para definir velocidades fijas tanto lineales como angulares
        # y evitar cambios de velocidades muy grandes
        self.action_space = spaces.Box(low=-1, high=1,
                                            shape=(2,), dtype=np.float16)

        # Define la dimension de la observaciones
        self.observation_space = spaces.Box(
                            low=-2,
                            high=2,
                            shape=(self.total_dim_input,),
                            dtype=np.float32,
                        )

    def seed(self, seed: int) -> None:
        self.obs_space.seed(seed)

    def _get_obs(self):

        ## Todos los elementos de la observacion son estandarizados entre -1 a 1 con respecto
        ## al tamaño del mapa y el centro del mismo
        ## Por ejemplo: Si el mapa mide 10 x 10, 1 es un 5 y -1 es un -5 (solo aplica a coordenadas, no a velocidades ni orientaciones)
        ############# Caputrar informacion de odometría del robot ####################

        # Posicion del robot
        self.robot_coor_abs = np.array([self.IA_nodes.robot_odom.pose.pose.position.x,
                            self.IA_nodes.robot_odom.pose.pose.position.y,
                            self.IA_nodes.robot_odom.pose.pose.position.z])

        # Posicion del robot estandarizado entre -1 a 1 con respecto al tamaño del mapa
        self.robot_coor_rel_std = (self.robot_coor_abs - np.array(self.options.grid_loc))/(np.array(self.options.map_size)/2)
        
        # Orientacion del robot en quaternion, -1 a 1
        self.robot_orien = np.array([self.IA_nodes.robot_odom.pose.pose.orientation.x,
                            self.IA_nodes.robot_odom.pose.pose.orientation.y,
                            self.IA_nodes.robot_odom.pose.pose.orientation.z,
                            self.IA_nodes.robot_odom.pose.pose.orientation.w])
        
        # Velocidad angulares y lineales del robot  -1 a 1
        self.robot_vel = np.array([self.IA_nodes.robot_odom.twist.twist.linear.x,
                            self.IA_nodes.robot_odom.twist.twist.linear.y,
                            self.IA_nodes.robot_odom.twist.twist.linear.z])
        
        self.robot_ang_vel = np.array([self.IA_nodes.robot_odom.twist.twist.angular.x,
                    self.IA_nodes.robot_odom.twist.twist.angular.y,
                    self.IA_nodes.robot_odom.twist.twist.angular.z])

        #############Goal point############## relative

        # Posición objetivo estandarizado a -1 a 1 con respecto al mapa
        self.goal_point_rel_std = np.array(self.IA_nodes.goal_point)/(np.array(self.options.map_size)/2)

        ############# Distancia entre punto objetivo y el robot ####################

        dist = np.linalg.norm(self.goal_point_rel_std-self.robot_coor_rel_std)

        # Tiempo restante del episodio (0-1) donde 0 = 0s, 1 = time limit 
        
        self.current_fact_time = self.timer / self.time_limit

        ################## Juntar todo los datos #################################

        sensors_data = np.concatenate((self.goal_point_rel_std,
                                    self.robot_coor_rel_std, #3
                                    self.robot_orien,   #4
                                    self.robot_vel,     #3
                                    self.robot_ang_vel, #3
                                    dist,
                                    self.current_fact_time,
                                    np.array(self.IA_nodes.controller_state)), axis=None) #2

        ####### Experimental!!!!!!!!! #########################################################
        ## Aumento de datos con auto enconder ######################################## 

        encoded_sensors_data = self.auto_encoder.encode(sensors_data)

        ################### Construir la observacion total ###########################
        ################### ya se incluye la imagen del ambiente #####################

        observation = np.concatenate((np.array([self.IA_nodes.flatten_map]),
                            encoded_sensors_data), axis=None) #2
        
        ## Este codigo lo deje por si no se usa el paso intermedio del auto encoder.
        # observation = np.concatenate((np.array([self.IA_nodes.flatten_map]),
        #                               self.goal_point_rel_std,
        #                             self.robot_coor_rel_std, #3
        #                             self.robot_orien,   #4
        #                             self.robot_vel,     #3
        #                             self.robot_ang_vel, #3
        #                             dist,
        #                             self.current_fact_time,
        #                             np.array(self.IA_nodes.controller_state)), axis=None) #2
        
        # print("-----------------------------------------------")
        # print(observation)
        # print("-----------------------------------------------")

        return observation

    def step(self, action) -> GymStepReturn:
        """
        Step into the env.

        :param action:
        :return:
        """

        # Codigo para calcular el tiempo interno del agente.
        current_time = datetime.datetime.now()
        if self.last_time == 0:
            self.last_time = current_time

        delta_time = (current_time - self.last_time).microseconds/1000000

        # Calcular el tiempo calculado del episodio
        self.timer = self.timer + delta_time * self.accel_time

        # self.timer = 0
        # input("pidos")

        # Tranformar de continuo a discreto
        discrete_actions = np.round(action,0).astype(int)

        # Publicar las acciones del agente
        message = Int16MultiArray()
        message.data = discrete_actions
        self.IA_nodes.publisher_IA_actions.publish(message)
        
        # Obtener una observacion del ambiente
        observation = self._get_obs()

        ###############################################################################################
        ##################################Funciones de recompensa######################################
        ###############################################################################################

        # Esta parte es muy importante ya que define las recompensas a considerar para el agente
        # MUY IMPORTANTE EXPERIMENTAR CON ESTO

        # Obtener recompensa de este step
        reward = self.compute_reward()

        terminated = False

        dist = np.linalg.norm(self.goal_point_rel_std-self.robot_coor_rel_std)

        # Si la distancia es menor un threshold se considera como éxito
        # RECOMPENSA POSITIVA 
        if dist < 0.12:
            terminated = True
            # Se recompensa más si el agente no tarda mucho en llegar
            reward = reward + 50 * (-1.25*self.current_fact_time + 1.75)
            self.succes_count += 1

        # Se agoto el tiempo del episodio y no llego
        # RECOMPENSA NEGATIVA
        if self.timer > self.time_limit:
            print("Time out")
            terminated = True
            reward = reward - 25*dist # Si el agente quedo muy cerca del objetivo, no se penaliza tanto
            self.failed_count += 1
        
        # Si el robot se sale del mapa, recompensa negativa y se regresa al punto spawn 
        # el episodio no termina
        if np.prod(abs(self.robot_coor_rel_std[:2]) < 0.95) == 0:
            print("Out of bounds")
            message = Int16MultiArray()
            message.data = [-2,-2]
            self.IA_nodes.publisher_IA_actions.publish(message)
            self.fast_relocate_model(f'pathfinder_{str(self.options.agente)}',
                                             [self.IA_nodes.spawn_point[0] + self.options.grid_loc[0],
                                             self.IA_nodes.spawn_point[1] + self.options.grid_loc[1],
                                             self.IA_nodes.spawn_point[2] + self.options.grid_loc[2]],
                                             [0,0,0,1],3,0.1)
            reward = reward - 5
            self.failed_count += 1

        # Si el robot se voltea, recompensa negativa y se regresa al punto spawn
        # El episodio no termina
        if self.inclination > 0.33:
            print("Flipped")
            message = Int16MultiArray()
            message.data = [-2,-2]
            self.IA_nodes.publisher_IA_actions.publish(message)
            self.fast_relocate_model(f'pathfinder_{str(self.options.agente)}',
                                             [self.IA_nodes.spawn_point[0] + self.options.grid_loc[0],
                                             self.IA_nodes.spawn_point[1] + self.options.grid_loc[1],
                                             self.IA_nodes.spawn_point[2] + self.options.grid_loc[2]],
                                             [0,0,0,1],3,0.1)
            reward = reward - 5
            self.failed_count += 1

        ###############################################################################################
        ################################# Fin de recompensas ##########################################
        ###############################################################################################

        # Si termina el episodio se detiene el robot por completo
        # [-2,-2] es un indicador para velocity controller IA para detener el robot.
        if terminated:
            message = Int16MultiArray()
            message.data = [-2,-2]
            self.IA_nodes.publisher_IA_actions.publish(message)

        truncated = False

        self.step_count += 1
        self.step_count_total += 1

        info = {"is_success": terminated}

        # IMPRIMIR INFORMACION RELEVANTE
        print(f'----------------------------------------------------------------- \n' + \
              f'No. Agente:  {self.options.agente} \n' + \
              f'Timer:  {self.timer} \n' + \
              f'Fact Timer:  {self.current_fact_time} \n' + \
              f'Step:  {self.step_count} \n' + \
              f'Total Step:  {self.step_count_total} \n' + \
              f'Robot position: {self.robot_coor_rel_std} \n' + \
              f'Robot orientation: {self.robot_orien} \n' + \
              f'Robot velocity: {self.robot_vel} \n' + \
              f'Robot velocity ang: {self.robot_ang_vel} \n' + \
              #f'Observation: {observation["observation"]} \n' + \
              f'goal_point: {self.goal_point_rel_std} \n' + \
              f'distance: {dist} \n' + \
              f'angle: {self.dot_product} \n' + \
              f'Inclination: {self.inclination} \n' + \
              f'Controller state: {self.IA_nodes.controller_state} \n' + \
              f'reward in step: {reward} \n' + \
              f'reward dist: {self.reward_dist} \n' + \
              f'reward ang: {self.reward_dir} \n' + \
              f'reward incl: {self.reward_incl} \n' + \
              f'reward total: {self.total_reward} \n' + \
              f'Counter success: {self.succes_count} \n' + \
              f'Counter failes: {self.failed_count} \n' + \
              f'disc Actions: {discrete_actions} \n' + \
              f'Actions: {action} \n'  )

        self.total_reward += reward
        self.last_reward = reward

        ############ Codigo para calcular los "APS" (Acciones por segundo)######
        if delta_time == 0:
            print("esperando comparacion")
        else:
            print(1.0/delta_time)

        self.last_time = current_time
        
        time.sleep(0.05/self.accel_time)
        #########################################################################

        return observation, reward, terminated, truncated, info

    def compute_reward(self):

        # Funcion que define las recompensas
        # IMPORTANTE EXPERIMENTAR CON ESTO

        ############# 1. Recompensa en funcion de la distancia entre el objetivo y el robot ###############

        dist = np.linalg.norm(self.goal_point_rel_std-self.robot_coor_rel_std)

        self.reward_dist = 0.5 - dist

        ############# 2. Recompensa si el robot está viendo el objetivo

        # Step 1: Convert Quaternion to Rotation Matrix
        rotation = R.from_quat(self.robot_orien)
        rotation_matrix = rotation.as_matrix()

        # Step 2: Extract the X-axis Direction
        x_axis_direction = rotation_matrix[:, 0]

        # Step 3: Compute the Vector to the Point
        vector_to_point = np.array(self.goal_point_rel_std)

        print(f"point vec: {vector_to_point}")
        print(f"orient vec: {x_axis_direction}")

        # Step 4: Calculate the Dot Product
        self.dot_product = np.dot(x_axis_direction, vector_to_point)

        self.reward_dir = self.dot_product/3.0

        ############# 3. Recompensa si el robot no se inclina 
        ############# es decir, que no caiga en algun hoyo o choque con algo.

        euler_angles = rotation.as_euler('xyz', degrees=True)

        roll, pitch, yaw = euler_angles

        # Inclination can be considered as the magnitude of the roll and pitch
        self.inclination = np.sqrt(roll**2 + pitch**2)/180

        self.reward_incl = 0.1 - self.inclination

        ######### Sumar todas las recompensas ########################
        ######### Por el momento solo he experimentado directamente con la distancia
        # reward = self.reward_dist + self.reward_dir + self.reward_incl

        reward = self.reward_dist

        return reward

    def init_environment(self):

        # Funcion para inicializar el ambiente
        # Solo se llama una vez

        # Se espera a obtener el nombre de los modelos de gazebo
        # Aqui puede estar vacia la lista pero no None
        print("Getting model states node")
        while self.IA_nodes.model_states is None:
            print("Waiting for model_states node")
            time.sleep(0.5)

        # Se crea un mapa
        self.current_file_world_name = f"world_{random.randint(0, 300)}"
        self.current_world_name = self.current_file_world_name + f"_ag_{self.options.agente}"
        self.create_a_world(world_file_name=self.current_file_world_name,
                            world_name=self.current_world_name,
                            offset=[0,0,0.1])


        # Se inicia el nodo para procesar la imagen de profundidad del mapa
        self.flatten_map_process = self.run_node(package='pathfinder',
                                script_name='get_map_flatten.py',
                                node_name='flatten_map',
                                namespace=f'/pathfinder_{self.options.agente}',
                                run_arguments=[
                                                f'map_size:={str(tuple(self.options.map_size))[1:-1]}',
                                                f'vox_size:={self.options.vox_size}',
                                                f'depth_image_dim:={str(self.options.depth_image_dim)[1:-1]}',
                                                f'x_loc:={float(self.options.grid_loc[0])}',
                                                f'y_loc:={float(self.options.grid_loc[1])}' ,
                                                f'z_loc:={float(self.options.grid_loc[2])}'
                                            ])
        
        # Se inicia el nodo de octomap para abrir el archivo .bt y enviar la información a get_map_flatten
        bt_path = os.path.join(self.pkg_pathfinder, 'gen_worlds_model', 'bt',f'{self.current_file_world_name}.bt')
        self.octomap_process = self.run_node(package='octomap_server',
                                             script_name='octomap_server_node',
                                             node_name='octomap_server_node',
                                             namespace=f'/pathfinder_{self.options.agente}',
                                             run_arguments=[
                                                            f'octomap_path:={bt_path}',
                                                            f'frame_id:=frame_id',
                                                        ]
                                            ,grp=True)

        # Esperar la información del mapa de profundidad proporcionado por get_map_flatten
        print("Getting spawn_point nodes")
        while self.IA_nodes.spawn_point is None:
            print("Waiting for spawn_points")
            time.sleep(0.5)

        print("Getting goal_point nodes")
        while self.IA_nodes.goal_point is None:
            print("Waiting for goal_point")
            time.sleep(0.5)

        print("Getting flatten_map nodes")
        while self.IA_nodes.flatten_map is None:
            print("Waiting for flatten_map")
            time.sleep(0.5)

        # self.IA_nodes.spawn_point = np.random.uniform(low=-(self.options.map_size[0]-1.2)/2, high=(self.options.map_size[0]-1.2)/2, size=(3,))
        # self.IA_nodes.spawn_point[2] = 0
        # self.goal_point = np.random.uniform(low=-0.8, high=0.8, size=(3,))
        # self.goal_point[2] = 2

        # Se inicia la creación del robot
        self.robot_process = self.init_robot_process()

        # Se crea un marcador para identificar el punto objetivo
        print('Creando marker')
        self.create_a_marker(self.options.agente)
        self.fast_relocate_model(f'marker_{self.options.agente}',
                    self.IA_nodes.goal_point + self.options.grid_loc,
                    [0,0,0,1],3,0.1)

        # Se espera la odometría del robot y el nodo del control de velocidad
        print("Getting odom nodes")
        while self.IA_nodes.robot_odom is None:
            print("Waiting for odom nodes")
            time.sleep(0.5)
        
        print("Getting controller state")
        while self.IA_nodes.controller_state is None:
            print("Waiting for controller state")
            time.sleep(0.5)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Union[float, np.ndarray]], Dict]:
        
        if seed is not None:
            self.obs_space.seed(seed)

        # Poner el robot en un lugar que no sea peligroso para la eliminación
        self.fast_relocate_model(f'pathfinder_{str(self.options.agente)}',
                            [self.options.grid_loc[0] + (self.options.map_size[0]/2) + 1,
                            0,
                            0.1],
                            [0,0,0,1],3,0.1)

        # Eliminar el mundo
        self.fast_remove_model(self.current_world_name,3,0.2)

        # Crear mundo nuevo
        self.current_file_world_name = f"world_{random.randint(0, 300)}"
        self.current_world_name = self.current_file_world_name + f"_ag_{self.options.agente}"
        self.create_a_world(world_file_name=self.current_file_world_name,
                            world_name=self.current_world_name,
                            offset=[0,0,0.1])

        ################Reacomodar mapas e iniciar nuevo octomap#################
        # Eliminar nodo octomap
        os.killpg(os.getpgid(self.octomap_process.pid), signal.SIGTERM)
        self.octomap_process.wait()

        # Eliminar toda la información anterior para esperar la siguiente
        self.IA_nodes.reset_buffers()

        # crea otro octomap con el nuevo mapa
        bt_path = os.path.join(self.pkg_pathfinder, 'gen_worlds_model', 'bt',f'{self.current_file_world_name}.bt')
        self.octomap_process = self.run_node(package='octomap_server',
                                             script_name='octomap_server_node',
                                             node_name='octomap_server_node',
                                             namespace=f'/pathfinder_{self.options.agente}',
                                             run_arguments=[
                                                            f'octomap_path:={bt_path}',
                                                            f'frame_id:=frame_id',
                                                        ]
                                              ,grp=True)
        
        ################Espera información del mapa##############

        print("Getting spawn_point nodes")
        while self.IA_nodes.spawn_point is None:
            print("Waiting for spawn_points")
            time.sleep(0.5)

        print("Getting goal_point nodes")
        while self.IA_nodes.goal_point is None:
            print("Waiting for goal_point")
            time.sleep(0.5)

        print("Getting flatten_map nodes")
        while self.IA_nodes.flatten_map is None:
            print("Waiting for flatten_map")
            time.sleep(0.5)

        ## Poner el robot y marcador en los spawn point y goal point respectivos.

        self.fast_relocate_model(f'pathfinder_{str(self.options.agente)}',
                                            [self.IA_nodes.spawn_point[0] + self.options.grid_loc[0],
                                            self.IA_nodes.spawn_point[1] + self.options.grid_loc[1],
                                            self.IA_nodes.spawn_point[2] + self.options.grid_loc[2]],
                                            [0,0,0,1],3,0.1)

        self.fast_relocate_model(f'marker_{self.options.agente}',
                    self.IA_nodes.goal_point + self.options.grid_loc,
                    [0,0,0,1], 3, 0.1)

        # Reinicia el step count y el tiempo
        self.step_count = 0
        self.timer = 0

        obs = self._get_obs()
        self.in_step = False

        print(f'flat image: {len(self.IA_nodes.flatten_map)}')
        print(f'goal: {len(self.IA_nodes.goal_point)}')
        print(f'robot_pos: {len(self.robot_coor_rel_std)}')
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
    
    def write_report(self, path_name):
       
        # Se escribe el reporte del agente, 
        # si se quiere agregar mas informacion al reporte, agregarlo al reporte

        report = {"agent_id" : self.options.agente,
                 "reward":self.total_reward,
                 "last_reward":self.last_reward,
                 "success":self.succes_count}
        
        file = open(path_name,'wb')
        pickle.dump(report,file)
        file.close()

    def create_a_marker(self, marker_num):
        # Esto es para crear un cubo que represente el punto objetivo
        xml = f"""
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
                    <size>0.2 0.2 0.2</size>
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

        self.fast_spawn_model(xml,3,0.8)

    def create_a_world(self, world_file_name, world_name, offset):

        # COdigo para crear el mundo

        self.IA_nodes.get_logger().info(f"Creating: {world_name}")
        self.IA_nodes.get_logger().info(f"With file: {world_file_name}")

        uri = os.path.join(self.pkg_pathfinder, 'gen_worlds_model') + f'/dae/{world_file_name}.dae'

        if not os.path.exists(uri):
            raise FileNotFoundError(f"The file '{uri}' does not exist.")

        xml = f"""
        <sdf version="1.6">
            <model name="{world_name}">
                <pose>{float(self.options.grid_loc[0] + offset[0])}
                      {float(self.options.grid_loc[1] + offset[1])}
                       {float(self.options.grid_loc[2] + offset[2])} 0 0 0</pose>
                <static>true</static>
                <link name="link">
                    <visual name="visual">
                        <geometry>
                            <mesh>
                                <uri>{uri}</uri>
                            </mesh>
                        </geometry>
                    </visual>
                    <collision name="collision">
                        <geometry>
                            <mesh>
                                <uri>{uri}</uri>
                            </mesh>
                        </geometry>
                    </collision>
                </link>
            </model>
        </sdf>
        """
        self.fast_spawn_model(xml, 1, 0.7)
    
    def init_robot_process(self):

        # Codigo para inicializar el robot y el controller velocity
        # Para mas detalle revisar launch/generic_spawn_launch.py 

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
                        f'z:={self.IA_nodes.spawn_point[2] + self.options.grid_loc[2] + 2}'
        ]

        # Construct the command
        command = [
            'ros2', 'launch', 'pathfinder', 'generic_spawn_launch.py'
        ] + launch_arguments
        self.IA_nodes.get_logger().info(f'Iniciando creacion de robot')
        # Start the launch file in a subprocess
        process = subprocess.Popen(command)

        return process 

    def run_node(self,package ,script_name, run_arguments, node_name, namespace='', grp = False):

        # Funcion para correr un nodo cualquiera

        command = [
            'ros2', 'run', package, script_name, '--ros-args',
            '-r', f'__node:={node_name}', '-r', f'__ns:={namespace}'
        ]
        for arg in run_arguments:
            command.extend(['-p',arg])

        # Start the launch file in a subprocess
        if grp:
            process = subprocess.Popen(command, preexec_fn=os.setpgrp)
        else:
            process = subprocess.Popen(command)

        return process 

    def remove_model(self, model_name):

        # Codigo para eliminar un modelo, principalmente el robot
        # para otras entidades utilizar fast_remove_model()

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
    
    def fast_relocate_model(self, model_name, pos, quaternion, tries, time_delay):

        # Manda un mensaje al plugin de gazebo para reposiconar un modelo.

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

        # Manda un mensaje al plugin de gazebo para eliminar un modelo.

        i = 0
        while True:

            time.sleep(time_delay)

            if i >= tries:
                print("Can not remove model")
                return

            if self.IA_nodes.model_states is None:
                print(f"No model state info")
            else:
                if model_name not in self.IA_nodes.model_states.name:
                    print(f"The model {model_name} does not exist, not removing")
                    return
                else:
                    msg = String()
                    msg.data = model_name
                    self.IA_nodes.publisher_fast_remove.publish(msg)
                    print(f"The model {model_name} removed correctly")
            i += 1
            
    def fast_spawn_model(self, xml, tries, time_delay):

        # Manda un mensaje al plugin de gazebo para spawnear un modelo.


        root = ET.fromstring(xml)
        
        # Assuming the SDF string has the standard structure
        model = root.find('model')
        if model is not None:
            model_name = model.get('name')
        else:
            print("No 'model' tag found in the SDF string.")
            return

        i = 0
        while True:

            time.sleep(time_delay)

            if i >= tries:
                print("Can not create model state")
                return

            if self.IA_nodes.model_states is None:
                print(f"No model state info")
            else:
                if model_name in self.IA_nodes.model_states.name:
                    print(f"The model {model_name} already exists, not creating")
                    return
                else:
                    msg = String()
                    msg.data = xml
                    self.IA_nodes.publisher_fast_spawn.publish(msg)
                    print(f"The model {model_name} spawn correctly")
            
            i += 1

    def render(self) -> Optional[np.ndarray]:  # type: ignore[override]
        if self.render_mode == "rgb_array":
            return self.state.copy()
        print(self.state)
        return None
    
    def close(self) -> None:

        os.killpg(os.getpgid(self.octomap_process.pid), signal.SIGTERM)
        self.octomap_process.wait()
        
        self.remove_model(f"pathfinder_{self.options.agente}").wait()
        self.fast_remove_model(f'marker_{self.options.agente}',tries=3, time_delay=0.4)
        self.fast_remove_model(self.current_world_name,3,0.3)

        time.sleep(1)

