import sys
import os

from datetime import datetime
import pickle

import time

from ament_index_python.packages import get_package_share_directory
import subprocess

import signal

from parser_tuple import parse_tuple

def kill_all_processes_by_name(process_name):
    try:
        # Run the killall command with the specified process name
        result = subprocess.run(['killall', process_name], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Successfully terminated all processes named '{process_name}'")
        print(result.stdout.decode())
    except subprocess.CalledProcessError as e:
        print(f"Error terminating processes named '{process_name}': {e}")
        print(e.stderr.decode())

def main():

    # Eliminación de procesos que pudieron haber quedado corriendo sobre ros2 #####

    node_names = ['robot_state_publisher',
                 'octomap_server_node',
                 'velocity_controller_IA',
                 'pt_main_thread',
                 'load_world_gazebo']
    for name in node_names:
        kill_all_processes_by_name(name)

    time.sleep(3)

    ################################################################################

    # Iniciar gazebo con el launch ubicado en /pathfinder/launch/###################
    
    command = [
        'ros2', 'launch', 'pathfinder', 'init_gazebo_launch.py'
    ]
    # Start the launch file in a subprocess
    process = subprocess.Popen(command)

    ########## Creacion de directorios, estan dentro de la carpeta /ros2_ws/share/pathfinder/DRL #######

    pkg_path = get_package_share_directory('pathfinder')
    dir_DRL = os.path.join(pkg_path, 'DRL')

    script_path = os.path.join(pkg_path, 'scripts', 'train_core_multi.py')
    logdir_DRL = os.path.join(pkg_path, 'DRL', 'logs_GEN')

    if not os.path.exists(logdir_DRL):
        os.makedirs(logdir_DRL)
    
    ####### Parametros para creacion del ambiente #################################

    # Nombre unico (la fecha)
    now = datetime.now() # current date and time
    fecha = now.strftime("%m %d %Y, %H-%M-%S")
    print(fecha)

    # Numero de agentes n y generaciones G.
    num_agentes = 1
    generaciones = 4

    # Tipo de modelo a crear.
    # off-policy: TD3, DDPG, SAC, DQN
    # on-policy: PPO, A2C
    modelo = "TD3"

    # usar esto si se tiene un cerebro pre entrenado
    # checkpoint_path = "/home/serapf/ros2_ws/install/pathfinder/share/pathfinder/DRL/models/TD3/TD3 - 08 11 2024, 16-15-07/gen 1/checkpoint TD3 brain - 08 11 2024, 16-15-07 - ag 0.zip"
    # buffer_path = "/home/serapf/ros2_ws/install/pathfinder/share/pathfinder/DRL/models/TD3/TD3 - 08 11 2024, 16-15-07/gen 1/replay TD3 brain - 08 11 2024, 16-15-07 - ag 0.pkl"

    # Usar si se va a iniciar un entrenamiento desde 0
    checkpoint_path = "None"
    buffer_path = "None"

    # Pasos de entrenamiento por generación
    train_steps = 50000

    # Tamaño de mapa de los archivos .obj, de preferencia no modificar por el momento
    map_size = '10,10,1' # x, y, z
    # La resolución de la imagen de profundidad representando al mapa
    depth_image_dim = '30,30' # x, y

    # Resolución de los mapas octomap, tampoco modificar por el momento.
    vox_size = 200
 
    # Solo para darle estructura al ambiente, no vale la pena modificar
    grid_column = 2
    # Espaciado entre el area de trabajo de cada agente
    grid_pad = 2

    ###### Comienzo de entrenamiento por generaciones ##########################################################

    for generacion in range(generaciones):

        # Directorio de la generacion
        directorio = f'{dir_DRL}/models/{modelo}/{modelo} - {fecha}/gen {generacion}'

        if not os.path.exists(directorio):
            os.makedirs(directorio)

        ############ Inicializar los agentes #################################################
        for agente in range(num_agentes):

            ################# calcular el area de trabajo de cada agente #####################

            tumple_map = parse_tuple(map_size)
            x_loc = (agente%grid_column)*(tumple_map[0] + grid_pad)
            y_loc = int(agente/grid_column)*(tumple_map[0] + grid_pad)

            grid_loc = f'{x_loc},{y_loc},0'

            # Capturar todos los datos del agente #############################################

            ## fecha: la fecha que van a tener todos los modelos de todas las generaciones, es un nombre diferenciador
            ## directorio: Directorio donde se van a guardar los agentes de la generación G
            ## Agente: el número del agente n de la generación G
            ## modelo: El nombre del modelo, TD3, DDPG, SAC, etc
            ## checkpoint_brain: el directorio y nombre del cerebro ".zip" del agente n de la generacion G
            ## checkpoint_pkl: el directorio y nombre del buffer ".pkl" del agente n de la generacion G. solo en caso de ser off-policy
            ## generacion: la generacion G que se esta corriendo
            ## train_steps: los steps que van a entrenar todos los agentes de la generación G.
            ## log_dir: directorio donde se van a poner los logs de tensorboard
            ## map_size: el tamaño del mapa de los archivo .obj
            ## vox_size: La resolución del mapa octomap (.bt) generado con el .obj
            ## grid_loc: el centro de coordenadas del mapa para el agente n
            ## depth_image_dim: La resolución de la imagen de profundidad representando al mapa. Lo que entra a la red neuronal.

            flags = f'--fecha "{fecha}" ' + \
                    f'--directorio "{directorio}" ' + \
                    f'--agente {agente} ' + \
                    f'--modelo {modelo} ' + \
                    f'--checkpoint_brain "{checkpoint_path}" ' + \
                    f'--checkpoint_pkl "{buffer_path}" ' + \
                    f'--generacion {generacion} ' + \
                    f'--train_steps {train_steps} ' + \
                    f'--log_dir {dir_DRL}/logs_GEN ' + \
                    f'--map_size {map_size} ' + \
                    f'--vox_size {vox_size} ' + \
                    f'--grid_loc {grid_loc} ' + \
                    f'--depth_image_dim {depth_image_dim}'
            
            print(flags)
            
            # Command to run in the new terminal
            command = f'python3 {script_path} {flags} && exit'

            # Open a new GNOME Terminal and execute the command
            subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'{command}; exec bash'])


    ########### Esperar a que los agentes escriban sus reportes ###################

        while True:
            if len(os.listdir(directorio+"/")) == num_agentes * 3: 
                break
            time.sleep(10)
    
    ##############################################################################
    # Leer sus reportes, escribir un log general y escoger el mejor para la siguiente generación #####

        report_files = []

        for file in os.listdir(directorio+"/"):
            if file.startswith("report"):
                report_files.append(os.path.join(directorio+"/", file))

        report_list = []

        gen_log_file = open(directorio + "/log.txt", "a")
        print_log = ''

        for file_string in report_files:
            file = open(file_string,'rb')
            data = pickle.load(file)
            report_list.append(data)
            print_log += "---------------------------\n"+\
                        f"Generacion: {generacion}" +\
                        f'{data} \n'
            file.close()
        
        # Se escoge al mejor dependiendo de la recompensa de cada uno 

        agents_ord = sorted(report_list, key=lambda x: x['reward'], reverse=True)

        selected = agents_ord[0]["agent_id"]
        print_log += "++++++++++++++++++++++++++++++ \n"+\
                     f'El seleccionado fue {selected} \n'
        print(print_log)
        gen_log_file.write(print_log)
        gen_log_file.close()

        ######################################################################

        # Se guarda el cerebro
        for file in os.listdir(directorio+"/"):
            if file.endswith(f'{selected}.zip'):
                checkpoint_path = os.path.join(directorio+"/", file)
                break
        
        # Solo se crea un buffer para los modelos off-policy
        if modelo in ["TD3", "DDPG", "SAC", "DQN"]:
            for file in os.listdir(directorio+"/"):
                if file.endswith(f'{selected}.pkl'):
                    buffer_path = os.path.join(directorio+"/", file)
                    break
        
        #######################################################################

    input('Enter para terminar')
    process.send_signal(signal.SIGINT)
    process.wait()
    print('Terminado')

if __name__ == "__main__":
    main()

