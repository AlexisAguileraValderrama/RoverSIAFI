
### Este archivo es para poner a entrenar el agente, no debe haber cambios grandes aqui
### Preferentemente no editar
### A menos que se deseen modificar los parametros de los modelos, lo cual es sumamente importante

from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3, A2C, PPO
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy

from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from RoverNavEnv_multi import RoverNavEnvMulti

import os

import argparse
import sys

import torch

from ament_index_python.packages import get_package_share_directory

from parser_tuple import parse_tuple

import signal
import subprocess

def getOptions(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Parses command.")

    parser.add_argument("--fecha",type=str, help="Your input file.", default="debug date")

    parser.add_argument("--directorio",type=str, help="Your destination output file.", 
                        default=os.path.join(get_package_share_directory('pathfinder'), 'DRL', 'DEBUG'))
    
    parser.add_argument("--agente", type=int, help="Your destination output file.", default=0)
    parser.add_argument("--modelo", type=str, help="Your destination output file.", default="TD3")
    parser.add_argument("--checkpoint_brain", type=str, help="Your destination output file.",default="None")
    parser.add_argument("--checkpoint_pkl", type=str, help="Your destination output file.",default="None")
    parser.add_argument("--generacion",type=int, help="Your destination output file.",default=0)
    parser.add_argument("--train_steps",type=int, help="Your destination output file.",default=50000)

    parser.add_argument("--log_dir",type=str, help="Your destination output file." , 
                        default=os.path.join(get_package_share_directory('pathfinder'), 'DRL', 'DEBUG', 'DEBUG_LOG'))

    parser.add_argument('--map_size', type=parse_tuple, default=(10,10,6), help='An integer argument with a default value')
    parser.add_argument('--vox_size', type=int, default=150, help='A string argument with a default value')

    parser.add_argument('--grid_loc', type=parse_tuple, default=(0,0,0), help='value has to be (x,y,z)')
    parser.add_argument('--depth_image_dim', type=parse_tuple, default=(100,100), help='value has to be (x,y,z) this is for the neuronal network')

    options = parser.parse_args(args)

    return options

def main():
    ## Obtener los parametros de meta data del agente ###########################
    ## No mover
    options = getOptions(sys.argv[1:])

    for arg in vars(options):
        print(f'{arg}, {getattr(options, arg)}')

    model_name = options.modelo
    model_class = getattr(sys.modules[__name__], model_name) 

    date_time = options.fecha

    brain_name = f'{model_name} brain - {date_time} - ag {options.agente}'

    log_name = f'{brain_name} - {options.generacion}'

    checkpoints_path = options.directorio

    model_info = {'name': model_name,
                'brain_name' : brain_name,
                'checkpoints_path' : checkpoints_path,
                'path_load': options.checkpoint_brain,
                'replay_load':options.checkpoint_pkl,
    }

    logdir = options.log_dir

    ###############################################################################
    
    ##### Creacion del ambiente para el rover #####################################
    ## Aqui se crea al robot, el mapa inicial, y se inicializa el nodo para procesar el mapa

    env = RoverNavEnvMulti(brain_name=brain_name,
                           multi_num=options.agente,
                           continous_space=True,
                           options=options)
    
    ########## Creacion del modelo #################################################

    if model_class in [TD3, DDPG, SAC, DQN]:
        ######### Si no encuentra el cerebro y buffer indicados crea uno nuevo
        ## Siempre se crea uno nuevo al iniciar un entrenamiento desde cero..
        try:
            model = model_class.load(model_info['path_load'],env)
            model.load_replay_buffer(model_info['replay_load'])
            print(f"Modelo {model_info['path_load']} {model_name} se cargo")

        except Exception as e:
            ## PARTE IMPORTANTE A EDITAR - parametros del modelo

            noise_fact = 2 # Factor de ruido, incrementar si se quiere que el modelo explore más
            action_noise = NormalActionNoise(mean=np.zeros(2), sigma=noise_fact * np.ones(2)) #0.6

            # Estructura de las redes actor, critico
            # pi = Actora
            # qf = Critica
            policy_kwargs = dict(net_arch=dict(pi=[700,800,500],
                                qf = [700,800,500]))
            
            model = model_class("MlpPolicy",
                                env,
                                verbose = 1,
                                tensorboard_log=logdir,
                                batch_size=300,
                                learning_starts=8000,
                                buffer_size=70000,
                                action_noise=action_noise,
                                policy_kwargs=policy_kwargs,
                                )
    else:
        # Mismo proceso pero para algoritmos on-policy (PPO, A2C)
        try:
            model = model_class.load(model_info['path_load'],env)
            print(f"Modelo {model_info['path_load']} {model_name} se cargo")
        except:
            action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))
            policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                    net_arch=dict(pi=[800, 600], vf=[800, 600]))
            model = model_class("MlpPolicy",
                                env,
                                n_steps=150,
                                policy_kwargs=policy_kwargs,
                                verbose = 1,
                                tensorboard_log=logdir,
                                batch_size=100,
                                ent_coef = 0.07
                                )

    print(model.policy)

    #Total de pasos por hacer
    TIMESTEPS = options.train_steps

    EPISODIES_PER_LOG = 1

    ## INICIO DE ENTRENAMIENTO
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, log_interval=EPISODIES_PER_LOG, tb_log_name=log_name)
    
    
    ## Guardar el cerebro del modelo
    print("saving the model...")
    save_path = model_info['checkpoints_path']+"/checkpoint "+model_info['brain_name']
    model.save(save_path)

    ## Guardar el buffer, en caso de que sea on-policy
    if model_class in [TD3, DDPG, SAC, DQN]:
        print("Saving buffer replay ")
        replay_path = model_info['checkpoints_path']+"/replay "+model_info['brain_name']
        model.save_replay_buffer(replay_path)

    ## Eliminar los restante del ambiente y escribir el reporte
    env.close()
    env.write_report(model_info['checkpoints_path']+"/report "+model_info['brain_name'])
    print("Terminado entrenamiento")
    os.killpg(os.getppid(), signal.SIGTERM)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
        input("wait xd")
        print("Finishing")
        os.killpg(os.getppid(), signal.SIGTERM)