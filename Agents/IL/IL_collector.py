import os
import h5py
import time
# import random
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import numpy as np

import gym
import gym_carla


@hydra.main(config_path=".", config_name="config")
def collect(cfg : DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    args = {
        'host' : cfg.env.host,
        'port' : cfg.env.port,
        'map_name': cfg.env.map,
        'behavior' : cfg.env.behavior,
        'width' : cfg.env.width,
        'height' : cfg.env.height,  
    }
    env = gym.make("CarlaIL-v0", **args)

    # state_space  = env.observation_space
    # action_space = env.action_space
    # print("S:", state_space)
    # print("A:", action_space)  

    h5_path = to_absolute_path("Datasets")
    print(f"Save to {h5_path}\n")
    f = h5py.File(os.path.join(h5_path, cfg.env.map+'_'+cfg.env.behavior)+".hdf5", 'w')
    num_dataset = 0
    episode = 0

    while(num_dataset < cfg.collect.max_trajectory):

        obs = env.reset()
        action  = np.empty(shape=(200,2))
        state = np.empty(shape=(200,120009))
        DONE = False

        for step in range(cfg.collect.max_step):
            next_obs, rew, done, info = env.step([0.1,0.1])

            action[step] = info["action"]
            state[step] = obs
            # print(info["action"][0])
            obs = next_obs
            if done:
                print("DONE before the TIMELIMIT")
                DONE = True
                break

        print(f"EPISODE:{episode:3d},  STEP:{step:3d},  #DS:{num_dataset:3d}")
        episode += 1

        if not DONE:
            f.create_dataset("Traj_"+str(num_dataset), shape=(200,120009+2))
            f["Traj_"+str(num_dataset)][:,:120009] = state
            f["Traj_"+str(num_dataset)][:,120009:] = action 
            num_dataset += 1

    f.close()
    env.close()
    print(f"\nGenerate {cfg.collect.max_trajectory} datasets of trajectories")


if __name__ == "__main__":

    BEGIN = time.time()
    collect()
    END = time.time()
    print(f"\nTIME:{(END - BEGIN):4.2f}")
    print("\nGoodbye, sir!")
'''
['/Game/Carla/Maps/Town05',
 '/Game/Carla/Maps/Town03',
 '/Game/Carla/Maps/Town02_Opt',
 '/Game/Carla/Maps/Town01_Opt',
 '/Game/Carla/Maps/Town04',
 '/Game/Carla/Maps/Town10HD_Opt',
 '/Game/Carla/Maps/Town05_Opt',
 '/Game/Carla/Maps/Town07_Opt',
 '/Game/Carla/Maps/Town06_Opt',
 '/Game/Carla/Maps/Town01',
 '/Game/Carla/Maps/Town03_Opt',
 '/Game/Carla/Maps/Town06',
 '/Game/Carla/Maps/Town02',
 '/Game/Carla/Maps/Town04_Opt',
 '/Game/Carla/Maps/Town07',
 '/Game/Carla/Maps/Town10HD',
 '/Game/Carla/Maps/Town11/Town11']
'''