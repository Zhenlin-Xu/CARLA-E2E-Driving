import random
import hydra
from omegaconf import DictConfig, OmegaConf

import torch


import gym
import gym_carla

from utils.net import MultiModActor

@hydra.main(config_path=".", config_name="config")
def display(cfg : DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))
    

    model = MultiModActor().cuda()
    model.load_state_dict(torch.load("/home/gav/Desktop/CARLA/Agents/IL/Models/model_weights_20220409_2311.pth"))
    model.eval()
    
    env = gym.make("CarlaRL-v0")

    try:
        for ep in range(10):
            obs = env.reset()

            for i in range(305):
                obs = torch.from_numpy(obs).unsqueeze(0)
                obs[:,:6*100*200] /= 255.0
                action, _ = model(obs, None)
                print(action[0])
                next_obs, rew, done, info = env.step(action=action[0])
                obs = next_obs
                if done:
                    break
    finally:
        del model
        env.close()
        print("Goodbye, sir!")

if __name__ == "__main__":
    display()