import gym
import gym_carla

import torch

import os
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger

from utils.net import MultiModCritic, MultiModActor

task = 'CarlaRL-v0'
training_num = 1
seed = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
actor_lr = 3e-4
critic_lr = 3e-4
lr_decay = True
step_per_epoch = 5000
step_per_collect = 1
epoch = 30
tau = 0.005
gamma = 0.99
exploration_noise = 0.1
policy_noise = 0.2
update_actor_freq = 8
update_per_step = 1
noise_clip = 0.5
n_step = 2
buffer_size = 25000
batch_size = 64
watch = False
test_num = 0
start_timesteps = 1000

def train_Carla_TD3():

    # state_shape = env.observation_space
    # action_shape = env.action_space.shape

    try:
 
        # Setup the Random Seeds
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)
        # train_envs.seed(seed=seed)

        actor = MultiModActor().to(device)
        critic1 = MultiModCritic().to(device)
        critic2 = MultiModCritic().to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)

        # Create Gym Vectorized Env
        if training_num > 1:
            train_envs = SubprocVectorEnv(
                [lambda: gym.make(task) for _ in range(training_num)]
            )
        else:
            train_envs = DummyVectorEnv(
                [lambda: gym.make(task) for _ in range(training_num)]
            )

        policy = TD3Policy(
            actor,
            actor_optim,
            critic1,
            critic1_optim,
            critic2,
            critic2_optim,
            tau=tau,
            gamma=gamma,
            exploration_noise=GaussianNoise(sigma=exploration_noise),
            policy_noise=policy_noise,
            update_actor_freq=update_actor_freq,
            noise_clip=noise_clip,
            estimation_step=n_step,
            action_space=train_envs.action_space,
            # action_scaling=False,
            # action_bound_method="clip",
            # **{"action_scaling":False}
        )

        if training_num > 1:
            buffer = VectorReplayBuffer(buffer_size, training_num)
        else:
            buffer = ReplayBuffer(buffer_size)

        train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
        # test_collector = Collector(policy, test_envs)
        train_collector.collect(n_step=start_timesteps, random=True)
        # def save_fn(policy):
        #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

        # log
        t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
        log_file = f'seed_{seed}_{t0}-{task.replace("-", "_")}_td3'
        log_path = os.path.join('logs', task, 'td3', log_file)
        writer = SummaryWriter(log_path)
        # writer.add_text("args", str(args))
        logger = TensorboardLogger(writer)

        def save_fn(policy):    
            torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))   

        if not watch:
            # trainer
            result = offpolicy_trainer(
                policy,
                train_collector,
                None,
                epoch,
                step_per_epoch,
                step_per_collect,
                test_num,
                batch_size,
                save_fn=save_fn,
                logger=logger,
                update_per_step=update_per_step,
                test_in_train=False
            )
        print(result)
    
    finally:
        train_envs.close()
        del actor
        del critic1
        del critic2
        print("Goodbye Sir!")

if __name__ == "__main__":

    train_Carla_TD3()
    