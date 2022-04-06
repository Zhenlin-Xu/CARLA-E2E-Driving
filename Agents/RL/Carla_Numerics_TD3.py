import gym
import torch
import torch.nn as nn
# from torch.distributions import Independent, Normal
# from torch.optim.lr_scheduler import LambdaLR

# import os
# from pprint import pprint
import numpy as np
# import tianshou

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.exploration import GaussianNoise
from tianshou.policy import TD3Policy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

training_num = 1
seed = 1
hidden_sizes = [255,255]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
actor_lr = 3e-4
critic_lr = 3e-4
lr_decay = True
step_per_epoch = 5000
step_per_collect = 1
epoch = 100
tau = 0.005
gamma = 0.99
exploration_noise = 0.1
policy_noise = 0.2
update_actor_freq = 2
update_per_step = 1
noise_clip = 0.5
n_step = 1
buffer_size = 1000000
batch_size = 256
watch = False
test_num = 0
start_timesteps = 2000

def train_Carla_TD3():

    state_shape = (9,)
    action_shape = (2,)

    try:
 
        # Setup the Random Seeds
        np.random.seed(seed=seed)
        torch.manual_seed(seed=seed)
        # train_envs.seed(seed=seed)


        net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
        actor = Actor(
            net_a, action_shape, device=device
        ).to(device)
        actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        net_c1 = Net(
            state_shape,
            action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=device
        )
        net_c2 = Net(
            state_shape,
            action_shape,
            hidden_sizes=hidden_sizes,
            concat=True,
            device=device
        )
        critic1 = Critic(net_c1, device=device).to(device)
        critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
        critic2 = Critic(net_c2, device=device).to(device)
        critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

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
            action_space=(2,)
        )

        if training_num > 1:
            buffer = VectorReplayBuffer(buffer_size, training_num)
        else:
            buffer = ReplayBuffer(buffer_size)
            # Create Gym Vectorized Env
        if training_num > 1:
            train_envs = SubprocVectorEnv(
                [lambda: gym.make("CarlaNumeric-v0") for _ in range(training_num)]
            )
        else:
            train_envs = DummyVectorEnv(
                [lambda: gym.make("CarlaNumeric-v0") for _ in range(training_num)]
            )

        train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
        # test_collector = Collector(policy, test_envs)
        train_collector.collect(n_step=start_timesteps, random=True)
        # def save_fn(policy):
        #     torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

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
                # save_fn=save_fn,
                # logger=logger,
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
    