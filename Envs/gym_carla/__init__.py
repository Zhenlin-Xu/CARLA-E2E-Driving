from gym.envs.registration import register

register(
    id="CarlaIL-v0",
    entry_point="gym_carla.envs:Carla_IL_Env",
    # max_episode_steps=200,
    # reward_threshold=25.0, 
    kwargs={
        'seed': None,
        'host': '127.0.0.1',
        'port': 2000,
        'sync': True,
        'width':1280,
        'height':720, 
        'behavior':"normal",
        'map_name': 'Town01',
        # "agent": "Basic",
        # "behavior": None
    },
)

register(
    id="CarlaRL-v0",
    entry_point="gym_carla.envs:Carla_RL_Env",
    max_episode_steps=200,
    # reward_threshold=25.0, 
    kwargs={
        'seed': None,
        'host': '127.0.0.1',
        'port': 2000,
        'sync': True,
        'width':640,
        'height':360, 
        'behavior':"normal",
        'map_name': 'Town01',
    },
)
