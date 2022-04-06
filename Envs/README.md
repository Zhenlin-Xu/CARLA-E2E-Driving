## Gym-CARLA

### Installation
```sh
pip install -e gym-carla
```
### Env for Imitation Learning
A Gym.Env class for **imitation learning** based self-driving car in CARLA. 

```python
import gym
import gym-carla

env = gym.make("CarlaIL-v0")
```

### Env for Reinforcement Learning
A Gym.Env class for **reinforcement learning** based self-driving car in CARLA. 

```python
import gym
import gym-carla

env = gym.make("CarlaRL-v0")
```