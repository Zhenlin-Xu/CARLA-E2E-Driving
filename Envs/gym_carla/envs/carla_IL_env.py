import random
import pygame
import numpy as np

import gym
import gym_carla

import carla

from .utils.world.hud import HUD
from .utils.world.keyboard import KeyboardControl

from .utils.agents.navigation.local_planner import RoadOption
# from .utils.agents.navigation.basic_agent import BasicAgent
from .utils.agents.navigation.behavior_agent import BehaviorAgent 

from .utils.helpers.helper import *

from .utils.world.world import World


# ==============================================================================
# -- Carla IL Env -------------------------------------------------------------
# ==============================================================================

class Carla_IL_Env(gym.Env):

    def __init__(self,
            seed,
            host,
            port,
            sync,
            width,
            height,
            behavior,
            map_name,
    ):
        super(Carla_IL_Env, self).__init__()

        pygame.init()
        pygame.font.init()
        self.world = None 
        self.actor_manager = []
        self.sync = sync
        self.width, self.height = width, height
        self.agentBehavior = behavior
        self.episode = 0    

        try:
            if seed:
                random.seed(seed)
            
            self.client = carla.Client(host, port)
            self.client.set_timeout(5.0)
            if map_name:
                assert type(map_name) == str
                self.client.load_world(map_name)
            # print("Succeed to connect to the server.")

            self.traffic_manager = self.client.get_trafficmanager()
            sim_world = self.client.get_world()
            
            if self.sync:
                settings = sim_world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                sim_world.apply_settings(settings)

                self.traffic_manager.set_synchronous_mode(True)

            self.display = pygame.display.set_mode((self.width, self.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

            arg = {
                "filter": 'vehicle.tesla.model3',
                "sync": sync,
            }

            self.hud = HUD(self.width, self.height)
            self.world = World(self.client.get_world(), self.hud, arg)
            self.controller = KeyboardControl(self.world)
            self.world.hud.notification("The client succeeds to connect the server.", seconds=1.0)

            self.agent = BehaviorAgent(self.world.player, behavior=self.agentBehavior)
            self.agent.ignore_stop_signs()
            self.agent.ignore_traffic_lights()
            self.agent.set_target_speed(speed=10)

            spawn_points = self.world.map.get_spawn_points()
            destination = random.choice(spawn_points).location
            self.agent.set_destination(destination)
            self.clock = pygame.time.Clock()
            
            self.current_speed = None
            self_previous_speed = None
            
            # define the action space dimension
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
            # define the state space dimension which is a multi-modality version concatenating together in 1d
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6*100*200+3+6,), dtype=np.float32)
        
        finally:
            pass

    def reset(self):   

        self.counter = 0
        self.hud.step = self.counter
        self.episode += 1
        self.hud.episode = self.episode

        for i in range(5):
            self.clock.tick()
            if self.sync:
                self.world.world.tick()
            else:
                self.world.world.wait_for_tick()
            if self.controller.parse_events():
                return
        self.world.next_weather(reverse=True)

        # retrieve the value-based state
        acceleration = vector_to_scalar(self.world.player.get_acceleration())
        speed  = vector_to_scalar(self.world.player.get_velocity())
        self.current_speed, self.previous_speed = speed, speed
        angular_speed = vector_to_scalar(self.world.player.get_angular_velocity())
        # collision_intensity = self.world.collision_sensor.current_intensity

        # img = np.random.random((6,88,200))
        # img = np.concatenate((self.world.camera_manager.camera_frame_0,self.world.camera_manager.camera_frame_0), axis=0)
        img = self.world.semantic_camera.stack

        obs = np.concatenate((
            img.flatten().astype(np.float32), # image
            np.array([acceleration, speed, angular_speed]).astype(np.float32), # measurements
            np.ones((6,), dtype=np.float32),
        ))
        return obs

    def step(self, action):

        self.counter += 1
        self.hud.step = self.counter
        obs, reward = None, 0
        done = False
        info = {}

        # world clock ticks:
        self.clock.tick()
        if self.sync:
            self.world.world.tick()
        else:
            self.world.world.wait_for_tick()
        if self.controller.parse_events():
            return
        self.world.tick(self.clock)
        self.world.render(self.display)
        pygame.display.flip()

        # define the reward:
        def _get_distance_reward():
            following_reward = 0
        #     location=self.world.player.get_location()
        #     nearest_waypoint = self.world.map.get_waypoint(location, project_to_road=True)
        #     following_reward = np.sqrt(
        #         (location.x - nearest_waypoint.transform.location.x) ** 2 +
        #         (location.y - nearest_waypoint.transform.location.y) ** 2 )
            return following_reward
        
        def _get_invasion_reward():
            invasion_reward = 0
            if self.world.lane_invasion_sensor.is_invade:
                self.world.lane_invasion_sensor.is_invade = False
                invasion_reward = -0.1
            return invasion_reward

        def _get_collision_reward():
            collision_reward = 0
            if self.world.collision_sensor.is_collision:
                self.world.collision_sensor.is_collision = False
                self.world.collision_sensor.last_intensity = self.world.collision_sensor.current_intensity
                self.world.collision_sensor.current_intensity = 0
                collision_reward = -0.5
            return collision_reward

        def _get_speed_reward():            
            speed_coefficient = 0.05
            self.target_speed = self.current_speed
            self.current_speed = vector_to_scalar(self.world.player.get_velocity()) # m/s
            return (self.current_speed - self.target_speed) * speed_coefficient * 3.6

        reward = _get_distance_reward() + _get_collision_reward() + _get_speed_reward() + _get_invasion_reward()

        # define the DONE:
        if self.counter == 200:
            # DONE at the max episode step:
            if self.world.collision_sensor.last_intensity > 0 or self.world.lane_invasion_sensor.is_invade:
                # reset the origin for next episode due to the collision or invasion: 
                self.world.player.set_transform(self.world.map.get_waypoint(self.world.player.get_location(),project_to_road=True, 
                    lane_type=(carla.LaneType.Driving)).transform)
                # reset the destination for next episode:
                self.agent.set_destination(random.choice(self.world.map.get_spawn_points()).location)
                # print("The agent has a collision event with other actors")
            self.world.hud.notification("The agent has run out of steptime in this episode", seconds=1.0)
            # print("The agent has run out of steptime in this episode")
            # done = True
        if self.agent.done():
        # DONE at the destination:
            # reset the destination for next episode:
            self.agent.set_destination(random.choice(self.world.map.get_spawn_points()).location)            
            self.world.hud.notification("The target has been reached, searching for another target", seconds=1.0) # -> pygame logging
            # print("The target has been reached, searching for another target") # -> terminal logging
            done = True
        
        # carla autopilot codeblock
        control, roadOption = self.agent.run_step()
        control.manual_gear_shift = False
        info["action"] = np.array([control.throttle-control.brake, control.steer])
        # print(f"{control.throttle:.5f}, {control.steer:.5f}, {control.brake:.5f}")
        # print(roadOption)

        # apply the action to the agent:
        # throttle, brake = 0.0, 0.0
        # steer, throttle_brake = float(action[0]), float(action[1])
        # if throttle_brake >= 0.0:
        #     throttle = float(throttle_brake)
        #     brake = 0.0
        # else:
        #     throttle = 0.0
        #     brake = float(-throttle_brake)

        # self.world.player.apply_control(carla.VehicleControl(throttle, steer, brake))      
        self.world.player.apply_control(control)

        highlevel_onehot = [0] * 6
        if roadOption == RoadOption.LANEFOLLOW:
            highlevel_onehot[0] = 1 # car-following
        elif roadOption == RoadOption.CHANGELANELEFT:
            highlevel_onehot[1] = 1 # lane-changing to left side
        elif roadOption == RoadOption.CHANGELANERIGHT:
            highlevel_onehot[2] = 1 # lane-changing to right side
        elif roadOption == RoadOption.STRAIGHT:
            highlevel_onehot[3] = 1 # keep straight in the intersection
        elif roadOption == RoadOption.LEFT:
            highlevel_onehot[4] = 1 # turn left in the intersection
        elif roadOption == RoadOption.RIGHT:
            highlevel_onehot[5] = 1 # turn right in the intersection
        else:
            print("VOID, what happened?")

        # retrieve the value-based state
        acceleration = vector_to_scalar(self.world.player.get_acceleration())
        speed  = vector_to_scalar(self.world.player.get_velocity())
        angular_speed = vector_to_scalar(self.world.player.get_angular_velocity())
        # retrieve the vision-based state 
        img = self.world.semantic_camera.stack
        obs = np.concatenate((
            img.flatten(),
            np.array([acceleration, speed, angular_speed], dtype=np.float32),
            np.array(highlevel_onehot, dtype=np.float32),
        ))
     
        return obs, reward, done, info

    def close(self):

        if self.world is not None:
                
                settings = self.world.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self.world.world.apply_settings(settings)
                # self.traffic_manager.set_synchronous_mode(True)

                self.world.destroy()
                # print("Destroyed my world")
        
        # destroy all the actor
        # actors = self.world.world.get_actors()
        # print(actors)
        # self.client.apply_batch([carla.command.DestroyActor(actor) for actor in actors])

        pygame.quit()
        # print("Goodbye, sir!")

    def _get_state_obs(self, vehicle):
        transform = vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        x_pos = location.x
        y_pos = location.y
        z_pos = location.z
        pitch = rotation.pitch
        yaw = rotation.yaw
        roll = rotation.roll

        acceleration = vector_to_scalar(vehicle.get_acceleration())
        angular_velocity = vector_to_scalar(vehicle.get_angular_velocity())
        velocity = vector_to_scalar(vehicle.get_velocity())
        return np.array([x_pos,
                            y_pos,
                            z_pos,
                            pitch,
                            yaw,
                            roll,
                            acceleration,
                            angular_velocity,
                            velocity], dtype=np.float32)
