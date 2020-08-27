import gym
from gym import spaces
import numpy as np
from gym_game.envs.roobet_crash import RoobetCrash


class RoobetCrashEnv(gym.Env):
    def __init__(self):
        self.crash = None
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0]), np.array([9, 9, 9, 9]), dtype=np.float)
        # self.observation_space = spaces.Discrete(1)

    def reset(self):
        del self.crash
        self.crash = RoobetCrash()
        self.crash.build_data_set()
        obs = self.crash.observe()
        return obs

    def step(self, action):
        self.crash.action(action)
        obs = self.crash.observe()
        reward = self.crash.evaluate(action)
        done = self.crash.is_done()
        return obs, reward, done, {}
