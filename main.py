import gym
import gym_game
import numpy as np
from src import QLearning
import time

start_time = time.time()

env = gym.make("RoobetCrash-v0")

agent = QLearning(epsilon=.2, discount=0.6, adaptive=True)

state_action_values, observation, N = agent.fit(env)

env.crash.train_or_test = "test"
agent.predict(env=env, state_action_values=state_action_values, observation=observation, N=N)

elapsed_time = time.time() - start_time
print("Time elapsed: ", elapsed_time)

