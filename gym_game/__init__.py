from gym.envs.registration import register
from .envs.roobet_crash import RoobetCrash

register(
    id='RoobetCrash-v0',
    entry_point='gym_game.envs:RoobetCrashEnv',
    max_episode_steps=700000)
