from gym.envs.registration import register

register(
    id='MultiverseGym-v0',
    entry_point='gym_multiverse.envs:MultiverseGym'
)