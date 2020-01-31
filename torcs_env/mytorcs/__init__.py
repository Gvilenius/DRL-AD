from gym.envs.registration import register

register(
    id='MyTorcs-v0',
    entry_point='mytorcs.env:MyTorcsEnv',
)