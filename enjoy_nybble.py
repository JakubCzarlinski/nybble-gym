"""Enjoy a rendering of the Nybble walking around the screen."""
import time

from stable_baselines3.sac.sac import SAC
from opencat_gym_env_nybble import OpenCatGymEnv

# Create OpenCatGym environment from class
env = OpenCatGymEnv(render=True)

model = SAC.load("sac_opencat")
obs = env.reset()

for _ in range(5000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
    if done:
        obs = env.reset()
    time.sleep(1.0/60.0)
