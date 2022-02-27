import time
import pybullet as p

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from opencat_gym_env_nybble import OpenCatGymEnv

# Create OpenCatGym environment from class
env = OpenCatGymEnv(render=True)

model = SAC.load("sac_opencat")
obs = env.reset()

for _ in range(5000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
    if done:
        obs = env.reset()
    time.sleep(1/60)