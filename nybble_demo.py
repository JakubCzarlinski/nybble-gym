"""Enjoy a rendering of the Nybble walking around the screen."""
from stable_baselines3.sac.sac import SAC

from nybble.pybullet_gym import PybulletGym
from nybble.serial_gym import SerialGym

SIMULATION = False

# Create OpenCatGym environment from class
if SIMULATION:
    env = PybulletGym(render=True, realtime=True)
else:
    env = SerialGym()

model = SAC.load("models/sac_opencat")
obs = env.reset()

for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render(mode="human")
    if done:
        obs = env.reset()