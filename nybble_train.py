from stable_baselines3.sac.sac import SAC

from nybble.pybullet_gym import PyBulletGym

if __name__ == '__main__':
    # Training
    env = PyBulletGym(render=False)
    policy_kwargs = dict(net_arch=[128, 128])
    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tensorboard/",
    )

    try:
        model.learn(total_timesteps=20000)
    except KeyboardInterrupt:
        print("Training interrupted... Now saving model.")

    model.save("models/sac_opencat")
    model.save_replay_buffer("models/replay_buffer/sac_replay_buffer")
