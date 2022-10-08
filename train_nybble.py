from stable_baselines3.sac.sac import SAC
from opencat_gym_env_nybble import OpenCatGymEnv

if __name__ == '__main__':
    # Training
    env = OpenCatGymEnv(render=True)
    policy_kwargs = dict(net_arch=[128, 128])
    model = SAC(
        policy="MlpPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./tensorboard/",
    )

    try:
        model.learn(total_timesteps=800000)
    except KeyboardInterrupt:
        print("Training interrupted... Now saving model.")

    model.save("sac_opencat")
    model.save_replay_buffer("sac_replay_buffer")
