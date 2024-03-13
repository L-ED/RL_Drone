from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS, HoverFullState
import time
import torch
import os
from stable_baselines3.common.callbacks import EvalCallback


def main(test=True):

    savedir = '/home/led/Simulators/Bullet/gym-pybullet-drones/gym_pybullet_drones/results' 
    savepath= os.path.join(
        savedir,
        'best_model'
    )

    trainer = PPO
    # trainer = SAC

    # env_class = HoverIMU
    # env_class = HoverGPS
    env_class = HoverFullState

    env = env_class()
    agent = trainer(
        'MlpPolicy', 
        env=env,
        verbose=1,
        tensorboard_log=savedir,
        n_steps=6000
    )

    eval_callback = EvalCallback(env, best_model_save_path=savedir,
                             log_path=savedir, eval_freq=10000,
                             deterministic=False, render=False)

    test_only=False
    # test_only=True
    if not test_only:
        agent.learn(5000000, callback=eval_callback)
        agent.save(savepath)
    env = env_class(visualize=True)
    agent = trainer.load(savepath, env=env)

    state, _=env.reset()
    rew = 0
    while test:
        action, _ = agent.predict(
            state.reshape(1,-1),
            deterministic=True
        )
        state, reward, terminated, truncated, info = env.step(action)
        # print(info['timestemp'], reward)
        rew+=reward

        time.sleep(env.timestep)
        if terminated or truncated:
            print(rew)
            rew=0
            state, _=env.reset()


if __name__=='__main__':
    main()