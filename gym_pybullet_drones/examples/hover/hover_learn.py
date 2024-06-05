from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS, HoverFullState
import time
import torch
import os
from stable_baselines3.common.callbacks import EvalCallback


def main(test=True):

    savedir = '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/hover' 
    savepath= os.path.join(
        savedir,
        'best_model'
        # "best_model_ppo_longlearn"
        # 'best_model_ppo_nonorm_imu_BEST'
        # 'best_model_ppo_nonorm'
        # 'best_model_random_noize'
    )

    trainer = PPO
    # trainer = SAC

    # env_class = HoverIMU
    # env_class = HoverGPS
    env_class = HoverFullState
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

    env = env_class()
    # env.randomize = False
    agent = trainer(
        'MlpPolicy', 
        env=env,
        verbose=1,
        tensorboard_log=savedir,
        # policy_kwargs=policy_kwargs
        # n_steps=10000
    )

    eval_callback = EvalCallback(env, best_model_save_path=savedir,
                             log_path=savedir, eval_freq=10000,
                             deterministic=True, render=False)

    test_only=False
    # test_only=True
    if not test_only:
        agent.learn(2000000, callback=eval_callback)
        agent.save(savepath)
    env = env_class(visualize=True)
    # env.randomize = False
    agent = trainer.load(savepath, env=env)

    state, _=env.reset()
    rew = 0
    while test:
        action, _ = agent.predict(
            state.reshape(1,-1),
            deterministic=True
        )
        state, reward, terminated, truncated, info = env.step(action)
        # print(state, reward)
        msg = f"POS {state[0, :3]}  VEL{state[0, 6:9]}, ACC {state[0, 12:15]}"
        print(msg)
        rew+=reward

        time.sleep(env.timestep)
        if terminated or truncated:
            print(rew)
            rew=0
            state, _=env.reset()


if __name__=='__main__':
    main()