from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS, HoverFullState
import time
import torch
import os
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, subproc_vec_env 

def make_env(env_class, rank):

    def init_():
        env = env_class(seed=rank, rank=rank)
        # env.rank = rank
        env.id = rank
        return env
    
    return init_


def main(test=True):

    proc_num = 5

    savedir = '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/hover/multienv' 
    savepath= os.path.join(
        savedir,
        'best_model'
        # "best_model_ppo_longlearn"
        # 'best_model_ppo_nonorm_imu_BEST'
        # 'best_model_ppo_nonorm'
        # 'best_model_random_noize'
    )

    trainer = PPO

    env_class = HoverFullState
    vec_env = SubprocVecEnv([make_env(env_class, i) for i in range(proc_num)])
    eval_env = HoverFullState()

    # env.randomize = False
    agent = trainer(
        'MlpPolicy', 
        env=vec_env,
        verbose=0,
        tensorboard_log=savedir,
        # policy_kwargs=policy_kwargs
        n_steps=4000
    )

    eval_callback = EvalCallback(eval_env, best_model_save_path=savedir,
                             log_path=savedir, eval_freq=50000,
                             deterministic=True, render=False)


    agent.learn(4000000, callback=eval_callback)
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


if __name__ == '__main__':
    main()