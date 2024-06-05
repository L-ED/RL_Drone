from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS, HoverFullState
import time
import torch
import os
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, subproc_vec_env 
from stable_baselines3.common.utils import get_latest_run_id

def make_env(env_class, rank):

    def init_():
        env = env_class(seed=rank, rank=rank)
        # env.rank = rank
        env.id = rank
        return env
    
    return init_


def get_save_path(savedir, dirname='PPO'):

    latest_run_id = get_latest_run_id(savedir, dirname)
    return os.path.join(savedir, f"{dirname}_{latest_run_id + 1}")


def main(test=True):

    proc_num = 6

    savedir = '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/hover/multienv' 
    savepath = get_save_path(savedir)

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
        n_steps=4000,
        # ent_coef=0.1
    )

    eval_freq = 20000
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=savepath,
        log_path=savepath, eval_freq=eval_freq,
        deterministic=True, render=False
    )

    agent.learn(2000000, callback=eval_callback)

    env = env_class(visualize=True)
    # env.randomize = False
    agent = trainer.load(
        os.path.join(savepath, 'best_model'), 
        env=env)

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