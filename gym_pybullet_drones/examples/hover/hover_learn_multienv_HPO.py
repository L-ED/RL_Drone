from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS, HoverFullState
import time
import torch
import os
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
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

    proc_num = 5

    savedir = '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/hover/multienv/HPO' 
    # savepath = get_save_path(savedir)

    trainer = PPO

    env_class = HoverFullState


    for batch_size in [64, 128, 512]:
        for buffer_size in [4000, 8000, 16000]:
            for model_scale in [1, 2, 4]:
                for i in range(1):

                    exp_dir = os.path.join(
                        savedir,
                        f"{buffer_size}_{batch_size}_{model_scale}")
                    
                    savepath = get_save_path(
                        exp_dir 
                    )
                
                    vec_env = SubprocVecEnv([make_env(env_class, i) for i in range(proc_num)])
                    eval_env = HoverFullState()

                    # env.randomize = False
                    agent = trainer(
                        'MlpPolicy', 
                        env=vec_env,
                        verbose=0,
                        tensorboard_log=exp_dir,
                        policy_kwargs=dict(net_arch=dict(pi=[64*model_scale, 64*model_scale], vf=[64*model_scale, 64*model_scale])),
                        n_steps=buffer_size,
                        batch_size=batch_size
                    )

                    eval_freq = 20000
                    eval_callback = EvalCallback(
                        eval_env, best_model_save_path=savepath,
                        log_path=savepath, eval_freq=eval_freq,
                        deterministic=True, render=False
                    )

                    agent.learn(4000000, callback=eval_callback)


if __name__ == '__main__':
    main()