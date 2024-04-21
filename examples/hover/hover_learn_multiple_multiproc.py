from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS, HoverFullState
import time
import torch
import os
import multiprocessing as mp
from stable_baselines3.common.callbacks import EvalCallback


def run_exp(buffer_size):

    batch_size = 128
    # for model_scale in [1, 2]:

    savedir = os.path.join(
        '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/hover/multiple',
        str(buffer_size)#, str(batch_size),
    )
    for i in range(10):
        savepath= os.path.join(
            savedir,
            'model_'+str(i)
        )

        trainer = PPO
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
            n_steps=buffer_size, 
            # batch_size=batch_size
        )

        eval_callback = EvalCallback(env, best_model_save_path=savedir,
                                log_path=savedir, eval_freq=20000,
                                deterministic=True, render=False)

        test_only=False
        # test_only=True
        agent.learn(3000000, callback=eval_callback)
        agent.save(savepath)


def main():
    args = [4000, 8000, 16000]# buffer size

    with mp.Pool(3) as p:
        p.map(run_exp, args)
        

if __name__=='__main__':
    main()