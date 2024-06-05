from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import FlightFullState
import time
import torch
import numpy as np
import os
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, subproc_vec_env 
from stable_baselines3.common.utils import get_latest_run_id

import tensorflow as tf

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


class CurriculumCalback(BaseCallback):

    parent: EvalCallback

    def __init__(self, train_env, max_iter: int, eval_freq: int, num_envs: int, reward_threshold=180, wait_eps=5, verbose=0):
        super().__init__(verbose=verbose)
        step_num = max_iter//(eval_freq*num_envs*wait_eps) -1 # should be learned with 1 in last step
        self.plan = np.linspace(start=0.1, stop=1, num=step_num)**2
        self.i = 0
        self.reward_threshold = reward_threshold
        self.train_env = train_env
        self.skip_eps = 0
        self.started = False


    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used with an ``EvalCallback``"
        if self.parent.last_mean_reward > self.reward_threshold:
            self.i+=1
            if self.i == len(self.plan):
                self.i -=1

            self.train_env.env_method("set_alpha", self.plan[self.i])

        # summary = tf.summary(value=[tf.summary.Value(tag='alpha', simple_value=self.plan[self.i])])
        # self.locals['writer'].add_summary(summary, self.num_timesteps)
        # self.locals['writer'].scalar('alpha', self.plan[self.i])
        self.logger.record("eval/alpha", self.plan[self.i])
        self.logger.dump(self.num_timesteps)
        dir_err, mag_err = np.array(self.parent.eval_env.get_attr('history')).mean(axis=1)[0]
        self.logger.record("eval/dir_err", dir_err)
        self.logger.record("eval/mag_err", mag_err)




def main(test=False):

    proc_num = 6

    savedir = '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/flight/multienv' 
    savepath = get_save_path(savedir)

    trainer = PPO

    env_class = FlightFullState
    vec_env = SubprocVecEnv([make_env(env_class, i) for i in range(proc_num)])
    eval_env = FlightFullState()

    # env.randomize = False
    agent = trainer(
        'MlpPolicy', 
        env=vec_env,
        verbose=0,
        tensorboard_log=savedir,
        # policy_kwargs=policy_kwargs
        # n_steps=4000,
        n_steps=8000,
        # ent_coef=0.1
    )

    eval_freq = 16000
    iter_num = 4000000

    curric_callback =CurriculumCalback(
        train_env=vec_env, max_iter=iter_num, 
        eval_freq=eval_freq, num_envs=proc_num
    )

    eval_callback = EvalCallback(
        eval_env, best_model_save_path=savepath,
        log_path=savepath, eval_freq=eval_freq,
        deterministic=True, render=False, 
        callback_after_eval= curric_callback
    )

    agent.learn(iter_num, callback=eval_callback)

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
        # print(msg)
        rew+=reward

        time.sleep(env.timestep)
        if terminated or truncated:
            # print(rew)
            rew=0
            state, _=env.reset()


if __name__ == '__main__':
    main()