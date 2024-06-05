from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS, HoverFullState
import time
import torch
import os
from stable_baselines3.common.callbacks import EvalCallback


def main(test=True):

    ep_size = 2000
    for buffer_size in [2000, 10000]:
        for batch_size in [64, 512, 1024]:
        # for ep_size in [2000]:

            savedir = '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/hover/multiple' 
            savepath= os.path.join(
                savedir,
                'model_' + str(buffer_size) + '_' + str(ep_size)
            )

            trainer = PPO
            env_class = HoverFullState
            policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

            env = env_class(max_step=ep_size)
            # env.randomize = False
            agent = trainer(
                'MlpPolicy', 
                env=env,
                verbose=1,
                tensorboard_log=savedir,
                # policy_kwargs=policy_kwargs
                n_steps=buffer_size, 
                batch_size=batch_size
            )

            eval_callback = EvalCallback(env, best_model_save_path=savedir,
                                    log_path=savedir, eval_freq=10000,
                                    deterministic=True, render=False)

            test_only=False
            # test_only=True
            agent.learn(1000000, callback=eval_callback)
            agent.save(savepath)
    # env = env_class(visualize=True)
    # # env.randomize = False
    # agent = trainer.load(savepath, env=env)

    # state, _=env.reset()
    # rew = 0
    # while test:
    #     action, _ = agent.predict(
    #         state.reshape(1,-1),
    #         deterministic=True
    #     )
    #     state, reward, terminated, truncated, info = env.step(action)
    #     # print(state, reward)
    #     msg = f"POS {state[0, :3]}  VEL{state[0, 6:9]}, ACC {state[0, 12:15]}"
    #     print(msg)
    #     rew+=reward

    #     time.sleep(env.timestep)
    #     if terminated or truncated:
    #         print(rew)
    #         rew=0
    #         state, _=env.reset()


if __name__=='__main__':
    main()