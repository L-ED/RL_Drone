from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS, HoverFullState
import time
# import torch
import os
import pybullet as pb
import numpy as np

def main(test=True):

    # savedir = '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/hover' 
    # savepath= os.path.join(
    #     savedir,
    #     '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/hover/multiple/model_2000_2000.zip'
    #     # 'best'
    #     # 'best_model'
    #     # "best_model_ppo_longlearn"
    #     # 'best_model_ppo_nonorm_imu_BEST'
    #     # 'best_model_ppo_nonorm'
    #     # 'best_model_random_noize'
    # )

    # trainer = PPO
    # # trainer = SAC

    # # env_class = HoverIMU
    # # env_class = HoverGPS
    env_class = HoverFullState

    # policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

    env = env_class(visualize=True)
    env.randomize = False
    # agent = trainer.load(savepath, env=env)

    mot1 = pb.addUserDebugParameter('motor1', -2, 2, -0.5)
    mot2 = pb.addUserDebugParameter('motor1', -2, 2, -0.5)
    mot3 = pb.addUserDebugParameter('motor1', -2, 2, -0.5)
    mot4 = pb.addUserDebugParameter('motor1', -2, 2, -0.5)

    state, _=env.reset()

    rew = 0
    action = np.array([-0.5, -0.5, -0.5, -0.5])

    while test:
        # action, _ = agent.predict(
        #     state.reshape(1,-1),
        #     deterministic=True
        # )
        action = np.array([
            pb.readUserDebugParameter(mot1),
            pb.readUserDebugParameter(mot2),
            pb.readUserDebugParameter(mot3),
            pb.readUserDebugParameter(mot4)
        ])

        state, reward, terminated, truncated, info = env.step(action)
        print(reward, action)
        
        msg = f"POS {state[0, :3]}  VEL{state[0, 6:9]}, ACC {state[0, 12:15]}"
        print(msg)
        rew+=reward

        time.sleep(env.timestep)
        if terminated or truncated:
            print(rew)
            print('reseting')
            rew=0
            state, _=env.reset()


if __name__=='__main__':
    main()