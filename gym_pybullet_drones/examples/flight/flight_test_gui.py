from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import FlightFullState
from scipy.spatial.transform import Rotation as R

import time
# import torch
import os
import pybullet as pb
import numpy as np

def main(test=True):

    savedir = '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/flight/multienv/' 
    savepath= os.path.join(
        savedir,
        # 'PPO_49',
        'PPO_56',

    )
    trainer = PPO
    # # trainer = SAC

    env_class = FlightFullState

    # policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

    env = env_class(visualize=True)
    env.randomize = False
    env.validation = True

    x = pb.addUserDebugParameter('x_angle', -1., 1., 0.)
    y = pb.addUserDebugParameter('y_angle', -1., 1., 0.)
    # z = pb.addUserDebugParameter('z', -1., 1., 1.)
    speed = pb.addUserDebugParameter('speed', 0., env.max_vel, 0.)
    
    state, _=env.reset()

    rew = 0

    agent = trainer.load(
        os.path.join(savepath, 'best_model'), 
        env=env)

    state, _=env.reset()
    rew = 0
    while test:

        x_val = pb.readUserDebugParameter(x)
        y_val = pb.readUserDebugParameter(y)
        speed_val = pb.readUserDebugParameter(speed)

        command = np.array([
            x_val,
            y_val,
            0,
            # pb.readUserDebugParameter(z),
            speed_val,
        ])
        # print(x_val, y_val)
        rot = R.from_euler("xyz", command[:3]*180, degrees=True)
        command[:3] = rot.apply([0, 0, 1])
        # print(command)
        # env.command = command
        env.command = command[:3]*command[3]

        action, _ = agent.predict(
            state.reshape(1,-1),
            deterministic=True
        )

        # action = np.array([6, 6, 6, 6])
        state, reward, terminated, truncated, info = env.step(action)
        print(state[0][6:9], command, 
              reward)
        rew+=reward
        print(env.drone.state.world.vel)

        time.sleep(env.timestep)
        if terminated or truncated:
            print("REWARD",rew)
            rew=0
            state, _=env.reset()


if __name__=='__main__':
    main()