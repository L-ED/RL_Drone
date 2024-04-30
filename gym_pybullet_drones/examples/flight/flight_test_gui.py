from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import FlightFullState
import time
# import torch
import os
import pybullet as pb
import numpy as np

def main(test=True):

    savedir = '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/flight/multienv/' 
    savepath= os.path.join(
        savedir,
        # 'PPO_35'
        'PPO_3'
    )
    trainer = PPO
    # # trainer = SAC

    env_class = FlightFullState

    # policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

    env = env_class(visualize=True)
    env.randomize = False
    env.validation = True

    x = pb.addUserDebugParameter('x', -1, 1, 0.)
    y = pb.addUserDebugParameter('y', -1, 1, 0.)
    z = pb.addUserDebugParameter('z', -1, 1, 1.)
    speed = pb.addUserDebugParameter('z', 0., env.drone.max_speed, 0.)
    
    state, _=env.reset()

    rew = 0

    agent = trainer.load(
        os.path.join(savepath, 'best_model'), 
        env=env)

    state, _=env.reset()
    rew = 0
    while test:

        env.command = np.array([
            pb.readUserDebugParameter(x),
            pb.readUserDebugParameter(y),
            pb.readUserDebugParameter(z),
            pb.readUserDebugParameter(speed),
        ])


        action, _ = agent.predict(
            state.reshape(1,-1),
            deterministic=True
        )


        state, reward, terminated, truncated, info = env.step(action)
        
        rew+=reward

        time.sleep(env.timestep)
        if terminated or truncated:
            print(rew)
            rew=0
            state, _=env.reset()


if __name__=='__main__':
    main()