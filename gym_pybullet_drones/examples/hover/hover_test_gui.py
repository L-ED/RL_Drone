from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS, HoverFullState
import time
# import torch
import os
import pybullet as pb
import numpy as np

def main(test=True):

    savedir = '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/hover/multienv/' 
    savepath= os.path.join(
        savedir,
        # 'PPO_35'
        # 'PPO_39' #best
        'PPO_43'
        # 'curriculum/PPO_2'
    )
    trainer = PPO
    # # trainer = SAC

    env_class = HoverFullState

    # policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

    env = env_class(visualize=True)
    env.randomize = False
    env.validation = True

    x = pb.addUserDebugParameter('x', -1, 10, 0.)
    y = pb.addUserDebugParameter('y', -1, 10, 0.)
    z = pb.addUserDebugParameter('z', 0.2, 20, 1.)
    reset = pb.addUserDebugParameter('reset', 1, 0, 1)
    
    state, _=env.reset()

    rew = 0

    agent = trainer.load(
        os.path.join(savepath, 'best_model'), 
        env=env)

    state, _=env.reset()
    rew = 0
    while test:

        term = pb.readUserDebugParameter(reset)

        env.target_pos = np.array([
            pb.readUserDebugParameter(x),
            pb.readUserDebugParameter(y),
            pb.readUserDebugParameter(z),
        ])


        action, _ = agent.predict(
            state.reshape(1,-1),
            deterministic=True
        )


        state, reward, terminated, truncated, info = env.step(action)
        print(state[0][9:12])
        
        rew+=reward

        time.sleep(env.timestep)
        if terminated or truncated or not term:
            print(rew)
            rew=0
            state, _=env.reset()


if __name__=='__main__':
    main()