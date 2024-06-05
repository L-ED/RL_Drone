from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS, HoverFullState
import time
# import torch
import os
import pybullet as pb
from scipy.spatial.transform import Rotation as R
import numpy as np


def fly_to_point(delta, env, agent, max_timestemp=500):

    target_point = np.array([0., 0., 10])
    state, _=env.reset_manual(target_point+delta)
    # env.drone.reset_state(state)

    term = False
    success = False

    while not term:

        env.target_pos = target_point

        action, _ = agent.predict(
            state.reshape(1,-1),
            deterministic=True
        )

        state, reward, terminated, truncated, info = env.step(action)

        pos = env.drone.state.world.pos.copy()
        if pos[2]<0.2 or env.step_idx>max_timestemp:
            term = True

        if np.sum((pos-target_point)**2)< 0.2:
            success = True
            term = True

        time.sleep(env.timestep)
    
    return success


def main(test=True):

    savedir = '/home/led/robotics/engines/Bullet_sym/gym-pybullet-drones/gym_pybullet_drones/results/hover/multienv/' 
    savepath= os.path.join(
        savedir,
        # 'PPO_35'
        # 'PPO_39' #best
        # 'PPO_43' #bestb
        'PPO_43'
        # 'curriculum/PPO_2'
    )
    trainer = PPO
    # # trainer = SAC

    env_class = HoverFullState

    env = env_class(visualize=True)
    env.randomize = False
    env.validation = True

    agent = trainer.load(
        os.path.join(savepath, 'best_model'), 
        env=env)

    # iterate through 100 points on 10 spheres with radius step 0.2
    radius = np.linspace(0.5, 3, 6)

    points = [
        np.array([0., 0., 1.]),
        np.array([0., 0., -1.])
    ]
    for pitch in np.linspace(180/4, 180 - 180/4, 3):
        for yaw in np.linspace(0.0, 360 - 360/8, 8):
            rot = R.from_euler("xyz", np.array([0, pitch, yaw]), degrees=True)
            points.append(rot.apply([0, 0, 1]))

    points = np.array(points)

    # radius = np.linspace(0.5, 3, 6)
    radius = np.linspace(1, 10, 19)

    for r in radius:
        sucess_num = 0
        for point in points*r:
            sucess_num += fly_to_point(point, env=env, agent=agent)
        print("Radius ", r, "success rate", sucess_num/len(points))



if __name__=='__main__':
    main()