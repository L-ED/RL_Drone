from stable_baselines3 import PPO, SAC, TD3
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS, HoverFullState
import time
# import torch
import os
import pybullet as pb
from scipy.spatial.transform import Rotation as R
import numpy as np
import json


def fly_to_point(delta, env, agent, max_timestemp=500):

    target_point = np.array([0., 0., 10])
    state, _=env.reset_manual(target_point+delta)
    # env.drone.reset_state(state)

    term = False
    success = False
    transitions = {
        'pos':[],
        'rpy':[],
        'vel':[],
        'angvel':[],
        "motor_speed":[]
    }

    while not term:

        env.target_pos = target_point

        action, _ = agent.predict(
            state.reshape(1,-1),
            deterministic=True
        )

        state, reward, terminated, truncated, info = env.step(action)
        transitions['pos'].append(env.drone.state.world.pos.copy().tolist())
        transitions['rpy'].append(env.drone.state.world.rpy.copy().tolist())
        transitions['vel'].append(env.drone.state.world.vel.copy().tolist())
        transitions['angvel'].append(env.drone.state.world.ang_vel.copy().tolist())
        transitions['motor_speed'].append(env.preprocess_action(action).tolist())


        pos = env.drone.state.world.pos.copy()
        if pos[2]<0.2 or env.step_idx>max_timestemp:
            term = True

        if np.sum((pos-target_point)**2)< 0.2:
            success = True
            term = True

        time.sleep(env.timestep)
    
    return success, transitions


def main(test=True):

    data = {}
    savedir = '/home/led/robotics/engines/Bullet_sym/RL_Drone/gym_pybullet_drones/results/hover/multienv/' 
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
        radius_transition= []
        for point in points*r:
            sucess, transition = fly_to_point(point, env=env, agent=agent)
            sucess_num += sucess
            radius_transition.append(transition)
        data[r.item()] = radius_transition
        print("Radius ", r, "success rate", sucess_num/len(points))
    
    json_path = os.path.join(
        savedir, 'data.json'
    )
    with open(json_path, 'w') as f:
        json.dump(data, f)



if __name__=='__main__':
    main()