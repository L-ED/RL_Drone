from stable_baselines3 import PPO, SAC
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU, HoverGPS
import time
import torch
import os

def main(test=True):

    savepath= os.path.join(
        '/home/led/Simulators/Bullet/gym-pybullet-drones/gym_pybullet_drones/results',
        'best'
    )

    # env_class = HoverIMU
    env_class = HoverGPS

    env = env_class()
    agent = PPO(
        'MlpPolicy', 
        env=env,
        verbose=1,
        # n_steps=2000
    )
    # agent = SAC(
    #     'MlpPolicy', 
    #     env=env,
    #     verbose=1,
    #     # n_steps=8000
    # )

    agent.learn(500000)
    agent.save(savepath)
    env = env_class(visualize=True)
    agent = PPO.load(savepath, env=env)

    state, _=env.reset()
    rew = 0
    while test:
        action, _ = agent.predict(
            state.reshape(1,-1),
            deterministic=True
        )
        state, reward, terminated, truncated, info = env.step(action)
        # print(info['timestemp'], reward)
        rew+=reward

        time.sleep(env.timestep)
        if terminated or truncated:
            print(rew)
            rew=0
            state, _=env.reset()


if __name__=='__main__':
    main()