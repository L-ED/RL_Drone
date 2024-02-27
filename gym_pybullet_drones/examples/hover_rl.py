from stable_baselines3 import PPO
from gym_pybullet_drones.envs.single_agent_rl import HoverIMU
import time

def main(test=True):

    env = HoverIMU()
    agent = PPO(
        'MlpPolicy', 
        env=env,
        verbose=1
    )
    # agent.learn(30000)
    env = HoverIMU(visualize=True)

    state, _=env.reset()
    while test:
        action, _ = agent.predict(
            state.reshape(1,-1),
            deterministic=True
        )
        state, reward, terminated, truncated, info = env.step(action)
        print(info['timestemp'])

        time.sleep(env.timestep)
        if terminated or truncated:
            state, _=env.reset()


if __name__=='__main__':
    main()