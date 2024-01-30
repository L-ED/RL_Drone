import numpy as np
import pybullet as pb

from gym_pybullet_drones.vehicles import QuadCopter
from gym_pybullet_drones.devices import Camera, mpu6000
from gym_pybullet_drones.envs import Base
from gym_pybullet_drones.utils.state import State

from gym_pybullet_drones.utils.utils import sync, str2bool
import time


def run():
    client = pb.connect(pb.GUI)

    sensors= [
        Camera(
            image_size = [640, 480],
            fov= 60,
            displ = np.array([0, 0, 10]),
            frequency = 60
        ),
        mpu6000()
        ]

    state = State()
    state.world.pos[2] = 1

    drone = QuadCopter(
        client=client,
        filename= 'cf2x.urdf',
        # sensors = [],
        sensors = sensors,
        state=state
    )

    env = Base(
        client=client,
        drone=drone,
        control_system=None
    )

    # START = time.time()
    print(env.timestep)
    while True:
        _ = env.step(np.array([0,0,0,0]))
        env.render()
        time.sleep(env.timestep)
run()