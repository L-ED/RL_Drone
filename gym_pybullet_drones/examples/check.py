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
        # Camera(
        #     image_size = [640, 480],
        #     fov= 60,
        #     displ = np.array([0, 0, 0.1]),
        #     frequency = 60
        # ),
        mpu6000()
        ]

    state = State()
    state.world.pos[2] += 1.0
    # state.world.rpy[1] = 1.57

    drone = QuadCopter(
        client=client,
        filename= 'custom.urdf',
        #filename='cf2x_cam.urdf',
        # sensors = [],
        sensors = sensors,
        state=state
    )

    env = Base(
        client=client,
        drone=drone,
        control_system=None
    )

    # camera_vis_id = pb.createVisualShape(pb.GEOM_BOX,
    #                                      halfExtents=[0.02, 0.02, 0.02],
    #                                      rgbaColor=[1,0,0,0.7])
    # camera_body = pb.createMultiBody(0, -1, camera_vis_id)

    # START = time.time()
    print(env.timestep)
    while True:
        _ = env.step(np.array([0,0,0,0]))

        # for i in range(4):
        pb.applyExternalForce(
            env.drone.ID,
            0,
            forceObj=[0, 0, 0.1],
            posObj=[0, 0, 0],
            flags=pb.LINK_FRAME,
            physicsClientId=client
        )
        pb.applyExternalForce(
            env.drone.ID,
            1,
            forceObj=[0, 0, 0.1],
            posObj=[0, 0, 0],
            flags=pb.LINK_FRAME,
            physicsClientId=client
        )
        env.render()
        time.sleep(env.timestep)
run()