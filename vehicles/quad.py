from dataclasses import dataclass, InitVar, field
from typing import Union, List
import pkg_resources 

import numpy as np

from gym_pybullet_drones.utils.state import State
from gym_pybullet_drones.devices import Device

import xml.etree.ElementTree as etxml

import pybullet as pb
from gymnasium import spaces

from gym_pybullet_drones.utils.enums import DroneModel

import copy


class QuadCopter:

    def __init__(
            self, 
            client: int,  
            filename: str, 
            sensors: List[Device]=[],
            state: State=None, 
        ):

        self.filepath = pkg_resources.resource_filename(
            'gym_pybullet_drones', 'assets/'+filename
        )

        self.motor_orientation=np.array([-1,1, 1, -1])
        
        self.client = client
        self.state = state
        if self.state is None:
            self.state = State()
        

        self._parseURDF()


        self.actionSpace = spaces.Box(
            low=np.zeros(4),
            high=np.ones(4),
            dtype=np.float32
        )
        
        # self.load_model(filename)
        self.obsSpace = self.load_sensors(sensors)


    def _parseURDF(self):
        urdf_tree = etxml.parse(self.filepath).getroot()
        self.mass = float(urdf_tree[1][0][1].attrib['value'])

        IXX = float(urdf_tree[1][0][2].attrib['ixx'])
        IYY = float(urdf_tree[1][0][2].attrib['iyy'])
        IZZ = float(urdf_tree[1][0][2].attrib['izz'])
        self.J = np.diag([IXX, IYY, IZZ])
        self.J_inv = np.linalg.inv(self.J)

        self.L = float(urdf_tree[0].attrib['arm'])

        self.lx, self.ly = [
            abs(float(x)) for x in 
            urdf_tree[2][0][0].attrib['xyz'].split()[:2]
        ]

        self.Ct = float(urdf_tree[0].attrib['kf'])
        self.Cq = float(urdf_tree[0].attrib['km'])
        self.COLLISION_H = float(urdf_tree[1][2][1][0].attrib['length'])
        self.COLLISION_R = float(urdf_tree[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in urdf_tree[1][2][0].attrib['xyz'].split(' ')]
        self.COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]

        self.GND_EFF_COEFF = float(urdf_tree[0].attrib['gnd_eff_coeff'])
        self.prop_diam = float(urdf_tree[0].attrib['prop_radius'])*2
        DRAG_COEFF_XY = float(urdf_tree[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(urdf_tree[0].attrib['drag_coeff_z'])
        self.DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        self.DW_COEFF_1 = float(urdf_tree[0].attrib['dw_coeff_1'])
        self.DW_COEFF_2 = float(urdf_tree[0].attrib['dw_coeff_2'])
        self.DW_COEFF_3 = float(urdf_tree[0].attrib['dw_coeff_3'])

        self.THRUST2WEIGHT_RATIO = float(urdf_tree[0].attrib['thrust2weight'])
        self.max_rps = np.sqrt((self.THRUST2WEIGHT_RATIO*9.8) / (4*self.Ct))/60

    def load_model(self):

        state = self.state.world

        if self.filepath.endswith('urdf'):
            self.ID = pb.loadURDF(
                self.filepath,
                state.pos,
                pb.getQuaternionFromEuler(
                    state.rpy
                ), #pb.getQuaternionFromEuler(self.INIT_RPYS),
                flags = (
                    pb.URDF_USE_INERTIA_FROM_FILE | \
                    # pb.URDF_MERGE_FIXED_LINKS | \
                    pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                ),
                physicsClientId=self.client
            )

        elif self.filepath.endswith('sdf'):
            self.ID = pb.loadSDF(
                self.filepath,
                physicsClientId=self.client
            )
            pb.resetBasePositionAndOrientation(
                self.ID,
                state.pos,
                pb.getQuaternionFromEuler(state.rpy),
                physicsClientId=self.client
            )

        else:
            raise ValueError(f"No loader for file {self.filepath}")


    def set_initial_state(self):

        state = self.state.world

        # Set initial velocity https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=11793
        pb.resetBaseVelocity(
            self.ID, # model
            state.lin_vel, # linear velocity
            state.ang_vel  # angular velocity
        )

        pb.applyExternalForce(
            self.ID,
            4, # base link
            forceObj=state.force,
            posObj=state.pos,
            flags=pb.WORLD_FRAME, #
            physicsClientId=self.client
        )

        pb.applyExternalTorque(
            self.ID,
            4, # base link
            torqueObj=state.torque,
            flags=pb.WORLD_FRAME,
            physicsClientId=self.client
        )

    def reset_state(self, state):

        self.state = state

        pb.resetBasePositionAndOrientation(
                self.ID,
                state.world.pos,
                pb.getQuaternionFromEuler(state.world.ang),
                physicsClientId=self.client
            )
        self.set_initial_state()


    def load_sensors(self, sensors):
        '''
        Parse sensors configs from URDF or SDF file and add to sensors list
        '''
        self.sensors = sensors
        observation_space = {}
        sensors_counter = {}
        for sensor in sensors:
            sensor.base=self
            basename = sensor.name_base
            if basename not in sensors_counter:
                sensors_counter[basename]=0
            else:
                sensors_counter[basename]+=1

            name = sensor.name_base + "_" +\
                str(sensors_counter[basename])
            sensor.name=name

            observation_space[sensor.name]=sensor.observation_space

        self.obsSpace = spaces.Dict(observation_space)
    
    
    def compute_observation(self, timestemp):

        observation = {}
        for sensor in self.sensors:
            observation[sensor.name] = sensor(timestemp)

        self._observation = observation
        return observation


    def take_dev_freqs(self):

        return list(set([
            sensor.freq for sensor in self.sensors 
        ]))


    def model(self, RPS, env, state=None):
        
        if state is None:
            state = copy.deepcopy(self.state)
        
        future_state = State()
        # X axis is direction of drone forward movement
        # Y axis directed from right to left side of copter
        # Z axis directed upward from drone botton 
        RPS = np.clip(RPS, 0 ,self.max_rps)

        thrust = self.Ct*env.rho*RPS*self.prop_diam**4
        torque = self.Cq*env.rho*RPS*self.prop_diam**5 
        # CW rotation produces CCW torque
        # motors orientation means rotation direction CW = 1, CCW = -1
        future_state.local.force = np.array([0, 0, sum(thrust)])

        moment_z = np.dot(self.motor_orientation, torque)      
        moment_x = self.ly*np.dot([-1, 1, -1, 1], thrust)
        moment_y = self.lx*np.dot([-1, -1, 1, 1], thrust)
        axes_torques = np.array([
            moment_x, moment_y, moment_z
        ])
        future_state.local.torque = axes_torques

        # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        angular_accel = np.dot(
            self.J_inv,
            axes_torques - np.cross(
                self.state.local.ang_vel,
                np.dot(self.J, self.state.local.ang_vel) 
            )
        )

        future_state.local.ang_acc = angular_accel
        future_state.local.acc = np.array([0, 0, sum(thrust)/self.mass])

        future_state.local.ang_vel = state.local.ang_vel + angular_accel*env.timestep
        future_state.local.vel = state.local.vel + np.array([0, 0, sum(thrust)/self.mass])*env.timestep

        return  future_state

        
    def apply_force(self, state):

        # pass
        for i in range(4):
            pb.applyExternalForce(self.ID,
                                 i,
                                 forceObj=state.local.force.tolist(),
                                 posObj=[0, 0, 0],
                                 flags=pb.LINK_FRAME,
                                 physicsClientId=self.client
                                 )
        pb.applyExternalTorque(
            self.ID,
            4,
            torqueObj=state.local.torque,
            flags=pb.LINK_FRAME,
            physicsClientId=self.client
        )

    
    def step(self, act, env):

        obs = self.compute_observation(env.timestemp)
        future_state = self.model(act, env)
        self.apply_force(future_state)
        self.update_state(future_state)
        return obs


    def update_state(self, new_state):
        self.state.local.force = new_state.local.force
        self.state.local.torque = new_state.local.torque
        self.state.local.ang_acc = new_state.local.ang_acc
        self.state.local.acc = new_state.local.acc 

        lin_vel, ang_vel = pb.getBaseVelocity(self.ID, self.client)
        pos, qtr = pb.getBasePositionAndOrientation(self.ID, self.client)

        self.state.world.vel = lin_vel
        self.state.world.ang_vel = ang_vel
        self.state.world.pos = pos
        self.state.world.qtr = qtr

        local_lin_vel = np.dot(
            self.state.R.T,
            lin_vel
        )

        self.state.local.vel = new_state.local.vel
        self.state.local.ang_vel = new_state.local.ang_vel


    def set_sensor_ticks(self, env_freq):
        for sensor in self.sensors:
            sensor.set_tick(env_freq)


    def visualize_sensors(self):
        for sensor in self.sensors:
            sensor.visualize()



class GymCopter(QuadCopter):

    def __init__(self, client: int, drone_model: DroneModel, sensors: List, state: State = None):
        self.drone_model = drone_model
        filename = self.drone_model.value + ".urdf"
        super().__init__(client, filename, sensors, state)


    def _parseURDF(self):

        URDF_TREE = etxml.parse(self.filepath).getroot()
        self.M = float(URDF_TREE[1][0][1].attrib['value'])
        self.L = float(URDF_TREE[0].attrib['arm'])
        self.THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        self.J = np.diag([IXX, IYY, IZZ])
        self.J_inv = np.linalg.inv(self.J)
        self.KF = float(URDF_TREE[0].attrib['kf'])
        self.KM = float(URDF_TREE[0].attrib['km'])
        self.COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        self.COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        self.COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        self.MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        self.GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        self.PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        self.DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        self.DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        self.DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        self.DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])

        self.GRAVITY = self.G*self.M

        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
            
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
            
        if self.drone_model in [DroneModel.CF2X, DroneModel.RACE]:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.drone_model == DroneModel.CF2P:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)

        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)


    def model(self, RPS, env, state=None):

        if state is None:
            state = copy.deepcopy(self.state)
        
        future_state = State()

        rpm = RPS*60
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        if self.drone_model == DroneModel.RACE:
            torques = -torques
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])

        if self.DRONE_MODEL in [DroneModel.CF2X, DroneModel.RACE]:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L/np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L/np.sqrt(2))
        elif self.DRONE_MODEL==DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L


        future_state.local.force = np.array([0, 0, np.sum(forces)])

        future_state.local.torque = np.array(
            [x_torque, y_torque, z_torque])

        future_state.local.ang_acc = np.dot(
            self.J_inv,
            future_state.local.torque - np.cross(
                state.local.ang_vel,
                np.dot(self.J, state.local.ang_vel) 
            ))

        future_state.local.acc = np.array(
            [0, 0, np.sum(forces)/self.mass])
        
        future_state.local.ang_vel += env.timestemp*future_state.local.ang_acc


        return future_state #linear_accel, ang_vel, thrust, torques, z_torque


# def get_default():
#     return 