from dataclasses import dataclass, InitVar, field
from typing import Union, List
import pkg_resources 

import numpy as np

from gym_pybullet_drones.utils import State
from gym_pybullet_drones.devices import Device, Control_System

import pybullet as pb
from gymnasium import spaces

from gym_pybullet_drones.utils.enums import DroneModel

import copy


class QuadCopter:

    def __init__(
            self, 
            client: int,  
            filename: str, 
            sensors: List[Device],
            state: State=None, 
            control_system: Control_System = None
        ):

        self.filepath = pkg_resources.resource_filename(
            'gym_pybullet_drones', 'assets/'+self.filename
        )
        
        self.client = client
        if state is None:
            self.state = State()
        self._parseURDF()


        actionSpace = spaces.Box(
            low=np.zeros(4),
            high=np.ones(4),
            dtype=np.float32
        )
        
        self.load_model(filename)
        obsSpace = self.load_sensors(sensors)
        self.set_initial_state()

        self.control_system = control_system
        self.control_system.set_base(self)

        return obsSpace, actionSpace


    def _parseURDF(self):
        raise NotImplementedError("No URDF parse in Default class")


    def load_model(self):

        state = self.state.world

        if self.filepath.endswith('urdf'):
            self.ID = pb.loadURDF(
                self.filepath,
                state.pos,
                pb.getQuaternionFromEuler(
                    state.ang
                ), #pb.getQuaternionFromEuler(self.INIT_RPYS),
                flags = (
                    pb.URDF_USE_INERTIA_FROM_FILE | \
                    pb.URDF_MERGE_FIXED_LINKS | \
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
                pb.getQuaternionFromEuler(state.ang),
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


    def load_sensors(self):
        '''
        Parse sensors configs from URDF or SDF file and add to sensors list
        '''
        observation_space = {}
        sensors_counter = {}
        for sensor in self.sensors:
            sensor.set_base(self)
            name = sensor.name_base + "_" +\
                str(sensors_counter[sensor.name_base])
            sensor.set_name(name)

            observation_space[sensor.name].append(sensor.observation_space)

        return spaces.Dict(observation_space)
    
    
    def compute_observation(self, timestemp):

        observation = {}
        for sensor in self.sensors:
            observation[sensor.name] = sensor(timestemp)

        self._observation = observation
        return observation
    

    def compute_action(self, timestemp):
        return self.control_system(self._observation, timestemp)


    def take_dev_freqs(self):

        self.freqs = set([
            sensor.freq for sensor in self.sensors 
        ])

        self.freqs.add(self.control_system.freq)


    def model(self, RPS, env, state=None):
        
        if state is None:
            state = copy.deepcopy(self.state)
        
        future_state = State()
        # X axis is direction of drone forward movement
        # Y axis directed from right to left side of copter
        # Z axis directed upward from drone botton 

        thrust = self.Ct*env.rho*RPS*self.prop_diam**4
        torque = self.Cq*env.rho*RPS*self.prop_diam**5 
        # CW rotation produces CCW torque
        # motors orientation means rotation direction CW = 1, CCW = -1
        future_state.local.forces = np.array([0, 0, sum(thrust)])

        moment_z = np.dot(self.motor_orientation, torque)      
        moment_x = self.ly*np.dot([-1, 1, -1, 1], thrust)
        moment_y = self.lx*np.dot([-1, -1, 1, 1], thrust)
        axes_torques = np.array([
            moment_x, moment_y, moment_z
        ])
        future_state.local.torques = axes_torques

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

        future_state.local.ang_vel = state.local.ang_vel + angular_accel*env.TIME_STEP
        future_state.local.vel = state.local.vel + np.array([0, 0, sum(thrust)/self.mass])*env.TIME_STEP

        return  future_state

        
    def apply_force(self, state):

        for i in range(4):
            pb.applyExternalForce(self.ID,
                                 i,
                                 forceObj=state.local.forces,
                                 posObj=[0, 0, 0],
                                 flags=pb.LINK_FRAME,
                                 physicsClientId=self.client
                                 )
        pb.applyExternalTorque(
            self.ID,
            4,
            torqueObj=state.local.torques,
            flags=pb.LINK_FRAME,
            physicsClientId=self.client
        )

    
    def create_step(self, env):

        obs = self.compute_observation(env.timestemp)
        act = self.compute_action(env.timestemp)
        future_state = self.model(act, env)
        self.apply_force(future_state)
        self.update_state(future_state)


    def update_state(self, new_state):
        self.state.local.forces = new_state.local.forces
        self.state.local.torques = new_state.local.torques
        self.state.local.ang_acc = new_state.local.ang_acc
        self.state.local.acc = new_state.local.acc 

        lin_vel, ang_vel = pb.getBaseVelocity(self.ID, self.client)
        pos, qtr = pb.getBasePositionAndOrientation(self.ID, self.client)

        self.state.world.vel = lin_vel
        self.state.world.ang_vel = ang_vel
        self.state.world.pos = pos
        self.state.world.qtr = qtr

        self.state.create_R_matrix()

        local_lin_vel = np.dot(
            self.state.R.T,
            lin_vel
        )

        self.state.local.vel = new_state.local.vel
        self.state.local.ang_vel = new_state.local.ang_vel


        


        



class GymCopter(QuadCopter):

    def __init__(self, client: int, drone_model: DroneModel, sensors: List, state: State = None, control_system: Control_System = None):
        self.drone_model = drone_model
        filename = self.drone_model.value + ".urdf"
        super().__init__(client, filename, sensors, state, control_system)


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


        future_state.local.forces = np.array([0, 0, np.sum(forces)])

        future_state.local.torques = np.array(
            [x_torque, y_torque, z_torque])

        future_state.local.ang_acc = np.dot(
            self.J_inv,
            future_state.local.torques - np.cross(
                state.local.ang_vel,
                np.dot(self.J, state.local.ang_vel) 
            ))

        future_state.local.acc = np.array(
            [0, 0, np.sum(forces)/self.mass])
        
        future_state.local.ang_vel += env.timestemp*future_state.local.ang_acc


        return future_state #linear_accel, ang_vel, thrust, torques, z_torque
