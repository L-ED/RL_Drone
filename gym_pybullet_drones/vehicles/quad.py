from dataclasses import dataclass, InitVar, field
from typing import Union, List
import pkg_resources 

import numpy as np

from gym_pybullet_drones.utils import State, Device, Control_System

import pybullet as pb
from gymnasium import spaces


@dataclass
class QuadCopter:
    client: int
    state: State
    filename: str = None # maybe in post init
    sensors: List[Device] = field(default_factory=list)
    control_system: Control_System = None 
    ID: int = None


    def __post_init__(self):
        
        self._parseURDF()
        actionSpace = self.load_model()
        obsSpace = self.load_sensors()
        self.set_initial_state()
        self.control_system.set_base(self)

        return obsSpace, actionSpace


    def _parseURDF(self):
        raise NotImplementedError("No URDF parse in Default class")


    def load_model(self, state):

        filepath = pkg_resources.resource_filename(
            'gym_pybullet_drones', 'assets/'+self.filename
        )

        if self.source_file.endswith('urdf'):
            self.ID = pb.loadURDF(
                filepath,
                self.state.pos,
                self.state.quat, #pb.getQuaternionFromEuler(self.INIT_RPYS),
                flags = (
                    pb.URDF_USE_INERTIA_FROM_FILE | \
                    pb.URDF_MERGE_FIXED_LINKS | \
                    pb.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
                ),
                physicsClientId=self.client
            )

        elif self.source_file.endswith('sdf'):
            self.ID = pb.loadSDF(
                filepath,
                physicsClientId=self.client
            )
            pb.resetBasePositionAndOrientation(
                self.ID,
                self.state.pos,
                self.state.quat,
                physicsClientId=self.client
            )

        else:
            raise ValueError(f"No loader for file {self.source_file}")


    def set_initial_state(self):

        # Set initial velocity https://pybullet.org/Bullet/phpBB3/viewtopic.php?t=11793
        pb.resetBaseVelocity(
            self.ID, # model
            self.state.lin_vel, # linear velocity
            self.state.ang_vel  # angular velocity
        )

        pb.applyExternalForce(
            self.ID,
            4, # base link
            forceObj=self.state.force,
            posObj=self.state.pos,
            flags=pb.WORLD_FRAME, #
            physicsClientId=self.client
        )

        pb.applyExternalTorque(
            self.ID,
            4, # base link
            torqueObj=self.state.torque,
            flags=pb.LINK_FRAME,
            physicsClientId=self.client
        )


    def load_sensors(self):
        '''
        Parse sensors configs from URDF or SDF file and add to sensors list
        '''
        observation_space = {}
        sensors_counter = {}
        for sensor in self.sensors:
            sensor.set_base(self.ID)
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
        return self.control_system(self._observation)


    def take_dev_freqs(self):

        self.freqs = set([
            sensor.freq for sensor in self.sensors 
        ])

        self.freqs.add(self.control_system.freq)


    def model(self, RPS, env):
        
        # X axis is direction of drone forward movement
        # Y axis directed from right to left side of copter
        # Z axis directed upward from drone botton 

        thrust = self.Ct*env.rho*RPS*self.prop_diam**4
        torque = self.Cq*env.rho*RPS*self.prop_diam**5 
        # CW rotation produces CCW torque
        # motors orientation means rotation direction CW = 1, CCW = -1
        
        moment_z = np.dot(self.motor_orientation, torque)      
        moment_x = self.ly*np.dot([-1, 1, -1, 1], thrust)
        moment_y = self.lx*np.dot([-1, -1, 1, 1], thrust)
        axes_torques = np.array([
            moment_x, moment_y, moment_z
        ])

        # https://en.wikipedia.org/wiki/Euler%27s_equations_(rigid_body_dynamics)
        angular_accel = np.dot(
            self.J_inv,
            axes_torques - np.cross(
                self.state.ang_vel,
                np.dot(self.J, self.state.ang_vel) 
            )
        )

        linear_accel = thrust/self.mass
        ang_vel += env.timestemp*angular_accel

        return linear_accel, ang_vel, thrust, torque, moment_z

        
    def apply_force(self, thrust, moment_z):
        for i in range(4):
            pb.applyExternalForce(self.ID,
                                 i,
                                 forceObj=[0, 0, thrust[i]],
                                 posObj=[0, 0, 0],
                                 flags=pb.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        pb.applyExternalTorque(
            self.ID,
            4,
            torqueObj=[0, 0, moment_z],
            flags=pb.LINK_FRAME,
            physicsClientId=self.CLIENT
        )







@dataclass
class GymCopter(QuadCopter):
    DRONE_MODEL: InitVar[str]
    G: float
    M: float = None
    L: float = None
    THRUST2WEIGHT_RATIO: float = None
    J: float = None
    J_INV: float = None
    KF: float = None
    KM: float = None 
    COLLISION_H: float = None
    COLLISION_R: float = None
    COLLISION_Z_OFFSET: float = None
    MAX_SPEED_KMH: float = None
    GND_EFF_COEFF: float = None
    PROP_RADIUS: float = None
    DRAG_COEFF: float = None
    DW_COEFF_1: float = None
    DW_COEFF_2: float = None
    DW_COEFF_3: float = None

    GRAVITY: float = None
    HOVER_RPM: float = None
    MAX_RPM: float = None
    MAX_THRUST: float = None
    MAX_XY_TORQUE: float = None
    MAX_Z_TORQUE: float = None
    GND_EFF_H_CLIP: float = None


    def _parseURDF(self):

        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        self.M = float(URDF_TREE[1][0][1].attrib['value'])
        self.L = float(URDF_TREE[0].attrib['arm'])
        self.THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        self.J = np.diag([IXX, IYY, IZZ])
        self.J_INV = np.linalg.inv(self.J)
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
            
        if self.DRONE_MODEL in [DroneModel.CF2X, DroneModel.RACE]:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        

        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)


        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)

    #### Create attributes for vision tasks ####################
    # if self.RECORD:
    #     self.ONBOARD_IMG_PATH = os.path.join(self.OUTPUT_FOLDER, "recording_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    #     os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH), exist_ok=True)
    # self.VISION_ATTR = vision_attributes
    # if self.VISION_ATTR:
    #     if self.IMG_RES is None:
    #         self.IMG_RES = np.array([160, 120])
    #     self.IMG_FRAME_PER_SEC = 24
    #     self.IMG_CAPTURE_FREQ = int(self.PYB_FREQ/self.IMG_FRAME_PER_SEC)
    #     self.rgb = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4)))
    #     self.dep = np.ones(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
    #     self.seg = np.zeros(((self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0])))
    #     if self.IMG_CAPTURE_FREQ%self.PYB_STEPS_PER_CTRL != 0:
    #         print("[ERROR] in BaseAviary.__init__(), PyBullet and control frequencies incompatible with the desired video capture frame rate ({:f}Hz)".format(self.IMG_FRAME_PER_SEC))
    #         exit()
    #     if self.RECORD:
    #         for i in range(self.NUM_DRONES):
    #             os.makedirs(os.path.dirname(self.ONBOARD_IMG_PATH+"/drone_"+str(i)+"/"), exist_ok=True)

    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        self.M = float(URDF_TREE[1][0][1].attrib['value'])
        self.L = float(URDF_TREE[0].attrib['arm'])
        self.THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        self.J = np.diag([IXX, IYY, IZZ])
        self.J_INV = np.linalg.inv(J)
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