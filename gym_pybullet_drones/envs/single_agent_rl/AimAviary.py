
import numpy as np
from gymnasium import spaces
import pybullet as p
import os 
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType, BaseSingleAgentAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class AimAviary(BaseAviary):
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.INIT_XYZS = np.array([
            [0, 0, 10]
        ])
        self.IMG_RES = np.array([160, 120])

        self.target_pose = np.array([30.5, -2.5, .5])
        self.targ_cls = 67108867
        
        self.targ_pitch=0
        self.targ_yaw=0

        self.h=self.IMG_RES[1]
        self.w=self.IMG_RES[0]
        self.hfov=60*np.pi/180
        self.f = self.w/(2*np.tan(self.hfov/2))
        self.vfov = 2*np.arctan(self.h/(2*self.f))
        self.fov=np.array(
            [self.vfov, self.hfov]
        )

        vision_attributes = True
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.EPISODE_LEN_SEC = 10
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
                self.ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
            else:
                print("[ERROR] in BaseSingleAgentAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)

        self._addObstacles()
    ################################################################################
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        state = self._getDroneStateVector(0)

        pos = state[0:3]
        rpy = state[7:10]
        vel = state[10:13]
        ang_vel = state[13:16]

        diff = self.target_pose - pos
        targ_dir = diff/np.linalg.norm(diff)
        flight_dir = vel/np.linalg.norm(vel)
        vel_mag = np.sum(vel**2)**0.5
        flight_cos = np.dot(targ_dir, flight_dir)
        diff = np.sum(diff**2)**0.5

        ground_hit = (pos[2]<0.1)*(0.1 - pos[2])

        # rot_mtrx = np.array(
        #     p.getMatrixFromQuaternion(state[3:7])
        # ).reshape((3,3))
        # cam_vec = np.dot(rot_mtrx, np.array([1, 0, 0]))
        # cam_dir = cam_vec/np.linalg.norm(cam_vec)
        # cam_cos = np.dot(targ_dir, cam_dir)
        # ang_reward = 0 if abs(cam_cos)<np.cos(np.pi/3) else 1
        rot_vel = np.sum((ang_vel/np.linalg.norm(ang_vel))**2)**0.5

        target_hit = (diff<2)*(2 - diff)

        aim_err = (((self.targ_pitch - np.pi/9)/self.fov[0])**2 + (self.targ_yaw/self.fov[0])**2)**0.5

        reward = flight_cos*vel_mag - 50*ground_hit - aim_err*vel_mag - rot_vel + target_hit*100
        print(
            "REWARD", reward,
            # "Dist_reward", diff, 
            "Flight_reward ", flight_cos*vel_mag,
            "Aim ", - aim_err*vel_mag,
            "Ground Hit ", - 50*ground_hit,
            "Rot vel", - rot_vel,
            "Target hit", target_hit*100

        )
        return reward
    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
        
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value(s).

        Unused in this implementation.

        Returns
        -------
        bool
            Always false.

        """
        return False

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years


    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
         
        self.rgb[0], self.dep[0], self.seg[0] = self._getDroneImages(0)

        shape = np.array(self.seg[0].shape[:2])
        targ_pixls = np.where(self.seg[0]==self.targ_cls)

        if len(targ_pixls[0])>0:
            targ_center = np.mean(np.where(self.seg[0]==self.targ_cls), axis=1)
            self.targ_pitch, self.targ_yaw = (targ_center - shape//2)/shape*self.fov
        else:
            # print("FOV is", self.fov)
            self.targ_pitch, self.targ_yaw = self.fov

        # print(self.targ_pitch, self.targ_yaw)
        
        obs = self._clipAndNormalizeState(self._getDroneStateVector(0))
        ret = np.hstack([obs[0:3], obs[7:10], obs[10:13], obs[13:16], self.targ_pitch/self.fov[0]*2, self.targ_yaw/self.fov[1]*2]).reshape(14,)
        return ret.astype('float32')



    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (H,W,4) or (12,) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.IMG_RES[1], self.IMG_RES[0], 4),
                              dtype=np.uint8
                              )
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            #### OBS OF SIZE 20 (WITH QUATERNION AND RPMS)
            #### Observation vector ### X        Y        Z       Q1   Q2   Q3   Q4   R       P       Y       VX       VY       VZ       WX       WY       WZ       P0            P1            P2            P3
            # obs_lower_bound = np.array([-1,      -1,      0,      -1,  -1,  -1,  -1,  -1,     -1,     -1,     -1,      -1,      -1,      -1,      -1,      -1,      -1,           -1,           -1,           -1])
            # obs_upper_bound = np.array([1,       1,       1,      1,   1,   1,   1,   1,      1,      1,      1,       1,       1,       1,       1,       1,       1,            1,            1,            1])          
            # return spaces.Box( low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32 )
            ############################################################
            #### OBS SPACE OF SIZE 14
            return spaces.Box(low=np.array([-1,-1,0, -1,-1,-1, -1,-1,-1, -1,-1,-1, -1,-1]),
                              high=np.array([1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1]),
                              dtype=np.float32
                              )
            ############################################################
        else:
            print("[ERROR] in BaseSingleAgentAviary._observationSpace()")


    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in HoverAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))


    def _addObstacles(self):
        """Add obstacles to the environment.

        These obstacles are loaded from standard URDF files included in Bullet.

        # """
        # p.loadURDF("samurai.urdf",
        #            physicsClientId=self.CLIENT
        #            )
        p.loadURDF("duck_vhacd.urdf",
                   [-.5, -.5, .05],
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        p.loadURDF("cube_no_rotation.urdf",
                   self.target_pose.tolist(),
                   p.getQuaternionFromEuler([0, 0, 0]),
                   physicsClientId=self.CLIENT
                   )
        p.loadURDF("sphere2.urdf",
                   [0, 2, .5],
                   p.getQuaternionFromEuler([0,0,0]),
                   physicsClientId=self.CLIENT
                   )
        

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, 4, or 6 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, new PID coefficients, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (4,)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        if self.ACT_TYPE == ActionType.RPM:
            return np.array(self.HOVER_RPM * (1+0.05*action))
        elif self.ACT_TYPE == ActionType.PID:
            state = self._getDroneStateVector(0)
            next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=action,
                    step_size=1,
                )
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=next_pos
                                                 )
            return rpm
        elif self.ACT_TYPE == ActionType.VEL:
            state = self._getDroneStateVector(0)
            if np.linalg.norm(action[0:3]) != 0:
                v_unit_vector = action[0:3] / np.linalg.norm(action[0:3])
            else:
                v_unit_vector = np.zeros(3)
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=state[0:3], # same as the current position
                                                 target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                 target_vel=self.SPEED_LIMIT * np.abs(action[3]) * v_unit_vector # target the desired velocity vector
                                                 )
            return rpm
        elif self.ACT_TYPE == ActionType.ONE_D_RPM:
            return np.repeat(self.HOVER_RPM * (1+0.05*action), 4)
        elif self.ACT_TYPE == ActionType.ONE_D_PID:
            state = self._getDroneStateVector(0)
            rpm, _, _ = self.ctrl.computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                 cur_pos=state[0:3],
                                                 cur_quat=state[3:7],
                                                 cur_vel=state[10:13],
                                                 cur_ang_vel=state[13:16],
                                                 target_pos=state[0:3]+0.1*np.array([0,0,action[0]])
                                                 )
            return rpm
        else:
            print("[ERROR] in BaseSingleAgentAviary._preprocessAction()")


    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        ndarray
            A Box() of size 1, 3, 4, or 6 depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE == ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseSingleAgentAviary._actionSpace()")
            exit()
        return spaces.Box(low=-1*np.ones(size),
        # return spaces.Box(low=np.zeros(size),  # Alternative action space, see PR #32
                          high=np.ones(size),
                          dtype=np.float32
                          )