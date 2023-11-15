from gym_pybullet_drones.devices import Device
from gym_pybullet_drones.vehicles import QuadCopter
import numpy as np
import pybullet as pb
from typing import List, Tuple
import numpy as np


class Camera(Device):
    def __init__(self, image_size: List[int], fov: int, displ: np.ndarray, frequency: float, base: QuadCopter =None) -> None:
        '''
            image_size in W, H 
            fov(Field of view) assumed to be vertical in degrees
            displ

        '''
        super().__init__(frequency, base)
        self.image_size = image_size
        self.fov = fov
        self.displ = displ
        # https://stackoverflow.com/questions/60430958/understanding-the-view-and-projection-matrix-from-pybullet
        self.aspect_ratio = image_size[0]/image_size[1]
         


    def make_obs(self) -> Tuple(np.ndarray):

        pos, quat = pb.getBasePositionAndOrientation(
            self.base.ID, physicsClientId=self.base.client)

        rot_mat = np.array(
            pb.getMatrixFromQuaternion(quat)
        ).reshape(3, 3)
        #### Set target point, camera view and projection matrices #
        target = np.dot(rot_mat,np.array([1000, 0, 0])) + np.array(pos)
        
        view_mat = pb.computeViewMatrix(
            cameraEyePosition=pos+np.dot(rot_mat,self.displ), #np.array([0, 0, self.L])
            cameraTargetPosition=target,
            cameraUpVector=np.dot(rot_mat,np.array([0, 0, 1])),
            physicsClientId=self.base.client
        )

        proj_mat = pb.computeProjectionMatrixFOV(
            fov=self.fov,
            aspect=self.aspect_ratio,
            nearVal=0.1,
            farVal=1000.0
        )

        SEG_FLAG = pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX# if segmentation else p.ER_NO_SEGMENTATION_MASK
        [w, h, rgb, dep, seg] = pb.getCameraImage(
            width=self.image_size[0],
            height=self.image_size[1],
            shadow=1,
            viewMatrix=view_mat,
            projectionMatrix=proj_mat,
            flags=SEG_FLAG,
            physicsClientId=self.base.client
        )

        rgb = np.reshape(rgb, (self.image_size[1], self.image_size[0], 4))
        dep = np.reshape(dep, (self.image_size[1], self.image_size[0]))
        seg = np.reshape(seg, (self.image_size[1], self.image_size[0]))

        return (rgb, seg, dep)