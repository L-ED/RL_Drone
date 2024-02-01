from gym_pybullet_drones.devices import Device
# from gym_pybullet_drones.vehicles.quad import QuadCopter
import numpy as np
import pybullet as pb
from typing import List, Tuple
import numpy as np
from gymnasium import spaces

from matplotlib import pyplot as plt


class Camera(Device):
    name_base = 'camera'

    def __init__(self, image_size: List[int], fov: int, displ: np.ndarray, frequency: float, base = None) -> None:
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
         
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=-np.inf, high=np.inf, shape=(self.image_size[1], self.image_size[0], 4)),
                spaces.Box(low=-np.inf, high=np.inf, shape=(self.image_size[1], self.image_size[0])),
                spaces.Box(low=-np.inf, high=np.inf, shape=(self.image_size[1], self.image_size[0]))
            )
        )

        plt.figure()
        self.plt_im = plt.imshow(np.zeros((image_size[1],image_size[0],4)))
        plt.axis('off')
        plt.tight_layout(pad=0)


    def make_obs(self) -> Tuple[np.ndarray]:

        # pos, quat = pb.getBasePositionAndOrientation(
        #     self.base.ID, physicsClientId=self.base.client)

        pos = self._base.state.world.pos
        quat = self._base.state.world.qtr

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
            nearVal=0.01,
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

        self.plt_im.set_array(rgb)
        plt.gca().set_aspect(self.image_size[1]/self.image_size[0])
        plt.draw()

        return (rgb, seg, dep)
    

    def visualize(self):
        print("CAM VIS")

        # camera_vis_id = pb.createVisualShape(pb.GEOM_BOX,
        #                                  halfExtents=[0.02, 0.02, 0.02],
        #                                  rgbaColor=[1,0,0,0.7])
        # camera_body = pb.createMultiBody(0, -1, camera_vis_id)

        # to_axs = self.displ
        # to_axs[0] += 10
        # ax_idx = pb.addUserDebugLine(
        #     lineFromXYZ=self.displ,
        #     lineToXYZ=to_axs,
        #     lineColorRGB=[1, 0, 0],
        #     parentObjectUniqueId=self._base.ID,
        #     parentLinkIndex=-1,
        #     replaceItemUniqueId=-1,
        #     physicsClientId=self._base.client
        # )

        # ax_idx = pb.addUserDebugLine(
        #     lineFromXYZ=[0, 0, 0.02],
        #     lineToXYZ=[0.04, 0, 0.04],
        #     lineColorRGB=[1, 0, 1],
        #     parentObjectUniqueId=self._base.ID,
        #     parentLinkIndex=-1,
        #     replaceItemUniqueId=-1,
        #     physicsClientId=self._base.client
        # )