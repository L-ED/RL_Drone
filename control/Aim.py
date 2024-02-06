import math
import numpy as np
import pybullet as p


class Aim:

    def __init__(self, h=160, w=120, vfov=60) -> None:
        self.h=h
        self.w=w
        self.hfov=60
        self.f = w/(2*np.tan(self.hfov/2))
        self.vfov = np.arctan(h/(2*self.f))
        self.fov=np.array(
            [self.vfov, self.hfov]
        )
        self.prev_state = None

    def step(self, timestemp, segm, targ_cls, cur_pos, cur_quat, vertical_speed):

        shape = np.array(segm.shape[:2])
        targ_center = np.mean(np.where(segm==targ_cls), axis=1)
        pitch, yaw = (targ_center - shape//2)/shape*self.fov

        pitch+= 20

        print("Agnle correction Pitch:", pitch, " Yaw: ", yaw)


        target_rpy = np.array(p.getEulerFromQuaternion(cur_quat))
        target_rpy[1]+=pitch
        target_rpy[2]+=yaw
        rot_matrix = np.array(p.getMatrixFromQuaternion(
            p.getQuaternionFromEuler(target_rpy)
        )).reshape((3,3))
        target_speed = np.dot(rot_matrix, np.array([0, 0, vertical_speed]).T)
        target_pos = cur_pos + target_speed*timestemp*0.8

        return target_pos, target_rpy
