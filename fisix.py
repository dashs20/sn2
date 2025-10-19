from typing import Annotated
from numpy.typing import NDArray
import numpy as np
from mafs import *

M3x3 = Annotated[NDArray[np.float64], (3, 3)]

"""define objects"""


# thing
class fizix_thing:
    def __init__(
        self,
        r_g2p_g: Vec3,
        v_g2p_g: Vec3,
        a_g2p_g: Vec3,
        q_g2b: Quat,
        ome_g2b: Vec3,
        alp_g2b: Vec3,
        M: float,
        I: M3x3,
        force_b: Vec3,
        momen_b: Vec3,
        dt: float,
    ):
        self.r_g2p_g = r_g2p_g  # global position.
        self.v_g2p_g = v_g2p_g  # global velocity.
        self.a_g2p_g = a_g2p_g  # global acceleration.
        self.q_g2b = q_g2b  # orientation of body in global frame.
        self.ome_g2b = ome_g2b  # ang vel of body wrt. global frame.
        self.alp_g2b = alp_g2b  # ang accel of body wrt. global frame.
        self.M = M  # mass of body.
        self.I = I  # inertia of body.
        self.I_inv = np.linalg.inv(I)  # inverse of inertia matrix.
        self.force_b = force_b  # force on body, in body frame.
        self.momen_b = momen_b  # moment on body, in body frame.
        self.dt = dt  # timestep

    # update the force and moment on the thing
    def update_forces(self, force_b: Vec3, momen_b: Vec3):
        self.force_b = force_b
        self.momen_b = momen_b

    # step the thing through time
    def step(self):
        # A: TRANSLATIONAL PHYSICS

        # 1) rotate force into global frame
        force_g = quatrotate(quatinv(self.q_g2b), self.force_b)
        # 2) calculate acceleration
        self.a_g2p_g = force_g / self.M
        # 3) Eulelr's method, integrate vel and pos
        self.v_g2p_g = self.v_g2p_g + self.a_g2p_g * self.dt
        self.r_g2p_g = self.r_g2p_g + self.v_g2p_g * self.dt

        # B: ROTATIONAL PHYSICS

        # 1) calculate alpha
        self.alp_g2b = self.I_inv @ (
            self.momen_b - np.cross(self.ome_g2b, self.I @ self.ome_g2b)
        )
        # 2) Euler's method, integrate omega
        self.ome_g2b = self.ome_g2b + self.alp_g2b * self.dt
        # 3) Get q-dot using quaternion kinematic equation
        omega_quat = np.array(
            [0.0, self.ome_g2b[0], self.ome_g2b[1], self.ome_g2b[2]], dtype=float
        )
        q_g2b_dot = 0.5 * quatmultiply(self.q_g2b, omega_quat)
        # 4) Euler's method, integrate quat
        self.q_g2b = self.q_g2b + q_g2b_dot * self.dt
        self.q_g2b = self.q_g2b / np.linalg.norm(self.q_g2b)
