from typing import Annotated
from numpy.typing import NDArray
import numpy as np
from mafs import *
from typing import List

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

"""
Helpers
"""
# Given a list of fisix things, figure out how they all affect each other and update their forces
def apply_grav(things: List[fizix_thing], G: float):
    n_things = len(things)
    for cur_thing in range(n_things):
        # get the gravity force of everything on you and sum it all up.
        f_grav_g_net = np.array([0, 0, 0])
        for other_thing in range(n_things):
            # don't calculate gravity wrt. yourself!
            if cur_thing != other_thing:
                # get some vectors
                r_cur2other = things[other_thing].r_g2p_g - things[cur_thing].r_g2p_g
                r_cur2other_mag = np.linalg.norm(r_cur2other)
                r_cur2other_hat = r_cur2other / r_cur2other_mag

                # calculate force magnitude
                f_mag = (
                    G
                    * (things[cur_thing].M * things[other_thing].M)
                    / r_cur2other_mag**2
                )
                f_vec_g = f_mag * r_cur2other_hat

                # add up
                f_grav_g_net = f_grav_g_net + f_vec_g
        # rotate force into body frame, and remember it.
        f_grav_b_net = quatrotate(things[cur_thing].q_g2b, f_grav_g_net)
        things[cur_thing].update_forces(f_grav_b_net, np.array([0, 0, 0]))

    return things