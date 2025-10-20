from typing import Annotated
from numpy.typing import NDArray
import numpy as np

Quat = Annotated[NDArray[np.float64], (4,)]
Vec3 = Annotated[NDArray[np.float64], (3,)]

"""math helpers"""


def quatinv(q: Quat):
    # get quaternion conjugate
    q = -q
    q[0] = -q[0]
    # get inverse (conj/norm^2)
    q = q / np.sum(q**2)
    return q


def quatmultiply(q: Quat, r: Quat):
    # get hamiltonian product
    t0 = r[0] * q[0] - r[1] * q[1] - r[2] * q[2] - r[3] * q[3]
    t1 = r[0] * q[1] + r[1] * q[0] - r[2] * q[3] + r[3] * q[2]
    t2 = r[0] * q[2] + r[1] * q[3] + r[2] * q[0] - r[3] * q[1]
    t3 = r[0] * q[3] - r[1] * q[2] + r[2] * q[1] + r[3] * q[0]
    # return
    return np.array([t0, t1, t2, t3])


def quatrotate(q: Quat, v: Vec3):
    # get quaternion rotation matrix
    R = np.array(
        [
            [
                1 - 2 * (q[2] ** 2 + q[3] ** 2),
                2 * (q[1] * q[2] + q[0] * q[3]),
                2 * (q[1] * q[3] - q[0] * q[2]),
            ],
            [
                2 * (q[1] * q[2] - q[0] * q[3]),
                1 - 2 * (q[1] ** 2 + q[3] ** 2),
                2 * (q[2] * q[3] + q[0] * q[1]),
            ],
            [
                2 * (q[1] * q[3] + q[0] * q[2]),
                2 * (q[2] * q[3] - q[0] * q[1]),
                1 - 2 * (q[1] ** 2 + q[2] ** 2),
            ],
        ]
    )
    # return rotated vector
    return R @ v


def axisquat(axis: str, theta: float) -> Quat:
    """
    Generate a quaternion that rotates by angle theta (radians)
    about axis 'x', 'y', or 'z'.
    Scalar-first format: [w, x, y, z]
    """
    h = theta * 0.5
    c = np.cos(h)
    s = np.sin(h)

    if axis == "x":
        return np.array([c, s, 0.0, 0.0], dtype=float)
    elif axis == "y":
        return np.array([c, 0.0, s, 0.0], dtype=float)
    elif axis == "z":
        return np.array([c, 0.0, 0.0, s], dtype=float)
    else:
        raise ValueError("axisquat error: axis must be 'x', 'y', or 'z'")
    
def cart2sph(v: Vec3):
    """
    Convert Cartesian coordinates to spherical coordinates.

    MATLAB-style definition:
      azimuth   = atan2(y, x)
      elevation = atan2(z, sqrt(x^2 + y^2))
      r         = sqrt(x^2 + y^2 + z^2)

    Args:
        v: 3D Cartesian vector [x, y, z]

    Returns:
        azimuth (float), elevation (float), radius (float)
    """
    x, y, z = v[0], v[1], v[2]
    r_xy = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)

    az = np.arctan2(y, x)
    el = np.arctan2(z, r_xy)

    return az, el, r

import numpy as np

def quat_from_to(v_from: Vec3, v_to: Vec3):
    """
    Shortest-arc quaternion that rotates v_from to v_to.
    Returns [w, x, y, z] (scalar first). Both vectors in same frame.
    """
    a = np.asarray(v_from, float)
    b = np.asarray(v_to,   float)

    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = np.dot(a, b)

    # If vectors are nearly opposite: pick a stable orthogonal axis
    if dot < -0.999999:
        # Choose the axis least aligned with 'a' to build an orthogonal
        ax = np.abs(a)
        if ax[0] <= ax[1] and ax[0] <= ax[2]:
            ortho = np.array([1.0, 0.0, 0.0])
        elif ax[1] <= ax[0] and ax[1] <= ax[2]:
            ortho = np.array([0.0, 1.0, 0.0])
        else:
            ortho = np.array([0.0, 0.0, 1.0])
        axis = np.cross(a, ortho)
        axis /= np.linalg.norm(axis)
        # 180 deg rotation about 'axis' -> w=0
        return np.array([0.0, axis[0], axis[1], axis[2]])

    # General case
    axis = np.cross(a, b)
    w = 1.0 + dot
    q = np.array([w, axis[0], axis[1], axis[2]])
    q /= np.linalg.norm(q)
    return q
