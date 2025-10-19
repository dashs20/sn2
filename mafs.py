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

