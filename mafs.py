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


def quat_from_to(v_from: Vec3, v_to: Vec3, preferred_axis: Vec3 | None = None) -> Quat:
    """
    Shortest-arc quaternion rotating v_from to v_to (both 3-vectors).
    Returns scalar-first quaternion [w, x, y, z].

    - Uses stable 180° handling with an optional preferred_axis to pick the flip axis.
    - Forces w >= 0 to keep quaternion sign consistent across frames.
    """
    a = np.asarray(v_from, dtype=float)
    b = np.asarray(v_to, dtype=float)

    # normalize inputs (guard tiny norms)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # no rotation if ill-defined
    a /= na
    b /= nb

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

    # Parallel → identity
    if dot > 1.0 - 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # Anti-parallel → 180° about a stable axis
    if dot < -1.0 + 1e-8:
        # Choose a stable axis not collinear with 'a'
        if preferred_axis is None:
            # pick a basis vector not aligned with 'a'
            pick_x = abs(a[0]) < 0.9
            ref = np.array([1.0, 0.0, 0.0]) if pick_x else np.array([0.0, 1.0, 0.0])
        else:
            ref = np.asarray(preferred_axis, dtype=float)
            if np.linalg.norm(ref) < 1e-12:
                ref = np.array([1.0, 0.0, 0.0])

        axis = np.cross(a, ref)
        if np.linalg.norm(axis) < 1e-12:
            # fallback if ref accidentally collinear: use Z
            axis = np.cross(a, np.array([0.0, 0.0, 1.0]))
        axis /= np.linalg.norm(axis)
        # 180° → w=0, xyz = axis
        q = np.array([0.0, axis[0], axis[1], axis[2]], dtype=float)
        # enforce hemisphere convention
        if q[0] < 0.0:
            q = -q
        return q

    # General case: closed form using half-angle
    axis = np.cross(a, b)
    s = np.sqrt((1.0 + dot) * 2.0)  # 2*cos(θ/2)
    invs = 1.0 / s
    q = np.array([0.5 * s, axis[0] * invs, axis[1] * invs, axis[2] * invs], dtype=float)

    # Keep quaternion sign consistent (avoid jumps q ↔ -q)
    if q[0] < 0.0:
        q = -q
    return q


# Helper functions to emulate MATLAB's internal utilities
def normalize_rows(A):
    """Normalize the rows of a 2D array A."""
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    # Avoid division by zero for zero vectors; they will remain zero.
    non_zero_mask = norms[:, 0] > 0
    A[non_zero_mask] = A[non_zero_mask] / norms[non_zero_mask]
    return A


def wrap_to_pi(angles):
    """Wraps angles (in radians) to the interval [-pi, pi]."""
    return np.arctan2(np.sin(angles), np.cos(angles))


def quat2axang(q: Quat) -> tuple[Vec3, float]:
    """
    Convert a single unit quaternion to axis-angle representation.

    Quaternion is scalar-first: [w, x, y, z].
    This is a streamlined version of the N-by-4 MATLAB equivalent.

    Args:
        q: A 4-element numpy array (Quat) of the quaternion.

    Returns:
        A tuple: (axis: Vec3, angle: float) 
        where angle is in radians and in the interval [0, pi] (or [-pi, pi]).
    """
    # 1. Normalize the input quaternion
    q_norm = np.asarray(q, dtype=float)
    norm_q = np.linalg.norm(q_norm)
    if norm_q < 1e-12:
        # If the quaternion is zero/degenerate, return no rotation
        return np.array([1.0, 0.0, 0.0]), 0.0
    q_norm /= norm_q
    
    w = q_norm[0]   # Scalar part
    v = q_norm[1:]  # Vector part [x y z]

    # 2. Angle Calculation: theta = 2 * acos(w)
    # Clip w to [-1, 1] to prevent floating point errors with acos
    w_clipped = np.clip(w, -1.0, 1.0)
    angle = 2.0 * np.arccos(w_clipped)

    # 3. Handle the zero/small rotation degenerate case
    s = np.sqrt(1.0 - w_clipped**2) # sin(angle/2)
    
    if s < 1e-8:
        # If angle is near 0 or 2*pi (s is near 0), axis is arbitrary.
        # Enforce angle = 0 and return an arbitrary axis (e.g., X-axis)
        axis = np.array([1.0, 0.0, 0.0])
        angle = 0.0
        return axis, angle
    
    # 4. Axis Calculation: axis = v / s
    axis = v / s
    
    # Note: The original MATLAB version used wrap_to_pi which converts the 
    # angle range from [0, 2*pi] to [-pi, pi]. Since acos(w) is only in [0, pi], 
    # 2*acos(w) is in [0, 2*pi]. Standard practice often leaves it in [0, 2*pi] 
    # or [0, pi] for axis-angle. We'll return the [0, 2*pi] angle, as is common 
    # when using the acos definition. If you specifically need [-pi, pi], 
    # you can uncomment the wrap line below.
    
    # angle = wrap_to_pi(angle) # Uncomment if [-pi, pi] range is required

    # 5. Return axis and angle
    return axis, angle

def dcm2quat(R: np.ndarray) -> Quat:
    """
    Convert a direction cosine matrix (DCM) to a quaternion [w, x, y, z].
    Assumes R maps from one frame to another using column direction vectors (3x3).

    Args:
        R : 3x3 rotation matrix (orthonormal)

    Returns:
        q : quaternion [w, x, y, z] (scalar-first)
    """
    R = np.asarray(R, dtype=float)
    assert R.shape == (3, 3), "dcm2quat: input matrix must be 3x3"

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        # find the largest diagonal
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([w, x, y, z], dtype=float)

    # Normalize to clean rounding noise
    q /= np.linalg.norm(q)
    return q
