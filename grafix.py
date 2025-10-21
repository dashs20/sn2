import pygame
from typing import Annotated
from numpy.typing import NDArray
import numpy as np
from mafs import *

ColorType = pygame.Color

"""define objects"""


# camera
class grafix_camera:
    def __init__(self, r_g2c_g: Vec3, q_g2c: Quat, fovDeg: float):
        self.r_g2c_g = r_g2c_g
        self.q_g2c = q_g2c
        self.fovDeg = fovDeg

    def update_state(self, r_g2c_g: Vec3, q_g2c: Quat):
        self.r_g2c_g = r_g2c_g
        self.q_g2c = q_g2c


# line
class grafix_line:
    def __init__(self, r1_g2p_g: Vec3, r2_g2p_g: Vec3, color: ColorType):
        self.r1_g2p_g = r1_g2p_g
        self.r2_g2p_g = r2_g2p_g
        self.color = color

    def draw(self, camera: "grafix_camera", screen: pygame.Surface):
        # 1) get the pygame coordinates for both points
        tmp1 = point_g2pg(camera, screen, self.r1_g2p_g)
        r1_pg = tmp1[0]
        tmp2 = point_g2pg(camera, screen, self.r2_g2p_g)
        r2_pg = tmp2[0]
        # 2) draw the pygame line
        if tmp1[2] and tmp2[2]:
            pygame.draw.line(screen, self.color, r1_pg, r2_pg)


# sphere
class grafix_sphere:
    def __init__(self, r_g2p_g: Vec3, radius: float, color: ColorType):
        self.r_g2p_g = r_g2p_g
        self.radius = radius
        self.color = color

    def update_state(self, r_g2p_g: Vec3, q_g2b: Quat):
        self.r_g2p_g = r_g2p_g

    def draw(self, camera: "grafix_camera", screen: pygame.Surface):
        # 1) get position & scale factor in pygame coordinates
        tmp_list = point_g2pg(camera, screen, self.r_g2p_g)
        r_pg = tmp_list[0]
        sf = tmp_list[1]
        # 2) draw the pygame sphere (circle)
        if tmp_list[2]:
            pygame.draw.circle(screen, self.color, r_pg, self.radius * sf)


# rectangular prism
class grafix_rec_prism:
    def __init__(self, r_g2p_g: Vec3, size: Vec3, q_g2b: Quat, color: ColorType):
        self.r_g2p_g = r_g2p_g
        self.size = size
        self.q_g2b = q_g2b
        self.color = color

    def update_state(self, r_g2p_g: Vec3, q_g2b: Quat):
        self.r_g2p_g = r_g2p_g
        self.q_g2b = q_g2b

    def draw(self, camera: "grafix_camera", screen: pygame.Surface):
        # 1) define 8 body-frame vertices
        sx, sy, sz = self.size
        verts_b = np.array(
            [
                [+sx, -sy, -sz],
                [+sx, +sy, -sz],
                [-sx, +sy, -sz],
                [-sx, -sy, -sz],
                [+sx, -sy, +sz],
                [+sx, +sy, +sz],
                [-sx, +sy, +sz],
                [-sx, -sy, +sz],
            ]
        )

        # 2) rotate to global & translate position
        q_b2g = quatinv(self.q_g2b)
        verts_g = np.array([quatrotate(q_b2g, v) + self.r_g2p_g for v in verts_b])

        # 3) edges by index pairs
        edges = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # bottom square
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # top square
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # vertical edges
        ]

        # 4) draw
        for i, j in edges:
            grafix_line(verts_g[i], verts_g[j], self.color).draw(camera, screen)


class grafix_tri_prism:
    def __init__(
        self, r_g2p_g: Vec3, side: float, height: float, q_g2b: Quat, color: ColorType
    ):
        self.r_g2p_g = r_g2p_g  # center of rotation in global frame
        self.side = side  # equilateral base side length
        self.height = height  # tip-to-base distance
        self.q_g2b = q_g2b
        self.color = color

    def update_state(self, r_g2p_g: Vec3, q_g2b: Quat):
        self.r_g2p_g = r_g2p_g
        self.q_g2b = q_g2b

    def draw(self, camera: "grafix_camera", screen: pygame.Surface):
        # base triangle math
        # Clean symmetric prism geometry (local body frame)
        # Centered at origin, tip along +Z
        a = self.side
        h_base = np.sqrt(3) / 2 * a  # equilateral triangle height
        verts_b = np.array([
            [-a/2, -h_base/3, -self.height/2],  # base left
            [ a/2, -h_base/3, -self.height/2],  # base right
            [ 0.0,  2*h_base/3, -self.height/2], # base top
            [ 0.0,  0.0,         self.height/2]  # tip (centered)
        ])

        # rotate and translate
        q_b2g = quatinv(self.q_g2b)
        verts_g = np.array([quatrotate(q_b2g, v) + self.r_g2p_g for v in verts_b])

        # edges of triangular prism
        edges = [
            (0, 1),
            (1, 2),
            (2, 0),  # base triangle
            (0, 3),
            (1, 3),
            (2, 3),  # sides to tip
        ]

        for i, j in edges:
            grafix_line(verts_g[i], verts_g[j], self.color).draw(camera, screen)


"""misc helpers"""


# point in global -> point in pygame
# returns list, 1st ind is pygame position, 2nd ind is scale factor
def point_g2pg(camera: "grafix_camera", screen: pygame.Surface, r_g2p_g: Vec3):
    # 1) vector from camera to point, in global frame
    r_c2p_g = r_g2p_g - camera.r_g2c_g
    # 2) rotate into camera frame
    r_c2p_c = quatrotate(camera.q_g2c, r_c2p_g)

    # ---- SAFETY: camera looks along -Z; points in front have z < 0 ----
    if r_c2p_c[2] >= 0:     # <- flipped sign
        return [(0, 0), 0, 0]

    # 3) get screen info
    screen_x, screen_y = screen.get_size()

    # 4) perspective divide (use positive depth = -z)
    depth = -r_c2p_c[2]     # <- new
    fov_w = depth * 2 * np.tan(np.deg2rad(camera.fovDeg) / 2)
    if depth <= 0 or fov_w < 1e-9:
        return [(0, 0), 0, 0]

    sf = screen_x / fov_w
    if not np.isfinite(sf):
        return [(0, 0), 0, 0]

    # 5) scale projected coords
    r_scaled = sf * r_c2p_c

    # 6) to pygame coords (x right, y down)
    x_pg = r_scaled[0] + screen_x / 2
    y_pg = -r_scaled[1] + screen_y / 2
    if np.abs(x_pg) > screen_x * 100 or np.abs(y_pg) > screen_y * 100:
        return [(0, 0), 0, 0]

    return [(int(x_pg), int(y_pg)), sf, 1]
