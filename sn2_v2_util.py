"""
This file contains class defs and helper functions for the sn2_v0.2 game.
"""

# import
from fisix import *
from grafix import *
from typing import List
import yaml
import numpy as np

"""
Classes
"""

# types
Vec2 = Annotated[NDArray[np.float64], (2,)]
Vec4 = Annotated[NDArray[np.float64], (4,)]


# attitude helpers
def pointing_quaternion_from_z_axis(z_axis: Vec3, up_reference: Vec3) -> Quat:
    """Return a quaternion whose body +Z aligns with ``z_axis``."""

    z_axis = np.asarray(z_axis, dtype=float)
    up_reference = np.asarray(up_reference, dtype=float)

    if np.linalg.norm(z_axis) < 1e-9:
        raise ValueError("z_axis must be non-zero")

    z_axis = z_axis / np.linalg.norm(z_axis)

    ref = up_reference
    if np.linalg.norm(ref) < 1e-9:
        ref = np.array([0.0, 1.0, 0.0])
    ref = ref / np.linalg.norm(ref)

    x_axis = np.cross(ref, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        fallback = np.array([1.0, 0.0, 0.0])
        if np.abs(z_axis[0]) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(fallback, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)

    dcm = np.column_stack((x_axis, y_axis, z_axis))
    return dcm2quat(dcm)


# planet
class sn2_planet:
    def __init__(self, fiz: "fizix_thing", graf: "grafix_sphere", name: str):
        self.fiz = fiz
        self.graf = graf
        self.name = name


# spaceship helper classes
class sn2_main_engine:
    def __init__(self, t_max_n: float, r_b2e_b: Vec3, t_hat_b: Vec3):
        self.t_max_n = t_max_n
        self.r_b2e_b = r_b2e_b
        self.t_hat_b = t_hat_b


class rcs:
    def __init__(self, t_max_n: float, r_b2rcs_b: List[Vec3], t_hat_b: List[Vec3]):
        self.t_max_n = t_max_n
        self.r_b2rcs_b = r_b2rcs_b
        self.t_hat_b = t_hat_b


# spaceship
class sn2_spaceship:
    def __init__(
        self,
        fiz: "fizix_thing",
        graf: "grafix_tri_prism",
        main_engine: sn2_main_engine,
        rcs: rcs,
    ):
        self.fiz = fiz
        self.graf = graf
        self.main_engine = main_engine
        self.rcs = rcs

        # additional RCS stuff
        # calculate moment matrix (its expensive, so only do it once)
        num_rcs = len(self.rcs.r_b2rcs_b)
        moment_matrix = np.zeros((3, num_rcs))
        for i_rcs in range(num_rcs):
            moment_matrix[:, i_rcs] = np.linalg.cross(
                self.rcs.r_b2rcs_b[i_rcs], self.rcs.t_hat_b[i_rcs]
            )
        self.pinv_moment_matrix = np.linalg.pinv(moment_matrix)
        self.thruster_forces = np.zeros((num_rcs))

        # commanded state
        self.cmd_throttle_pct_b = 0
        self.cmd_moment_pct_b = np.array([0, 0, 0])

        # controls stuff
        self.pointing_error_integral = np.array([0,0,0])

    def add_thrusts(self):
        # ensure throttle is clipped
        self.cmd_throttle_pct_b = np.clip(self.cmd_throttle_pct_b, 0, 1)

        # convert percent commands to actual commands
        cmd_throttle_b = self.cmd_throttle_pct_b * self.main_engine.t_max_n
        cmd_moment_b = self.cmd_moment_pct_b * 25000

        # add engine thrust
        self.fiz.force_b = self.fiz.force_b + cmd_throttle_b * self.main_engine.t_hat_b

        # RCS

        # get command scalars
        self.thruster_forces = np.clip(
            self.pinv_moment_matrix @ cmd_moment_b, 0, self.rcs.t_max_n
        )

        # get resultant force and moment of firing RCS thrusters
        f_rcs_b = np.zeros((3))
        m_rcs_b = np.zeros((3))
        for i_rcs in range(len(self.rcs.r_b2rcs_b)):
            # get force of single thruster t_hat * thrust
            cur_f_b = self.rcs.t_hat_b[i_rcs] * self.thruster_forces[i_rcs]
            # add to net force
            f_rcs_b = f_rcs_b + cur_f_b
            # get current moment (r x f) and add to net moment
            m_rcs_b = m_rcs_b + np.linalg.cross(self.rcs.r_b2rcs_b[i_rcs], cur_f_b)
        self.fiz.momen_b = m_rcs_b
        self.fiz.force_b = self.fiz.force_b + f_rcs_b

    def draw_thrusts(self, cam: "grafix_camera", screen: pygame.Surface):
        # main engine
        if self.cmd_throttle_pct_b > 0:
            draw_thruster(
                throttle_pct=self.cmd_throttle_pct_b,
                t_hat_b=self.main_engine.t_hat_b,
                pos_thruster_b=self.main_engine.r_b2e_b,
                plume_length=10,
                jitter_factor=1,
                q_g2b=self.fiz.q_g2b,
                r_g2b_g=self.fiz.r_g2p_g,
                cam=cam,
                screen=screen,
            )

        # rcs
        for i in range(len(self.rcs.r_b2rcs_b)):
            rcs_throttle_pct = self.thruster_forces[i] / self.rcs.t_max_n
            if rcs_throttle_pct > 0:
                draw_thruster(
                    throttle_pct=rcs_throttle_pct,
                    t_hat_b=self.rcs.t_hat_b[i],
                    pos_thruster_b=self.rcs.r_b2rcs_b[i],
                    plume_length=3,
                    jitter_factor=1,
                    q_g2b=self.fiz.q_g2b,
                    r_g2b_g=self.fiz.r_g2p_g,
                    cam=cam,
                    screen=screen,
                )

    def rate_control(self, omega_cmd: Vec3):
        # generate a moment command, and assign it to the ship's moment.

        # get error
        omega_err = omega_cmd - self.fiz.ome_g2b

        # TODO: replace this goober logic
        principal_inertias = np.array(
            [self.fiz.I[0, 0], self.fiz.I[1, 1], self.fiz.I[2, 2]]
        )
        kp = principal_inertias * 5 / 10000

        # construct command
        cmd = np.clip(omega_err * kp, -1, 1)

        # apply deadband
        for i in range(len(cmd)):
            if np.abs(cmd[i]) < 1e-3:
                cmd[i] = 0

        self.cmd_moment_pct_b = cmd

    def angle_control(self, q_g2b_cmd: Quat):
        # obtain error quaternion
        # Ensure quaternion sign consistency to avoid 360° error states.
        # The controller expects the smallest-angle difference between the
        # commanded and current attitude. Because quaternions q and -q encode
        # the same physical rotation, a raw command that ends up on the
        # opposite hemisphere of the current attitude would produce an error
        # quaternion close to [-1, 0, 0, 0] (i.e. 360°). This collapses the
        # axis-angle error to zero, preventing the controller from issuing the
        # needed 180° rotation for retrograde pointing. Align the command
        # quaternion with the current attitude before computing the error.
        if np.dot(q_g2b_cmd, self.fiz.q_g2b) < 0:
            q_g2b_cmd = -q_g2b_cmd

        q_err = quatmultiply(q_g2b_cmd, quatinv(self.fiz.q_g2b))
        # obtain axis-angle from error
        axis,angle = quat2axang(q_err)
        
        # get error, and iterate on error integral
        angle_error = axis * angle
        self.pointing_error_integral = self.pointing_error_integral + angle_error * self.fiz.dt

        # enforce saturation (anti wind-up)
        self.pointing_error_integral = np.clip(self.pointing_error_integral,-np.pi,np.pi)
        
        kp = 5
        ki = 0
        kd = 10

        omega_cmd = kp * angle_error + ki * self.pointing_error_integral - kd * self.fiz.ome_g2b

        self.rate_control(omega_cmd)

    def att_ctrl(
        self,
        user_cmd_moment_pct_b: Vec3,
        pointing_setting: int,
        planet: sn2_planet,
        cam: "grafix_camera",
        screen: pygame.Surface,
    ):
        # if pointing setting is 0, assign cmd_moment directly
        if pointing_setting == 0:
            self.cmd_moment_pct_b = user_cmd_moment_pct_b
        # if pointing setting is 1...
        elif pointing_setting == 1:
            # try to null rate if no moment is commanded.
            if np.linalg.norm(user_cmd_moment_pct_b) == 0:
                self.rate_control(np.array([0, 0, 0]))
            # pass through commmand if moment is provided.
            else:
                self.cmd_moment_pct_b = user_cmd_moment_pct_b
        else:
            # relative velocity to planet
            v = self.fiz.v_g2p_g - planet.fiz.v_g2p_g
            v = v / np.linalg.norm(v)

            # relative position to planet
            r = self.fiz.r_g2p_g - planet.fiz.r_g2p_g
            r = r / np.linalg.norm(r)

            # orbit angular momentum vector
            h = np.cross(r, v)
            h = h / np.linalg.norm(h)

            q_g2b_cmd = None
            if pointing_setting == 2:
                q_g2b_cmd = pointing_quaternion_from_z_axis(v, h)
            elif pointing_setting == 3:
                q_g2b_cmd = pointing_quaternion_from_z_axis(-v, h)
            else:
                return

            triad(cam,screen,q_g2b_cmd,self.fiz.r_g2p_g)
            self.angle_control(q_g2b_cmd)


# little userInput class
class sn2_userInput:
    def __init__(self, throttle_increment: float, cmd_moment_pct_b: Vec3):
        self.throttle_increment = throttle_increment
        self.cmd_moment_pct_b = cmd_moment_pct_b


"""
YAML
"""


# import planets
def import_planets(filepath: str, dt: float):
    with open(filepath, "r") as f:
        planet_data = yaml.safe_load(f)

    # pre-allocate list
    planet_list = []

    # import planets
    for planet_name in planet_data:
        # extract data from current planet
        cur_planet = planet_data[planet_name]

        # build fisix object
        cur_fiz = cur_planet["fiz"]
        fiz_obj = fizix_thing(
            r_g2p_g=np.array(cur_fiz["r_g2p_g"]),
            v_g2p_g=np.array(cur_fiz["v_g2p_g"]),
            a_g2p_g=np.array([0, 0, 0]),
            q_g2b=np.array(cur_fiz["q_g2b"]),
            ome_g2b=np.array(cur_fiz["ome_g2b"]),
            alp_g2b=np.array([0, 0, 0]),
            M=float(cur_fiz["M"]),
            I=np.array(cur_fiz["I"]),
            force_b=np.array([0, 0, 0]),
            momen_b=np.array([0, 0, 0]),
            dt=dt,
        )

        # build grafix object
        cur_graf = cur_planet["graf"]
        graf_obj = grafix_sphere(
            r_g2p_g=np.array(cur_fiz["r_g2p_g"]),
            radius=float(cur_graf["radius"]),
            color=tuple(cur_graf["color"]),
        )

        # build planet object
        planet_list.append(
            sn2_planet(fiz=fiz_obj, graf=graf_obj, name=cur_planet["name"])
        )

    return planet_list


# import ship
def import_ship(filepath: str, dt: float):
    with open(filepath, "r") as f:
        ship_data = yaml.safe_load(f)

    # build fisix object
    cur_fiz = ship_data["fiz"]
    fiz_obj = fizix_thing(
        r_g2p_g=np.array(cur_fiz["r_g2p_g"]),
        v_g2p_g=np.array(cur_fiz["v_g2p_g"]),
        a_g2p_g=np.array([0, 0, 0]),
        q_g2b=np.array(cur_fiz["q_g2b"]),
        ome_g2b=np.array(cur_fiz["ome_g2b"]),
        alp_g2b=np.array([0, 0, 0]),
        M=float(cur_fiz["M"]),
        I=np.array(cur_fiz["I"]),
        force_b=np.array([0, 0, 0]),
        momen_b=np.array([0, 0, 0]),
        dt=dt,
    )
    # build grafix object
    cur_graf = ship_data["graf"]
    graf_obj = grafix_tri_prism(
        r_g2p_g=np.array(cur_fiz["r_g2p_g"]),
        q_g2b=np.array(cur_fiz["q_g2b"]),
        side=float(cur_graf["side"]),
        height=float(cur_graf["height"]),
        color=tuple(cur_graf["color"]),
    )
    # build engine object
    cur_eng = ship_data["main_engine"]
    eng_obj = sn2_main_engine(
        t_hat_b=np.array(cur_eng["t_hat_b"]),
        r_b2e_b=np.array(cur_eng["r_b2e_b"]),
        t_max_n=float(cur_eng["t_max_n"]),
    )
    # build rcs object
    cur_rcs = ship_data["rcs"]
    r_b2rcs_b_LIST = []
    t_hat_b_LIST = []
    for i in range(len(cur_rcs["r_b2rcs_b"])):
        r_b2rcs_b_LIST.append(np.array(cur_rcs["r_b2rcs_b"][i]))
        t_hat_b_LIST.append(np.array(cur_rcs["t_hat_b"][i]))
    rcs_obj = rcs(
        t_hat_b=t_hat_b_LIST,
        r_b2rcs_b=r_b2rcs_b_LIST,
        t_max_n=float(cur_rcs["t_max_n"]),
    )
    # build ship
    return sn2_spaceship(
        fiz=fiz_obj,
        graf=graf_obj,
        main_engine=eng_obj,
        rcs=rcs_obj,
    )


"""
Misc helpers
"""


# create star graphics objects
def make_stars(NUM_STARS: float, STAR_DISTANCE: float):
    stars = []
    for _ in range(NUM_STARS):
        # random azimuth + elevation
        az = np.random.uniform(0, 2 * np.pi)
        el = np.random.uniform(-np.pi / 2, np.pi / 2)

        # convert to cartesian at fixed radius
        x = STAR_DISTANCE * np.cos(el) * np.cos(az)
        y = STAR_DISTANCE * np.cos(el) * np.sin(az)
        z = STAR_DISTANCE * np.sin(el)

        stars.append(
            grafix_sphere(np.array([x, y, z]), 10000000, pygame.Color("white"))
        )
    return stars


# get user input
def getKeyInput():
    # get state of keyb
    keys = pygame.key.get_pressed()

    # handle commanded moment
    cmd_moment_pct_b = np.array([0.0, 0.0, 0.0])
    if keys[pygame.K_w]:
        cmd_moment_pct_b[0] = 1
    if keys[pygame.K_s]:
        cmd_moment_pct_b[0] = -1
    if keys[pygame.K_a]:
        cmd_moment_pct_b[1] = 1
    if keys[pygame.K_d]:
        cmd_moment_pct_b[1] = -1
    if keys[pygame.K_q]:
        cmd_moment_pct_b[2] = -1
    if keys[pygame.K_e]:
        cmd_moment_pct_b[2] = 1

    # handle throttle
    throttle_increment = 0
    if keys[pygame.K_LSHIFT]:
        throttle_increment = 1 / 100
    if keys[pygame.K_LCTRL]:
        throttle_increment = -1 / 100

    return sn2_userInput(
        throttle_increment=throttle_increment,
        cmd_moment_pct_b=cmd_moment_pct_b,
    )


# draw_thruster
def draw_thruster(
    throttle_pct: float,
    t_hat_b: Vec3,
    pos_thruster_b: Vec3,  # vector from body origin to plume rotation point expressed in body coords.
    plume_length: float,
    jitter_factor: float,
    q_g2b: Quat,
    r_g2b_g: Vec3,  # vector from global origin to body origin expressed in global coords.
    cam: "grafix_camera",
    screen: pygame.Surface,
):
    # obtain q_b2p
    q_b2p = quat_from_to(np.array([0, 0, 1]), -t_hat_b)

    # calculate jitter modifier
    flame_jitter = (np.random.rand() - 0.5) * throttle_pct * jitter_factor
    plume_height = throttle_pct * plume_length + flame_jitter

    # transform rotation point 2 plume origin into global frame.
    r_r2p_p = np.array([0, 0, plume_height / 2])
    r_r2p_g = quatrotate(quatinv(quatmultiply(q_g2b, q_b2p)), r_r2p_p)

    # get plume rotation point position in global frame
    r_b2r_g = quatrotate(quatinv(q_g2b), pos_thruster_b)

    # add them all up
    r_g2p_g = r_g2b_g + r_b2r_g + r_r2p_g

    # create plume grafix object
    grafix_tri_prism(
        r_g2p_g=r_g2p_g,
        side=plume_length / 10,
        height=plume_height,
        color=pygame.Color("orange"),
        q_g2b=quatmultiply(q_g2b, q_b2p),
    ).draw(cam, screen)


# keplerian function
def keplerian(ship: sn2_spaceship, planet: sn2_planet, G: float):
    # obtain relative position and velocity
    R = ship.fiz.r_g2p_g - planet.fiz.r_g2p_g
    V = ship.fiz.v_g2p_g - planet.fiz.v_g2p_g

    r = np.linalg.norm(R)
    v = np.linalg.norm(V)

    # calculate intermediate variables
    mu = planet.fiz.M * G
    H = np.cross(R, V)
    h = np.linalg.norm(H)
    r_hat = R / r
    E = np.cross(V, H) / mu - r_hat  # eccentricity vector
    epsilon = v**2 / 2 - mu / r  # specific orbital energy

    # orbital elements
    e = np.linalg.norm(E)  # eccentricity
    i_rad = np.arccos(H[2] / h)  # inclination
    a = -mu / (2 * epsilon) if epsilon != 0 else np.inf  # semi-major axis
    b = a * np.sqrt(1 - e**2) if e < 1 else np.nan  # semiminor axis

    # periapsis & apoapsis
    if e < 1:
        rp = a * (1 - e)
        ra = a * (1 + e)
    else:
        rp = a * (1 - e)
        ra = np.inf

    # true anomaly (ν)
    # dot(E,R) = e*r*cos(nu)
    cos_nu = np.dot(E, R) / (e * r)
    sin_nu = np.dot(R, V) / (e * h)  # <-- FIX
    nu = np.arctan2(sin_nu, cos_nu)

    c3 = 2 * epsilon  # characteristic energy
    alt = r - planet.graf.radius  # altitude

    return [e, i_rad, rp, ra, c3, v, alt, a, b, nu]


def hud(
    hud_pos: Vec2,
    ship: sn2_spaceship,
    nearest_planet: sn2_planet,
    pointing_setting: int,
    screen: pygame.Surface,
    click_array: Vec3,
    font: pygame.font.Font,
    font_small: pygame.font.Font,
    G: float,
):
    red = np.array([255, 0, 89])
    green = np.array([0, 255, 0])
    blue = np.array([0, 157, 255])

    """pointer box definition and click checking"""
    # define names of settings
    box_texts = [
        "OFF",  # 0
        "SAS",  # 1
        "PRO",  # 2
        "RET",  # 3
        "R-IN",  # 4
        "R-OUT",  # 5
        "+H",  # 6
        "-H",  # 7
    ]

    # define setting colors
    box_text_colors = [
        (255, 255, 255),
        (255, 255, 255),
        (0, 255, 0),
        (0, 255, 0),
        (52, 171, 235),
        (52, 171, 235),
        (177, 52, 235),
        (177, 52, 235),
    ]

    # pointing box sizes
    box_positions = [
        np.array([15, 15]),
        np.array([15, 48]),
        np.array([15, 81]),
        np.array([15, 114]),
        np.array([63, 15]),
        np.array([63, 48]),
        np.array([63, 81]),
        np.array([63, 114]),
    ]
    box_size = np.array([40, 24])

    # if you're clicking, see if you're in a box. Else, don't change the pointing_setting
    mouse_x = click_array[1]
    mouse_y = click_array[2]
    if click_array[0]:
        for i in range(len(box_positions)):
            box_min_xy = hud_pos + box_positions[i]
            box_max_xy = box_min_xy + box_size
            in_x = mouse_x > box_min_xy[0] and mouse_x < box_max_xy[0]
            in_y = mouse_y > box_min_xy[1] and mouse_y < box_max_xy[1]
            if in_x and in_y:
                pointing_setting = i

    """draw actual hud boxes"""

    def box(
        hud_pos: Vec2,
        box_pos: Vec2,
        size: Vec2,
        screen: pygame.Surface,
        border_color: Vec3 = np.array([255, 255, 255]),
        inner_color: Vec3 = np.array([0, 0, 0]),
    ):
        box_coords = np.concatenate((np.array([0, 0]), size))
        global_pos = hud_pos + box_pos
        box_coords[0:2] = box_coords[0:2] + global_pos
        inner_box_coords = box_coords + np.array([1.5, 1.5, -3, -3])
        pygame.draw.rect(screen, tuple(border_color), tuple(box_coords))
        pygame.draw.rect(screen, tuple(inner_color), tuple(inner_box_coords))

    # main rectangle
    box(hud_pos, np.array([0, 0]), np.array([450, 150]), screen)

    # draw the pointing boxes
    box(
        hud_pos,
        np.array([8, 8]),
        np.array([105, 135]),
        screen,
        border_color=red,
    )
    for i in range(len(box_positions)):
        if i == pointing_setting:
            box(
                hud_pos,
                box_positions[i],
                box_size,
                screen,
                inner_color=np.array([100, 100, 100]),
            )
        else:
            box(hud_pos, box_positions[i], box_size, screen)
        screen.blit(
            font.render(box_texts[i], False, box_text_colors[i]),
            tuple(hud_pos + box_positions[i] + np.array([3, 3])),
        )

    # draw the orbit vis box
    box(
        hud_pos,
        np.array([120, 8]),
        np.array([210, 135]),
        screen,
        border_color=green,
    )
    box(hud_pos, np.array([128, 15]), np.array([195, 27]), screen)
    # add nearest planet name
    screen.blit(
        font.render(f"Focus: {nearest_planet.name}", False, green),
        tuple(hud_pos + np.array([128, 15]) + np.array([5, 3])),
    )

    ## draw the mini orbit with you on it
    #  0    1     2   3   4  5   6   7   8    9
    # [e, i_rad, rp, ra, c3, v, alt, a   b   nu]
    box_pos = np.array([128, 49])
    box_size = np.array([195, 85])
    box(hud_pos, box_pos, box_size, screen)
    kep_params = keplerian(ship, nearest_planet, G)

    if kep_params[4] < 0:  # bound orbit only
        # box geometry
        top_left = hud_pos + box_pos
        center_pos = top_left + box_size / 2.0

        # semimajor/minor axes
        a = float(kep_params[7])
        b = float(kep_params[8])
        e = float(kep_params[0])
        nu = float(kep_params[9])

        # scale to fit orbit inside box
        pad = np.array([8.0, 8.0])
        inner_size = box_size - 2 * pad
        s_orbit = min(inner_size[0] / (2.0 * a), inner_size[1] / (2.0 * b))
        a_scaled = a * s_orbit
        b_scaled = b * s_orbit

        # focus offset
        c = e * a
        c_scaled = c * s_orbit

        # ellipse top-left for pygame
        ellipse_rect = (
            int(center_pos[0] - a_scaled),
            int(center_pos[1] - b_scaled),
            int(2.0 * a_scaled),
            int(2.0 * b_scaled),
        )
        pygame.draw.ellipse(screen, red, ellipse_rect, width=1)

        # draw planet at right focus
        planet_x = center_pos[0] + c_scaled
        planet_y = center_pos[1]
        pygame.draw.circle(
            screen,
            nearest_planet.graf.color,
            (int(planet_x), int(planet_y)),
            nearest_planet.graf.radius * s_orbit,
        )

        # draw periapsis (left side)
        pe_x = center_pos[0] + a_scaled
        pe_y = center_pos[1]
        pygame.draw.circle(screen, (255, 204, 0), (int(pe_x), int(pe_y)), 3, width=1)

        # draw apoapsis (right side)
        ap_x = center_pos[0] - a_scaled
        ap_y = center_pos[1]
        pygame.draw.circle(screen, blue, (int(ap_x), int(ap_y)), 3, width=1)

        # draw current spacecraft position (convert ν -> E for center-param ellipse)
        E = 2.0 * np.arctan2(np.tan(nu / 2.0), np.sqrt((1.0 + e) / (1.0 - e)))
        x_orbit = a_scaled * np.cos(E)
        y_orbit = b_scaled * np.sin(E)
        ship_dot_x = center_pos[0] + x_orbit
        ship_dot_y = center_pos[1] - y_orbit  # pygame Y is down!
        pygame.draw.circle(screen, green, (int(ship_dot_x), int(ship_dot_y)), 3)

    else:
        screen.blit(
            font.render(f"Escape Trajectory!", False, red),
            tuple(hud_pos + np.array([128, 50]) + np.array([5, 3])),
        )

    # draw the orbit stats box
    box(
        hud_pos,
        np.array([338, 8]),
        np.array([105, 135]),
        screen,
        border_color=blue,
    )

    # fill in stats box
    # build the list of text
    stat_texts = [
        f"v: {kep_params[5]:+.4e}",
        f"alt: {kep_params[6]:+.4e}",
        f"ap: {kep_params[3]:+.4e}",
        f"pe: {kep_params[2]:+.4e}",
        f"i: {np.rad2deg(kep_params[1]):+.4}",
        f"e: {kep_params[0]:+.4}",
        f"c3: {kep_params[4]:+.4}",
    ]
    # blit it
    x_orbit_text = hud_pos[0] + 345
    y_orbit_text = hud_pos[1] + 12
    for i in range(len(stat_texts)):
        screen.blit(
            font_small.render(stat_texts[i], False, blue), (x_orbit_text, y_orbit_text)
        )
        y_orbit_text += 18

    return pointing_setting
