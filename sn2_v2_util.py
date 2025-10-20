"""
This file contains class defs and helper functions for the sn2_v0.2 game.
"""

# import
from fisix import *
from grafix import *
from typing import List
import yaml

"""
Classes
"""


# planet
class sn2_planet:
    def __init__(self, fiz: "fizix_thing", graf: "grafix_sphere"):
        self.fiz = fiz
        self.graf = graf


# spaceship helper classes
class sn2_main_engine:
    def __init__(self, t_max_n: float, r_b2e_b: Vec3, t_hat_b: Vec3):
        self.t_max_n = t_max_n
        self.r_b2e_b = r_b2e_b
        self.t_hat_b = t_hat_b / np.linalg.norm(t_hat_b)


class rcs:
    def __init__(self, t_max_n: float, r_b2rcs_b: List[Vec3], t_hat_b: List[Vec3]):
        self.t_max_n = t_max_n
        self.r_b2rcs_b = r_b2rcs_b
        self.t_hat_b = t_hat_b / np.linalg.norm(t_hat_b)


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

        # calculate moment matrix (its expensive, so only do it once)
        num_rcs = len(self.rcs.r_b2rcs_b)
        moment_matrix = np.zeros((3, num_rcs))
        for i_rcs in range(num_rcs):
            moment_matrix[:, i_rcs] = np.linalg.cross(
                self.rcs.r_b2rcs_b[i_rcs], self.rcs.t_hat_b[i_rcs]
            )

        self.pinv_moment_matrix = np.linalg.pinv(moment_matrix)

    def add_thrusts(self, cmd_moment_pct_b: Vec3, throttle_pct: float):
        # convert percent commands to actual commands
        cmd_moment_b = cmd_moment_pct_b * self.rcs.t_max_n
        throttle = throttle_pct * self.main_engine.t_max_n

        # add engine thrust
        self.fiz.force_b = self.fiz.force_b + throttle * self.main_engine.t_hat_b

        # RCS

        # get command scalars
        thruster_commands = self.pinv_moment_matrix @ cmd_moment_b

        # get resultant force and moment of firing RCS thrusters
        f_rcs_b = np.zeros((3))
        m_rcs_b = np.zeros((3))
        for i_rcs in range(len(self.rcs.r_b2rcs_b)):
            f_rcs_b = f_rcs_b + self.rcs.t_hat_b[i_rcs] * thruster_commands[i_rcs]
            m_rcs_b = m_rcs_b + np.linalg.cross(self.rcs.r_b2rcs_b[i_rcs], f_rcs_b)
        self.fiz.momen_b = m_rcs_b
        self.fiz.force_b = self.fiz.force_b + f_rcs_b


# little userInput class
class sn2_userInput:
    def __init__(
        self, throttle_pct: float, cmd_moment_pct_b: Vec3, cam_az: float, cam_el: float
    ):
        self.throttle_pct = throttle_pct
        self.cmd_moment_pct_b = cmd_moment_pct_b
        self.cam_az = cam_az
        self.cam_el = cam_el


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
        planet_list.append(sn2_planet(fiz=fiz_obj, graf=graf_obj))

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

        stars.append(grafix_sphere(np.array([x, y, z]), 1000000, pygame.Color("white")))
    return stars


# get user input
def getInput(cam_az: float, cam_el: float, throttle_pct: float):
    # get state of keyb
    keys = pygame.key.get_pressed()

    # handle commanded moment
    cmd_moment_pct_b = np.array([0.0, 0.0, 0.0])
    if keys[pygame.K_w]:
        cmd_moment_pct_b[0] = 1
    if keys[pygame.K_s]:
        cmd_moment_pct_b[0] = -1
    if keys[pygame.K_a]:
        cmd_moment_pct_b[1] = -1
    if keys[pygame.K_d]:
        cmd_moment_pct_b[1] = 1
    if keys[pygame.K_q]:
        cmd_moment_pct_b[2] = 1
    if keys[pygame.K_e]:
        cmd_moment_pct_b[2] = -1

    # handle camera
    if keys[pygame.K_UP]:
        cam_el += np.deg2rad(1)
    if keys[pygame.K_DOWN]:
        cam_el -= np.deg2rad(1)
    if keys[pygame.K_LEFT]:
        cam_az += np.deg2rad(1)
    if keys[pygame.K_RIGHT]:
        cam_az -= np.deg2rad(1)

    # handle throttle
    if keys[pygame.K_LSHIFT]:
        throttle_pct += 1 / 100
    if keys[pygame.K_LCTRL]:
        throttle_pct -= 1 / 100
    throttle_pct = np.clip(throttle_pct, 0, 1)

    return sn2_userInput(
        throttle_pct=throttle_pct,
        cmd_moment_pct_b=cmd_moment_pct_b,
        cam_az=cam_az,
        cam_el=cam_el,
    )
