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

        # calculate moment matrix (its expensive, so only do it once)
        num_rcs = len(self.rcs.r_b2rcs_b)
        moment_matrix = np.zeros((3, num_rcs))
        for i_rcs in range(num_rcs):
            moment_matrix[:, i_rcs] = np.linalg.cross(
                self.rcs.r_b2rcs_b[i_rcs], self.rcs.t_hat_b[i_rcs]
            )

        self.pinv_moment_matrix = np.linalg.pinv(moment_matrix)
        self.thruster_forces = np.zeros((num_rcs))
        self.throttle_pct = 0

    def add_thrusts(self, cmd_moment_pct_b: Vec3, throttle_pct: float):
        # convert percent commands to actual commands
        self.throttle_pct = throttle_pct
        cmd_moment_b = cmd_moment_pct_b * 2000
        throttle = throttle_pct * self.main_engine.t_max_n

        # add engine thrust
        self.fiz.force_b = self.fiz.force_b + throttle * self.main_engine.t_hat_b

        # RCS

        # get command scalars
        self.thruster_forces = np.clip(self.pinv_moment_matrix @ cmd_moment_b,0,self.rcs.t_max_n)

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

    def draw_thrusts(self,cam: "grafix_camera", screen: pygame.Surface):
        # main engine
        if(self.throttle_pct>0):
            draw_thruster(throttle_pct=self.throttle_pct,
                        t_hat_b=self.main_engine.t_hat_b,
                        pos_thruster_b=self.main_engine.r_b2e_b,
                        plume_length=10,
                        jitter_factor=1,
                        q_g2b=self.fiz.q_g2b,
                        r_g2b_g=self.fiz.r_g2p_g,
                        cam=cam,
                        screen=screen)
        
        # rcs
        for i in range(len(self.rcs.r_b2rcs_b)):
            rcs_throttle_pct = self.thruster_forces[i]/self.rcs.t_max_n
            if(rcs_throttle_pct > 0):
                draw_thruster(throttle_pct=rcs_throttle_pct,
                        t_hat_b=self.rcs.t_hat_b[i],
                        pos_thruster_b=self.rcs.r_b2rcs_b[i],
                        plume_length=1,
                        jitter_factor=1,
                        q_g2b=self.fiz.q_g2b,
                        r_g2b_g=self.fiz.r_g2p_g,
                        cam=cam,
                        screen=screen)

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
        planet_list.append(sn2_planet(fiz=fiz_obj, graf=graf_obj, name=cur_planet["name"]))

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

        stars.append(grafix_sphere(np.array([x, y, z]), 10000000, pygame.Color("white")))
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
    q_b2p = quat_from_to(np.array([0,0,1]),-t_hat_b)

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
        side=plume_length/10,
        height=plume_height,
        color=pygame.Color("orange"),
        q_g2b=quatmultiply(q_g2b, q_b2p),
    ).draw(cam, screen)

# orbit hud
def orbit_hud(draw_start_pos_pg: tuple[float, float], ship: sn2_spaceship, planet: sn2_planet, G: float, screen: pygame.Surface):
    # relative position and velocity
    v_rel = ship.fiz.v_g2p_g - planet.fiz.v_g2p_g
    r_rel = ship.fiz.r_g2p_g - planet.fiz.r_g2p_g

    v_mag = np.linalg.norm(v_rel)
    r_mag = np.linalg.norm(r_rel)

    # gravitational parameter μ = GM
    mu = G * planet.fiz.M

    c3 = v_mag**2 - 2 * mu / r_mag

    # hud box
    hud_w, hud_h = 250, 250
    hud_x, hud_y = int(draw_start_pos_pg[0]), int(draw_start_pos_pg[1])
    pygame.draw.rect(screen, (10, 10, 10), (hud_x, hud_y, hud_w, hud_h))
    pygame.draw.rect(screen, (255, 255, 255), (hud_x, hud_y, hud_w, hud_h), 2)

    # ----- focused planet label -----
    font = pygame.font.SysFont("consolas", 16)
    planet_text = font.render(f"Focus: {planet.name}", True, (180, 180, 180))
    screen.blit(planet_text, (hud_x + 10, hud_y + 10))

    # if c3 < 0 calculate orbital elements
    if c3 < 0:
        a = 1 / (2/r_mag - v_mag**2/mu)
        rDotV = np.dot(r_rel, v_rel)
        e_vec = (1/mu) * ((v_mag**2 - mu/r_mag) * r_rel - rDotV * v_rel)
        e = np.linalg.norm(e_vec)
        cos_TA = np.dot(e_vec, r_rel) / (e * r_mag)
        cos_TA = np.clip(cos_TA, -1, 1)  # avoid NaN from float roundoff
        TA = np.arccos(cos_TA)
        if rDotV < 0:
            TA = -TA

        # apoapsis and periapsis
        rp = a * (1 - e)
        ra = a * (1 + e)

        # inclination
        h_vec = np.cross(r_rel, v_rel)
        h_mag = np.linalg.norm(h_vec)
        inc = np.arccos(h_vec[2] / h_mag)

        # center of HUD frame
        cx = hud_x + hud_w // 2
        cy = hud_y + hud_h // 2

        # scale orbit to fit box
        max_orbit = max(ra, abs(rp))
        if max_orbit == 0:
            max_orbit = 1
        scale = (hud_w * 0.4) / max_orbit

        # planet drawn to scale with orbit
        planet_rad = planet.graf.radius * scale
        pygame.draw.circle(screen, planet.graf.color, (cx, cy), int(planet_rad))

        # calculate ellipse parameters (projected orbit)
        a_px = a * scale
        b_px = a * np.sqrt(1 - e**2) * scale  # semi-minor axis
        focus_shift = e * a_px  # shift from center

        # orbit ellipse
        orbit_rect = pygame.Rect(0, 0, 2*a_px, 2*b_px)
        orbit_rect.center = (cx - focus_shift, cy)
        pygame.draw.ellipse(screen, (255, 0, 0), orbit_rect, 2)

        # spacecraft position dot
        sc_x = ( a_px * np.cos(TA) - focus_shift ) + cx
        sc_y = ( b_px * np.sin(TA) ) + cy
        pygame.draw.circle(screen, ship.graf.color, (int(sc_x), int(sc_y)), 4)

    else:
        # text message
        escape_text = font.render("on an escape trajectory with %s", True, (255, 200, 200))
        screen.blit(escape_text, (hud_x + 10, hud_y + 40))

