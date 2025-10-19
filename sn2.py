from fisix import *
from grafix import *
from typing import List

""" ~~~ GLOBAL CONSTANTS ~~~ """
dt_g = 1 / 120
G = 6.67e-11

""" ~~~ INIT PYGAME ~~~ """
pygame.init()
screen = pygame.display.set_mode((1200, 800))
pygame.display.set_caption("sn2")
font = pygame.font.SysFont("consolas", 18)

""" ~~~ SETUP VARIOUS GAME OBJECTS ~~~ """
"""stars"""
NUM_STARS = 1000
stars = []
np.random.seed(1)
STAR_DISTANCE = 384472282
for _ in range(NUM_STARS):
    # random azimuth + elevation
    az = np.random.uniform(0, 2 * np.pi)
    el = np.random.uniform(-np.pi / 2, np.pi / 2)

    # convert to cartesian at fixed radius
    x = STAR_DISTANCE * np.cos(el) * np.cos(az)
    y = STAR_DISTANCE * np.cos(el) * np.sin(az)
    z = STAR_DISTANCE * np.sin(el)

    stars.append(grafix_sphere(np.array([x, y, z]), 1000000, pygame.Color("white")))

"""earth"""
# physics object
earth_fiz = fizix_thing(
    r_g2p_g=np.array([0, 0, 0]),
    v_g2p_g=np.array([0, 0, 0]),
    a_g2p_g=np.array([0, 0, 0]),
    q_g2b=np.array([1, 0, 0, 0]),
    ome_g2b=np.array([0, 0, 0]),
    alp_g2b=np.array([0, 0, 0]),
    M=5.97219 * 10**24,
    I=np.eye(3),
    force_b=np.array([0, 0, 0]),
    momen_b=np.array([0, 0, 0]),
    dt=dt_g,
)
# graphics object
earth_graf = grafix_sphere(
    r_g2p_g=earth_fiz.r_g2p_g, radius=6371000, color=pygame.Color("blue")
)

"""moon"""
# physics object
moon_fiz = fizix_thing(
    r_g2p_g=np.array([384472282, 0, 0]),
    v_g2p_g=np.array([0, 1022.828, 0]),
    a_g2p_g=np.array([0, 0, 0]),
    q_g2b=np.array([1, 0, 0, 0]),
    ome_g2b=np.array([0, 0, 0]),
    alp_g2b=np.array([0, 0, 0]),
    M=7.34767309 * 10**22,
    I=np.eye(3),
    force_b=np.array([0, 0, 0]),
    momen_b=np.array([0, 0, 0]),
    dt=dt_g,
)
# graphics object
moon_graf = grafix_sphere(
    r_g2p_g=moon_fiz.r_g2p_g, radius=1737447.78, color=pygame.Color("gray")
)

"""spaceship"""
# physics object
spaceship_fiz = fizix_thing(
    r_g2p_g=np.array([6773336, 0, 0]),
    v_g2p_g=np.array([0, 7778.496, 0]),
    a_g2p_g=np.array([0, 0, 0]),
    q_g2b=np.array([1, 0, 0, 0]),
    ome_g2b=np.array([0, 0, 0]),
    alp_g2b=np.array([0, 0, 0]),
    M=5000,
    I=np.eye(3) * 4000,
    force_b=np.array([0, 0, 0]),
    momen_b=np.array([0, 0, 0]),
    dt=dt_g,
)
# graphics object
spaceship_graf = grafix_tri_prism(
    r_g2p_g=spaceship_fiz.r_g2p_g,
    side=3,
    height=6,
    q_g2b=spaceship_fiz.q_g2b,
    color=pygame.Color("green"),
)
# graphics object
x_axis = grafix_line(
    r1_g2p_g=np.array([0, 0, 0]),
    r2_g2p_g=np.array([0, 0, 0]),
    color=pygame.Color("red"),
)
y_axis = grafix_line(
    r1_g2p_g=np.array([0, 0, 0]),
    r2_g2p_g=np.array([0, 0, 0]),
    color=pygame.Color("green"),
)
z_axis = grafix_line(
    r1_g2p_g=np.array([0, 0, 0]),
    r2_g2p_g=np.array([0, 0, 0]),
    color=pygame.Color("cyan"),
)
torque_mag = 3000.0  # Nm, adjust as needed
engine_max = 500000000  # N
throttle_pct = 0

"""camera"""
cam = grafix_camera(r_g2c_g=spaceship_fiz.r_g2p_g, q_g2c=spaceship_fiz.q_g2b, fovDeg=40)
cam_dist = 30  # wrt. spaceship
cam_az = 0  # wrt. spaceship
cam_el = 0

""" ~~~ HELPERS ~~~ """


# gravity
def apply_grav(things: List[fizix_thing]):
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


# draw text
def draw_text(surface, text, x, y, color=(0, 255, 0)):
    img = font.render(text, True, color)
    surface.blit(img, (x, y))


# plume
def plume(
    throttle_pct: float,
    r_b2r_b: Vec3,  # vector from body origin to plume rotation point expressed in body coords.
    plume_length: float,
    jitter_factor: float,
    q_g2b: Quat,
    q_b2p: Quat,
    r_g2b_g: Vec3,  # vector from global origin to body origin expressed in global coords.
):
    # calculate jitter modifier
    flame_jitter = (np.random.rand() - 0.5) * throttle_pct * jitter_factor
    plume_height = throttle_pct * plume_length + flame_jitter

    # transform rotation point 2 plume origin into global frame.
    r_r2p_p = np.array([0, 0, plume_height / 2])
    r_r2p_g = quatrotate(quatinv(quatmultiply(q_g2b, q_b2p)), r_r2p_p)

    # get plume rotation point position in global frame
    r_b2r_g = quatrotate(quatinv(q_g2b), r_b2r_b)

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


""" ~~~ MAIN GAME LOOP ~~~ """
clock = pygame.time.Clock()
t = 0.0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((0, 0, 20))
    t += dt_g

    """A: physics"""
    # step 1) apply gravity to everything
    tmp = apply_grav([earth_fiz, moon_fiz, spaceship_fiz])
    earth_fiz = tmp[0]
    moon_fiz = tmp[1]
    spaceship_fiz = tmp[2]

    # step 2) get user inputs to get spacecraft engine and attitude thrusters
    keys = pygame.key.get_pressed()
    # 2a) attitude
    cmd_moment_b = np.array([0.0, 0.0, 0.0])
    if keys[pygame.K_w]:
        cmd_moment_b[0] = torque_mag
    if keys[pygame.K_s]:
        cmd_moment_b[0] = -torque_mag
    if keys[pygame.K_a]:
        cmd_moment_b[1] = -torque_mag
    if keys[pygame.K_d]:
        cmd_moment_b[1] = torque_mag
    if keys[pygame.K_q]:
        cmd_moment_b[2] = torque_mag
    if keys[pygame.K_e]:
        cmd_moment_b[2] = -torque_mag
    spaceship_fiz.momen_b = cmd_moment_b
    # 2b) engine
    force_b = np.array([0.0, 0.0, 0.0])
    if keys[pygame.K_LSHIFT]:
        throttle_pct += 1 / 100
    if keys[pygame.K_LCTRL]:
        throttle_pct -= 1 / 100
    throttle_pct = np.clip(throttle_pct, 0, 1)
    force_b[2] = throttle_pct * engine_max
    spaceship_fiz.force_b = force_b

    # step 3) integrate
    earth_fiz.step()
    moon_fiz.step()
    spaceship_fiz.step()

    """B: graphics"""
    # step 1) update ship and planet graphics objects
    earth_graf.update_state(earth_fiz.r_g2p_g, earth_fiz.q_g2b)
    moon_graf.update_state(moon_fiz.r_g2p_g, moon_fiz.q_g2b)
    spaceship_graf.update_state(spaceship_fiz.r_g2p_g, spaceship_fiz.q_g2b)

    # step 2) camera
    # 2a) get user input
    if keys[pygame.K_UP]:
        cam_el += np.deg2rad(1)
    if keys[pygame.K_DOWN]:
        cam_el -= np.deg2rad(1)
    if keys[pygame.K_LEFT]:
        cam_az += np.deg2rad(1)
    if keys[pygame.K_RIGHT]:
        cam_az -= np.deg2rad(1)

    # 2b) calculate camera position and orientation
    # orientation
    q_az = axisquat("y", cam_az)
    q_el = axisquat("x", cam_el)
    q_b2c = quatmultiply(q_az, q_el)
    q_g2c = quatmultiply(spaceship_fiz.q_g2b, q_b2c)
    # position
    r_c2b_c = np.array([0, 0, cam_dist])
    r_c2b_g = quatrotate(quatinv(q_g2c), r_c2b_c)
    r_g2c_g = spaceship_fiz.r_g2p_g - r_c2b_g
    # update camera
    cam.update_state(r_g2c_g=r_g2c_g, q_g2c=q_g2c)

    # step 3) draw stars
    for star in stars:
        star.draw(cam, screen)
    # step 4) draw ship and planets
    moon_graf.draw(cam, screen)
    earth_graf.draw(cam, screen)
    spaceship_graf.draw(cam, screen)

    # step 5) draw thrusts
    # 5a) main engine thrust
    if throttle_pct > 0:
        plume(
            throttle_pct=throttle_pct,
            r_b2r_b=np.array([0, 0.5, -3]),
            plume_length=10,
            jitter_factor=1,
            q_g2b=spaceship_fiz.q_g2b,
            q_b2p=axisquat("y", np.deg2rad(180)),
            r_g2b_g=spaceship_fiz.r_g2p_g,
        )
    # moments
    if cmd_moment_b[0] > 0:
        plume(
            throttle_pct=1,
            r_b2r_b=np.array([0, -0.8, -2.5]),
            plume_length=1,
            jitter_factor=1,
            q_g2b=spaceship_fiz.q_g2b,
            q_b2p=axisquat("x", np.deg2rad(90)),
            r_g2b_g=spaceship_fiz.r_g2p_g,
        )
    if cmd_moment_b[0] < 0:
        plume(
            throttle_pct=1,
            r_b2r_b=np.array([0, 1.75, -3]),
            plume_length=1,
            jitter_factor=1,
            q_g2b=spaceship_fiz.q_g2b,
            q_b2p=axisquat("x", np.deg2rad(270)),
            r_g2b_g=spaceship_fiz.r_g2p_g,
        )
    if cmd_moment_b[1] < 0:
        plume(
            throttle_pct=1,
            r_b2r_b=np.array([0, 0, 2.5]),
            plume_length=1,
            jitter_factor=1,
            q_g2b=spaceship_fiz.q_g2b,
            q_b2p=axisquat("y", np.deg2rad(90)),
            r_g2b_g=spaceship_fiz.r_g2p_g,
        )
    if cmd_moment_b[1] > 0:
        plume(
            throttle_pct=1,
            r_b2r_b=np.array([0, 0, 2.5]),
            plume_length=1,
            jitter_factor=1,
            q_g2b=spaceship_fiz.q_g2b,
            q_b2p=axisquat("y", np.deg2rad(270)),
            r_g2b_g=spaceship_fiz.r_g2p_g,
        )
    if cmd_moment_b[2] > 0:
        plume(
            throttle_pct=1,
            r_b2r_b=np.array([1.5, 1.75, -3]),
            plume_length=1,
            jitter_factor=1,
            q_g2b=spaceship_fiz.q_g2b,
            q_b2p=axisquat("y", np.deg2rad(90)),
            r_g2b_g=spaceship_fiz.r_g2p_g,
        )
        plume(
            throttle_pct=1,
            r_b2r_b=np.array([0, -0.8, -3]),
            plume_length=1,
            jitter_factor=1,
            q_g2b=spaceship_fiz.q_g2b,
            q_b2p=axisquat("y", np.deg2rad(270)),
            r_g2b_g=spaceship_fiz.r_g2p_g,
        )
    if cmd_moment_b[2] < 0:
        plume(
            throttle_pct=1,
            r_b2r_b=np.array([-1.5, 1.75, -3]),
            plume_length=1,
            jitter_factor=1,
            q_g2b=spaceship_fiz.q_g2b,
            q_b2p=axisquat("y", np.deg2rad(270)),
            r_g2b_g=spaceship_fiz.r_g2p_g,
        )
        plume(
            throttle_pct=1,
            r_b2r_b=np.array([0, -0.8, -3]),
            plume_length=1,
            jitter_factor=1,
            q_g2b=spaceship_fiz.q_g2b,
            q_b2p=axisquat("y", np.deg2rad(90)),
            r_g2b_g=spaceship_fiz.r_g2p_g,
        )

    # step 6) ship HUD
    sx, sy = 10, 10
    line = 18
    # Spaceship state
    offset = 0
    draw_text(screen, "SHIP:", sx, sy + offset + line * 0, (0, 255, 0))
    draw_text(
        screen,
        f"pos  = {spaceship_fiz.r_g2p_g}",
        sx,
        sy + offset + line * 1,
        (0, 255, 0),
    )
    draw_text(
        screen,
        f"vel  = {spaceship_fiz.v_g2p_g}",
        sx,
        sy + offset + line * 2,
        (0, 255, 0),
    )
    draw_text(
        screen,
        f"acc  = {spaceship_fiz.a_g2p_g}",
        sx,
        sy + offset + line * 3,
        (0, 255, 0),
    )
    draw_text(
        screen, f"quat = {spaceship_fiz.q_g2b}", sx, sy + offset + line * 4, (0, 255, 0)
    )
    draw_text(
        screen,
        f"ome  = {spaceship_fiz.ome_g2b}",
        sx,
        sy + offset + line * 5,
        (0, 255, 0),
    )

    # step 7) draw ship triad
    # 7a) get the vectors in global coordinates
    # x_body_g = (
    #     quatrotate(quatinv(spaceship_fiz.q_g2b), np.array([1, 0, 0]))
    #     + spaceship_fiz.r_g2p_g
    # )
    # y_body_g = (
    #     quatrotate(quatinv(spaceship_fiz.q_g2b), np.array([0, 1, 0]))
    #     + spaceship_fiz.r_g2p_g
    # )
    # z_body_g = (
    #     quatrotate(quatinv(spaceship_fiz.q_g2b), np.array([0, 0, 1]))
    #     + spaceship_fiz.r_g2p_g
    # )
    # # 7b) set and draw
    # x_axis.r2_g2p_g = spaceship_fiz.r_g2p_g
    # x_axis.r1_g2p_g = x_body_g
    # x_axis.draw(cam, screen)

    # y_axis.r2_g2p_g = spaceship_fiz.r_g2p_g
    # y_axis.r1_g2p_g = y_body_g
    # y_axis.draw(cam, screen)
    # z_axis.r2_g2p_g = spaceship_fiz.r_g2p_g
    # z_axis.r1_g2p_g = z_body_g
    # z_axis.draw(cam, screen)

    # step 8) update the display
    pygame.display.flip()
    clock.tick(120)

pygame.quit()
