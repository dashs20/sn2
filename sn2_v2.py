from sn2_v2_util import *


"""
Initialization
"""
# define global constants
dt_g = 1 / 120
G = 6.67e-11

# init pygame
pygame.init()
screen_x = 1200
screen_y = 800
screen = pygame.display.set_mode((screen_x, screen_y))
pygame.display.set_caption("sn2")
pygame.font.init()
font = pygame.font.Font("7seg.ttf", 14)
font_small = pygame.font.Font("7seg.ttf", 12)

# debug triad
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

# make planets, ship, stars
planets = import_planets("planets.yaml", dt_g)

ship = import_ship("ship.yaml", dt_g)
pointing_setting = 0  # default pointing setting is "off"
throttle_pct = 0

stars = make_stars(1000, 4e9)

# define camera
cam = grafix_camera(
    r_g2c_g=np.array([0, 0, 0]),
    fovDeg=60,
    q_g2c=np.array([1, 0, 0, 0]),
)
cam_dist = -30
cam_az = 21.70
cam_el = 156.72
panning = False
sensitivity = 0.003

"""
Main Loop
"""
clock = pygame.time.Clock()
t = 0.0
running = True

while running:
    # FIRST
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # watch for mouse clicks
        # left click
        left_click_array = np.array([0, 0, 0])
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            left_click_array = np.concatenate((np.array([1]), np.array(event.pos)))

        # right click
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            panning = True
            mouse_start_pos = pygame.mouse.get_pos()
        if event.type == pygame.MOUSEBUTTONUP and event.button == 3:
            panning = False

        # mouse wheel
        if event.type == pygame.MOUSEWHEEL:
            cam_dist -= event.y*5

        if panning:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            dx = mouse_x - mouse_start_pos[0]
            dy = mouse_y - mouse_start_pos[1]

            cam_az -= dx * sensitivity
            cam_el -= dy * sensitivity

            # Reset start point to current so it becomes relative movement each frame
            mouse_start_pos = (mouse_x, mouse_y)

    t += dt_g

    # get user input
    user = getKeyInput(throttle_pct)
    throttle_pct = user.throttle_pct
    cmd_moment_pct_b = user.cmd_moment_pct_b

    """Physics"""

    # update gravity forces on all the planets
    for cur_planet in range(len(planets)):
        planets[cur_planet].fiz.force_b = np.array([0, 0, 0])
        f_grav_g = np.array([0, 0, 0])
        for other_planet in range(len(planets)):
            if cur_planet != other_planet:
                # vector from current planet to other planet
                r_cur2other_g = (
                    planets[other_planet].fiz.r_g2p_g - planets[cur_planet].fiz.r_g2p_g
                )
                dist = np.linalg.norm(r_cur2other_g)
                r_cur2other_g_hat = r_cur2other_g / dist
                # point gravity equation
                f_mag = (
                    G
                    * (planets[cur_planet].fiz.M * planets[other_planet].fiz.M)
                    / dist**2
                )
                # add to current gravity force
                f_grav_g = f_grav_g + f_mag * r_cur2other_g_hat
        # rotate net force into planet body frame
        f_grav_b = quatrotate(planets[cur_planet].fiz.q_g2b, f_grav_g)
        # update force field of planet struct. This operation clears out the last force
        planets[cur_planet].fiz.force_b = f_grav_b

    # update gravity forces on the ship
    f_grav_g = np.array([0, 0, 0])
    for cur_planet in range(len(planets)):
        # vector from ship to current planet
        r_ship2planet_g = planets[cur_planet].fiz.r_g2p_g - ship.fiz.r_g2p_g
        dist = np.linalg.norm(r_ship2planet_g)
        r_ship2planet_g_hat = r_ship2planet_g / dist
        # point gravity equation
        f_mag = G * (planets[cur_planet].fiz.M * ship.fiz.M) / dist**2
        f_grav_g = f_grav_g + f_mag * r_ship2planet_g_hat
    # update force on ship. This operation clears out the last force.
    ship.fiz.force_b = quatrotate(ship.fiz.q_g2b, f_grav_g)

    # update thruster forces on the ship
    ship.add_thrusts(cmd_moment_pct_b, throttle_pct)

    # step each planet. also update its graphics state.
    for cur_planet in range(len(planets)):
        planets[cur_planet].fiz.step()
        planets[cur_planet].graf.update_state(
            r_g2p_g=planets[cur_planet].fiz.r_g2p_g, q_g2b=planets[cur_planet].fiz.q_g2b
        )

    # do the same for the ship.
    ship.fiz.step()
    ship.graf.update_state(ship.fiz.r_g2p_g, ship.fiz.q_g2b)

    # update camera
    # orientation
    q_az = axisquat("y", cam_az)
    q_el = axisquat("x", cam_el)
    q_b2c = quatmultiply(q_az, q_el)
    q_g2c = quatmultiply(ship.fiz.q_g2b, q_b2c)
    # position
    r_c2b_c = np.array([0, 0, cam_dist])
    r_c2b_g = quatrotate(quatinv(q_g2c), r_c2b_c)
    r_g2c_g = ship.fiz.r_g2p_g - r_c2b_g
    cam.update_state(r_g2c_g=r_g2c_g, q_g2c=q_g2c)

    """Graphics"""
    # fill screen
    screen.fill((0, 0, 20))

    # draw stars
    for star in stars:
        star.draw(cam, screen)

    # for the planets, we need to figure out how far they are from the camera and draw them in that order.
    planet_dists = []
    planet_inds = []
    for cur_planet in range(len(planets)):
        planet_dists.append(
            np.linalg.norm(planets[cur_planet].fiz.r_g2p_g - cam.r_g2c_g)
        )
        planet_inds.append(cur_planet)

    # Sort together by values in a
    planet_dists_sorted, planet_inds_sorted = zip(
        *sorted(zip(planet_dists, planet_inds), reverse=True)
    )
    planet_inds_sorted = list(planet_inds_sorted)

    for i in range(len(planets)):
        planets[planet_inds_sorted[i]].graf.draw(cam, screen)

    # draw ship
    ship.graf.draw(cam, screen)
    ship.draw_thrusts(cam, screen)

    # debug triad
    x_body_g = (
        quatrotate(quatinv(ship.fiz.q_g2b), np.array([1, 0, 0])) + ship.fiz.r_g2p_g
    )
    y_body_g = (
        quatrotate(quatinv(ship.fiz.q_g2b), np.array([0, 1, 0])) + ship.fiz.r_g2p_g
    )
    z_body_g = (
        quatrotate(quatinv(ship.fiz.q_g2b), np.array([0, 0, 1])) + ship.fiz.r_g2p_g
    )
    x_axis.r2_g2p_g = ship.fiz.r_g2p_g
    x_axis.r1_g2p_g = x_body_g
    x_axis.draw(cam, screen)
    y_axis.r2_g2p_g = ship.fiz.r_g2p_g
    y_axis.r1_g2p_g = y_body_g
    y_axis.draw(cam, screen)
    z_axis.r2_g2p_g = ship.fiz.r_g2p_g
    z_axis.r1_g2p_g = z_body_g
    z_axis.draw(cam, screen)

    # hud
    pointing_setting = hud(
        np.array([375, screen_y - 160]),
        ship,
        planets[planet_inds_sorted[-1]],
        pointing_setting,
        screen,
        left_click_array,
        font,
        font_small,
        G,
    )

    # LAST
    pygame.display.flip()
    clock.tick(120)

pygame.quit()
