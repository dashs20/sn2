from sn2_v2_util import *


"""
Initialization
"""
# define global constants
dt_g = 1 / 120
G = 6.67e-11

# init pygame
pygame.init()
screen = pygame.display.set_mode((1200, 800))
pygame.display.set_caption("sn2")

# make planets, ship, stars
planets = import_planets("planets.yaml", dt_g)
ship = import_ship("ship.yaml", dt_g)
throttle_pct = 0
stars = make_stars(1000, 4e8)

# define camera
cam = grafix_camera(
    r_g2c_g=np.array([0, 0, 0]),
    fovDeg=60,
    q_g2c=np.array([1, 0, 0, 0]),
)
cam_dist = 30
cam_az = 0
cam_el = 0

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
    t += dt_g

    # get user input
    user = getInput(throttle_pct=throttle_pct, cam_az=cam_az, cam_el=cam_el)
    cam_az = user.cam_az
    cam_el = user.cam_el
    throttle_pct = user.throttle_pct
    cmd_moment_pct_b = user.cmd_moment_pct_b

    '''Physics'''

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
    ship.add_thrusts(cmd_moment_pct_b,throttle_pct)


    # step each planet. also update its graphics state.
    for cur_planet in range(len(planets)):
        planets[cur_planet].fiz.step()
        planets[cur_planet].graf.update_state(r_g2p_g=planets[cur_planet].fiz.r_g2p_g,q_g2b=planets[cur_planet].fiz.q_g2b)

    # do the same for the ship.
    ship.fiz.step()
    ship.graf.update_state(ship.fiz.r_g2p_g,ship.fiz.q_g2b)

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
        star.draw(cam,screen)

    # for the planets, we need to figure out how far they are from the camera and draw them in that order.
    planet_dists = []
    planet_inds = []
    for cur_planet in range(len(planets)):
        planet_dists.append(np.linalg.norm(planets[cur_planet].fiz.r_g2p_g - cam.r_g2c_g))
        planet_inds.append(cur_planet)

    # Sort together by values in a
    planet_dists_sorted, planet_inds_sorted = zip(*sorted(zip(planet_dists, planet_inds), reverse=True))
    planet_inds_sorted = list(planet_inds_sorted)

    for i in range(len(planets)):
        planets[planet_inds_sorted[i]].graf.draw(cam,screen)

    # draw ship
    ship.graf.draw(cam,screen)

    # LAST
    pygame.display.flip()
    clock.tick(120)

pygame.quit()
