import pygame
import numpy as np
from mafs import *  # quaternion utilities
from fisix import thing
from grafix import grafix_camera, grafix_sphere

# ------------------------
# STAR FIELD (fixed background)
# ------------------------
NUM_STARS = 1000
stars = []
np.random.seed(1)  # lock randomness for consistent look

for _ in range(NUM_STARS):
    # Spread stars randomly around the scene
    pos = np.random.uniform(-500, 500, size=3)
    stars.append(grafix_sphere(pos, 0.5, pygame.Color("white")))


# ------------------------
# SIMPLE GRAVITY CONSTANTS
# ------------------------
G = 1.0
M_SUN = 50.0
M_EARTH = 1.0
DT = 0.01

# ------------------------
# INITIAL CONDITIONS
# ------------------------
sun_pos = np.array([0.0, 0.0, 10.0])
earth_pos = np.array([5.0, 0.0, 10.0])
earth_vel = np.array([0.0, -3.2, 0.0])  # CW orbit
sun_vel = np.zeros(3)

I3 = np.eye(3)
Q_ID = np.array([1.0, 0.0, 0.0, 0.0])

# ------------------------
# CREATE PHYSICS OBJECTS
# ------------------------
sun = thing(
    sun_pos,
    sun_vel,
    np.zeros(3),
    Q_ID,
    np.zeros(3),
    np.zeros(3),
    M_SUN,
    I3,
    np.zeros(3),
    np.zeros(3),
    DT,
)
earth = thing(
    earth_pos,
    earth_vel,
    np.zeros(3),
    Q_ID,
    np.zeros(3),
    np.zeros(3),
    M_EARTH,
    I3,
    np.zeros(3),
    np.zeros(3),
    DT,
)


# ------------------------
# GRAVITY
# ------------------------
def gravity_force(m1, m2, r1, r2):
    d = r2 - r1
    dist = max(np.linalg.norm(d), 0.01)
    return G * m1 * m2 / (dist**2) * (d / dist)


# ------------------------
# QUAT HELPERS
# ------------------------
def quat_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2)
    return np.array([np.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s])


# ------------------------
# CAMERA SETUP
# ------------------------
pygame.init()
screen = pygame.display.set_mode((800, 800))

cam = grafix_camera(r_g2c_g=np.array([-8.0, 4.0, 2.0]), q_g2c=Q_ID.copy(), fovDeg=70)

# ------------------------
# RENDER OBJECTS
# ------------------------
sun_draw = grafix_sphere(sun_pos, 1.0, pygame.Color("yellow"))
earth_draw = grafix_sphere(earth_pos, 0.5, pygame.Color("dodgerblue"))

clock = pygame.time.Clock()
running = True

move_speed = 0.2
rot_speed = 0.02


def cam_local_direction(cam):
    # Build basis vectors from camera orientation
    # camera looks +Z in its frame
    forward = quatrotate(cam.q_g2c, np.array([0, 0, 1.0]))
    right = quatrotate(cam.q_g2c, np.array([1.0, 0, 0]))
    up = quatrotate(cam.q_g2c, np.array([0, 1.0, 0]))
    return forward, right, up


# ------------------------
# MAIN LOOP
# ------------------------
while running:
    for evt in pygame.event.get():
        if evt.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()

    # ---- ROTATE CAMERA (arrow keys)
    if keys[pygame.K_LEFT]:  # yaw left
        cam.q_g2c = quatmultiply(
            quat_from_axis_angle(np.array([0, 1, 0]), +rot_speed), cam.q_g2c
        )
    if keys[pygame.K_RIGHT]:  # yaw right
        cam.q_g2c = quatmultiply(
            quat_from_axis_angle(np.array([0, 1, 0]), -rot_speed), cam.q_g2c
        )
    if keys[pygame.K_UP]:  # pitch up
        cam.q_g2c = quatmultiply(
            quat_from_axis_angle(np.array([1, 0, 0]), +rot_speed), cam.q_g2c
        )
    if keys[pygame.K_DOWN]:  # pitch down
        cam.q_g2c = quatmultiply(
            quat_from_axis_angle(np.array([1, 0, 0]), -rot_speed), cam.q_g2c
        )
    cam.q_g2c /= np.linalg.norm(cam.q_g2c)

    # ---- MOVE CAMERA (relative to camera orientation)
    forward, right, up = cam_local_direction(cam)

    if keys[pygame.K_w]:
        cam.r_g2c_g += forward * move_speed
    if keys[pygame.K_s]:
        cam.r_g2c_g -= forward * move_speed
    if keys[pygame.K_a]:
        cam.r_g2c_g -= right * move_speed
    if keys[pygame.K_d]:
        cam.r_g2c_g += right * move_speed
    if keys[pygame.K_SPACE]:
        cam.r_g2c_g += up * move_speed
    if keys[pygame.K_LSHIFT]:
        cam.r_g2c_g -= up * move_speed

    # ---- GRAVITY
    F_e = gravity_force(M_EARTH, M_SUN, earth.r_g2p_g, sun.r_g2p_g)
    F_s = -F_e
    earth.update_forces(F_e, np.zeros(3))
    sun.update_forces(F_s, np.zeros(3))
    earth.step()
    sun.step()

    # ---- DRAW ----
    screen.fill((10, 10, 20))

    # assemble drawables (all grafix_sphere)
    drawables = []

    # add stars
    for star in stars:
        dist = np.linalg.norm(star.r_g2p_g - cam.r_g2c_g)
        drawables.append((dist, star))

    # add planets
    drawables.append((np.linalg.norm(sun.r_g2p_g - cam.r_g2c_g), sun_draw))
    drawables.append((np.linalg.norm(earth.r_g2p_g - cam.r_g2c_g), earth_draw))

    # sort back-to-front
    drawables.sort(reverse=True, key=lambda x: x[0])

    # draw in that order
    for _, obj in drawables:
        obj.draw(cam, screen)

    # draw planets
    sun_draw.update_state(sun.r_g2p_g, Q_ID)
    earth_draw.update_state(earth.r_g2p_g, Q_ID)
    sun_draw.draw(cam, screen)
    earth_draw.draw(cam, screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
