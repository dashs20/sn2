import pygame
import numpy as np
from math import sin, cos
from grafix import *

pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("grafix demo with setters")

# ---- Camera ----
cam = grafix_camera(
    r_g2c_g=np.array([0, 0, 0]), q_g2c=np.array([1.0, 0.0, 0.0, 0.0]), fovDeg=90
)

# ---- Create objects ONCE ----
rec = grafix_rec_prism(
    r_g2p_g=np.zeros(3),
    size=np.array([1.0, 0.6, 0.4]),
    q_g2b=np.array([1.0, 0.0, 0.0, 0.0]),
    color=pygame.Color("white"),
)

tri = grafix_tri_prism(
    r_g2p_g=np.zeros(3),
    side=1.6,
    height=2.4,
    q_g2b=np.array([1.0, 0.0, 0.0, 0.0]),
    color=pygame.Color("cyan"),
)

sphere = grafix_sphere(r_g2p_g=np.zeros(3), radius=0.22, color=pygame.Color("red"))

# ---- Animation loop ----
clock = pygame.time.Clock()
t = 0.0
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill((15, 15, 25))
    t += 0.02

    # ---- Shared motion path ----
    center_path = np.array([2.0 * sin(t), 1.0 * cos(t * 0.7), 6.0 + 2.0 * cos(t)])

    # ---- Rotation ----
    qz = np.array([np.cos(t / 2), 0, 0, np.sin(t / 2)])
    qx = np.array([np.cos(t / 3), np.sin(t / 3), 0, 0])
    q_spin = quatmultiply(qz, qx)

    # ---- Update & draw rectangular prism ----
    rec.update_state(center_path + np.array([1.8, 0, 0]), q_spin)
    rec.draw(cam, screen)

    # ---- Update & draw triangular prism ----
    tri.update_state(center_path + np.array([-1.8, 0, 0]), q_spin)
    tri.draw(cam, screen)

    # ---- Update & draw sphere ----
    sphere.update_state(center_path, np.array([1, 0, 0, 0]))  # no rotation needed
    sphere.draw(cam, screen)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
