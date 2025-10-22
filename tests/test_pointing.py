import numpy as np
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from sn2_v2_util import (
    import_planets,
    import_ship,
    pointing_quaternion_from_z_axis,
    quatrotate,
    quatinv,
)


def _orbital_vectors():
    dt = 1 / 120
    planets = import_planets("planets.yaml", dt)
    planet = next(p for p in planets if p.name.lower() == "earth")
    ship = import_ship("ship.yaml", dt)

    v = ship.fiz.v_g2p_g - planet.fiz.v_g2p_g
    v_hat = v / np.linalg.norm(v)

    r = ship.fiz.r_g2p_g - planet.fiz.r_g2p_g
    r_hat = r / np.linalg.norm(r)

    h = np.cross(r_hat, v_hat)
    h_hat = h / np.linalg.norm(h)

    return ship, v_hat, h_hat


def test_prograde_quaternion_aligns_velocity():
    ship, v_hat, h_hat = _orbital_vectors()

    q = pointing_quaternion_from_z_axis(v_hat, h_hat)
    thrust_dir = quatrotate(quatinv(q), ship.main_engine.t_hat_b)

    assert np.allclose(thrust_dir, v_hat, atol=1e-6)


def test_retrograde_quaternion_aligns_velocity():
    ship, v_hat, h_hat = _orbital_vectors()

    q = pointing_quaternion_from_z_axis(-v_hat, h_hat)
    thrust_dir = quatrotate(quatinv(q), ship.main_engine.t_hat_b)

    assert np.allclose(thrust_dir, -v_hat, atol=1e-6)
