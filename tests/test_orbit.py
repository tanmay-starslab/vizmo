"""Orbit integration tests: circular closure, energy conservation,
NFW recovery, Jacobi radius."""

import numpy as np
import pytest

from vizmo.analysis import G_KPC_KMS2_MSUN
from vizmo.orbit import (
    integrate_orbit, compute_orbital_properties, fit_nfw_to_mass_profile,
    _nfw_potential_factory, jacobi_radius, generate_mock_stream,
    KMS_GYR_TO_KPC,
)


def point_mass_phi(M):
    def phi(xyz):
        xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
        r = np.linalg.norm(xyz, axis=1)
        out = -G_KPC_KMS2_MSUN * M / np.maximum(r, 1e-12)
        return out if out.size > 1 else float(out[0])
    return phi


def test_circular_orbit_closes():
    M = 1e12  # Msun
    r0 = 50.0  # kpc
    vc = np.sqrt(G_KPC_KMS2_MSUN * M / r0)  # km/s
    phi = point_mass_phi(M)
    T = 2 * np.pi * r0 / vc / KMS_GYR_TO_KPC  # Gyr
    o = integrate_orbit([r0, 0, 0], [0, vc, 0], phi,
                        t_end_gyr=T, n_steps=2000)
    # After one period the particle returns to within 1% of r0.
    assert np.linalg.norm(o["pos"][-1] - np.array([r0, 0, 0])) < 0.01 * r0
    # Radius stays circular throughout.
    assert np.abs(o["r"] - r0).max() < 1e-3 * r0


def test_energy_conservation():
    M = 1e12
    phi = point_mass_phi(M)
    # Eccentric orbit: tangential speed = 0.6 v_circ.
    r0 = 80.0
    vc = np.sqrt(G_KPC_KMS2_MSUN * M / r0)
    o = integrate_orbit([r0, 0, 0], [0, 0.6 * vc, 0], phi,
                        t_end_gyr=3.0, n_steps=3000)
    dE = np.abs(o["E"] - o["E"][0]) / np.abs(o["E"][0])
    assert dE.max() < 1e-3  # 0.1%


def test_orbital_properties_eccentric():
    M = 1e12
    phi = point_mass_phi(M)
    r0 = 80.0
    vc = np.sqrt(G_KPC_KMS2_MSUN * M / r0)
    o = integrate_orbit([r0, 0, 0], [0, 0.6 * vc, 0], phi,
                        t_end_gyr=5.0, n_steps=4000)
    props = compute_orbital_properties(o, phi=phi)
    # Keplerian analytic: launch at apocenter with v = 0.6 vc.
    # specific E and L give r_peri/r_apo: r_apo = r0 (launch at apo
    # since v < vc), r_peri from L,E.
    assert props["r_apo"] == pytest.approx(r0, rel=0.01)
    E = 0.5 * (0.6 * vc) ** 2 - G_KPC_KMS2_MSUN * M / r0
    L = r0 * 0.6 * vc
    # Solve 1/2 L^2/r^2 - GM/r = E for r (quadratic in 1/r).
    GM = G_KPC_KMS2_MSUN * M
    a, b, c = L**2 / 2, -GM, -E
    roots = np.roots([a, b, c])  # in 1/r
    r_peri_analytic = 1.0 / max(roots)
    assert props["r_peri"] == pytest.approx(r_peri_analytic, rel=0.02)
    # Circularity < 1 for an eccentric orbit, > 0 for prograde.
    assert props["circularity"] is not None
    assert 0.0 < props["circularity"] < 1.0


def test_nfw_fit_recovery():
    rho_s, r_s = 5e6, 20.0  # Msun/kpc^3, kpc
    r = np.geomspace(1.0, 200.0, 64)
    x = r / r_s
    m = 4 * np.pi * rho_s * r_s**3 * (np.log(1 + x) - x / (1 + x))
    rho_fit, rs_fit = fit_nfw_to_mass_profile(r, m)
    assert rho_fit == pytest.approx(rho_s, rel=0.05)
    assert rs_fit == pytest.approx(r_s, rel=0.05)
    # The factory matches the exact NFW potential formula.
    phi = _nfw_potential_factory(rho_fit, rs_fit)
    r_test = 50.0
    expected = (-4 * np.pi * G_KPC_KMS2_MSUN * rho_fit * rs_fit**3
                * np.log(1 + r_test / rs_fit) / r_test)
    assert phi(np.array([r_test, 0, 0])) == pytest.approx(expected, rel=1e-9)


def test_jacobi_radius_analytic():
    # Uniform-density host: M(<r) = M_tot (r/R)^3 — r_J independent of r:
    # r_J = r (m_sat / (3 M(<r)))^(1/3) = R (m_sat / 3 M_tot)^(1/3).
    M_tot, R = 1e12, 100.0
    m_sat = 1e9
    for r in (20.0, 50.0, 80.0):
        m_enc = M_tot * (r / R) ** 3
        rj = jacobi_radius(r, m_sat, m_enc)
        expected = R * (m_sat / (3 * M_tot)) ** (1 / 3)
        assert rj == pytest.approx(expected, rel=1e-12)


def test_mock_stream_strips():
    M = 1e12
    phi = point_mass_phi(M)
    r0 = 60.0
    vc = np.sqrt(G_KPC_KMS2_MSUN * M / r0)
    stream = generate_mock_stream(
        [r0, 0, 0], [0, 0.8 * vc, 0], phi, m_satellite=1e8,
        n_tracers=24, t_end_gyr=2.0, n_steps=300)
    assert stream["positions"].shape == (24, 300, 3)
    assert stream["r_jacobi"] > 0
    # Some tracers strip within 2 Gyr around a 1e8 satellite.
    assert np.isfinite(stream["stripping_time"]).sum() > 5
