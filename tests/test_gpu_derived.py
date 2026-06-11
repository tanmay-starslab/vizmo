"""GPU derived-field equivalence (Item 2). Skips without an adapter."""

import numpy as np
import pytest


def test_gpu_temperature_matches_cpu():
    import wgpu

    try:
        adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance")
        device = adapter.request_device_sync()
    except Exception:
        pytest.skip("no wgpu adapter")

    from vizmo.gpu_compute import GPUCompute
    from vizmo.physics import (UnitSystem, mean_molecular_weight,
                               PROTONMASS_CGS, BOLTZMANN_CGS,
                               XH_DEFAULT)

    gc = GPUCompute(device)
    units = UnitSystem({"Time": 1.0, "HubbleParam": 0.6774,
                        "Omega0": 0.3089, "OmegaLambda": 0.6911,
                        "UnitLength_in_cm": 3.085678e21,
                        "UnitMass_in_g": 1.989e43,
                        "UnitVelocity_in_cm_per_s": 1e5})
    rng = np.random.default_rng(0)
    n = 10000
    u = rng.uniform(10, 1e4, n).astype(np.float32)
    xe = rng.uniform(0.0, 1.2, n).astype(np.float32)
    rho = rng.uniform(1e-9, 1e-3, n).astype(np.float32)

    t_gpu, nh_gpu = gc.compute_derived_fields(u, xe, rho, units)
    assert gc.derived_fields_gpu is True

    # CPU reference (physics.py formulas, X_H = 0.76 default).
    mu = mean_molecular_weight(xe.astype(np.float64), XH_DEFAULT)
    u_cgs = u.astype(np.float64) * units.u_to_cgs
    t_cpu = (2.0 / 3.0) * u_cgs * mu * PROTONMASS_CGS / BOLTZMANN_CGS
    nh_cpu = (rho.astype(np.float64) * units.density_to_cgs
              * XH_DEFAULT / PROTONMASS_CGS)
    assert np.allclose(t_gpu, t_cpu, rtol=2e-4)
    assert np.allclose(nh_gpu, nh_cpu, rtol=2e-4)
