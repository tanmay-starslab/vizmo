"""Phase 7 export + Phase 6/8 foundation tests."""

import os
import zipfile

import numpy as np
import pytest

from .test_physics_analysis import cosmo_snapshot, cosmo_data  # fixtures


def test_latex_table(tmp_path):
    from vizmo.export import export_latex_table

    rows = [("M200c", "4.109e+10 Msun"), ("R200c", "73 kpc"),
            ("f_gas", "1.000"), ("WeirdLabel_x", "7")]
    out = export_latex_table(rows, str(tmp_path / "t.tex"))
    text = open(out).read()
    assert text.startswith(r"\begin{tabular}{lcc}")
    assert r"$M_{200c}$" in text
    assert r"$4.11 \times 10^{10}$" in text
    assert r"\end{tabular}" in text
    assert r"WeirdLabel\_x" in text  # unknown labels escaped


def test_vtu_export(tmp_path):
    import xml.etree.ElementTree as ET

    from vizmo.export import export_vtk

    rng = np.random.default_rng(0)
    pos = rng.random((50, 3)) * 10
    fields = {"Temperature": rng.random(50) * 1e6,
              "Velocities": rng.standard_normal((50, 3))}
    out = export_vtk(pos, fields, str(tmp_path / "p.vtu"))
    tree = ET.parse(out)  # must be valid XML
    root = tree.getroot()
    piece = root.find(".//Piece")
    assert piece.get("NumberOfPoints") == "50"
    names = [d.get("Name") for d in root.findall(".//PointData/DataArray")]
    assert set(names) == {"Temperature", "Velocities"}
    vec = [d for d in root.findall(".//PointData/DataArray")
           if d.get("Name") == "Velocities"][0]
    assert vec.get("NumberOfComponents") == "3"


def test_zip_bundle(cosmo_data, tmp_path):
    from vizmo.analysis import (radial_profile, region_stats,
                                phase_histogram, halo_properties)
    from vizmo.export import export_all_data

    r, prof, unit = radial_profile(cosmo_data, "Density", n_bins=10)
    rows, used_r = region_stats(cosmo_data, radius_kpc=10.0)
    halo = halo_properties(cosmo_data, radius_kpc=10.0)
    ph = phase_histogram(cosmo_data, "NumberDensity", "Temperature",
                         n_bins=16)
    zp = export_all_data(
        str(tmp_path), profiles={"Density": (r, prof, "Density", unit)},
        stats_rows=rows, halo_rows=halo, phase=ph,
        metadata={"snapshot": "test", "redshift": 0.0})
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        assert "profiles/Density.csv" in names
        assert "stats.json" in names
        assert "stats_table.tex" in names
        assert "phase_diagram.fits" in names
        assert "README.txt" in names


def test_dark_theme_tokens():
    from vizmo.themes import DarkTheme

    for name, c in DarkTheme.colors().items():
        assert len(c) == 4, name
        assert all(isinstance(v, int) and 0 <= v <= 255 for v in c), name
    assert DarkTheme.CORNER_RADIUS > 0


def test_checksum_cache_key(tmp_path):
    from vizmo.data_manager import _snapshot_content_key

    p1 = tmp_path / "a.hdf5"
    p1.write_bytes(b"x" * 100000)
    k1 = _snapshot_content_key(str(p1))
    # Renaming/moving must not change the key.
    p2 = tmp_path / "renamed.hdf5"
    p1.rename(p2)
    assert _snapshot_content_key(str(p2)) == k1
    # Content change must change the key.
    p2.write_bytes(b"y" * 100000)
    assert _snapshot_content_key(str(p2)) != k1
