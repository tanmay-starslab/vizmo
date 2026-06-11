"""Synthetic absorption sightlines: kernel-integrated column densities
and a Trident subprocess launcher for full spectra.

Column densities for HI and total H are computed exactly from the
snapshot fields (NeutralHydrogenAbundance when present) by integrating
the M4 cubic-spline kernel along the line of sight — the same
projection meshoid uses, in closed numeric form.

Metal-ion columns (O VI, C IV, Mg II) use a deliberately crude
collisional-ionization-equilibrium approximation: Gaussian ion
fractions in log T with peak values from Gnat & Sternberg (2007).
These are order-of-magnitude diagnostics only — photoionization by
the UV background (which dominates for low-density CGM gas) is NOT
included. For publication numbers, use the Trident launcher.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np

from .physics import (
    UnitSystem,
    PROTONMASS_CGS,
    KPC_CGS,
    XH_DEFAULT,
)

# ---------------------------------------------------------------------------
# Kernel column-integral table
# ---------------------------------------------------------------------------

def _m4_kernel(u: np.ndarray) -> np.ndarray:
    """M4 cubic spline W(u) * h^3 (dimensionless), u = r/h, support u<1."""
    out = np.zeros_like(u)
    m1 = u < 0.5
    m2 = (u >= 0.5) & (u < 1.0)
    out[m1] = 1.0 - 6.0 * u[m1] ** 2 + 6.0 * u[m1] ** 3
    out[m2] = 2.0 * (1.0 - u[m2]) ** 3
    return out * (8.0 / np.pi)


def _build_column_kernel_table(n: int = 256) -> tuple:
    """c(q): dimensionless LOS column through the M4 kernel at impact
    ratio q = b/h, such that the particle's column is m/h^2 * c(q) and
    the kernel integral satisfies  ∫0^1 c(q) 2 pi q dq = 1."""
    q = np.linspace(0.0, 1.0, n)
    c = np.zeros(n)
    z = np.linspace(-1.0, 1.0, 4001)
    dz = z[1] - z[0]
    for i, qi in enumerate(q):
        r = np.sqrt(qi**2 + z**2)
        c[i] = _m4_kernel(r).sum() * dz
    return q, c


_QGRID, _CGRID = _build_column_kernel_table()


def column_kernel(q: np.ndarray) -> np.ndarray:
    """Interpolated c(q); zero outside the kernel support."""
    return np.where(q < 1.0, np.interp(q, _QGRID, _CGRID), 0.0)


# ---------------------------------------------------------------------------
# Crude CIE ion fractions (Gaussian in log T)
# ---------------------------------------------------------------------------

# (peak fraction, log10 T_peak, sigma_dex) — peaks from Gnat &
# Sternberg (2007) CIE tables; Gaussian widths are approximate.
# Solar abundances (Asplund 2009): n_X/n_H.
CIE_IONS = {
    "OVI": {"f_peak": 0.20, "logT_peak": 5.45, "sigma": 0.18,
            "abund": 4.90e-4},
    "CIV": {"f_peak": 0.30, "logT_peak": 5.00, "sigma": 0.18,
            "abund": 2.69e-4},
    "MgII": {"f_peak": 0.30, "logT_peak": 4.15, "sigma": 0.25,
             "abund": 3.98e-5},
}


def cie_ion_fraction(ion: str, t_kelvin: np.ndarray) -> np.ndarray:
    """Crude CIE Gaussian ion fraction f_ion(T). Order-of-magnitude
    only; no photoionization."""
    p = CIE_IONS[ion]
    logt = np.log10(np.clip(np.asarray(t_kelvin, dtype=np.float64),
                            10.0, None))
    return p["f_peak"] * np.exp(
        -0.5 * ((logt - p["logT_peak"]) / p["sigma"]) ** 2)


# ---------------------------------------------------------------------------
# Sightline
# ---------------------------------------------------------------------------

@dataclass
class Sightline:
    start: np.ndarray            # (3,) code units
    end: np.ndarray              # (3,) code units
    label: str = "SL-001"
    color: tuple = (120, 220, 255, 255)
    impact_b_kpc: float | None = None
    NHI: float | None = None     # cm^-2
    N_total_H: float | None = None
    N_OVI: float | None = None
    N_MgII: float | None = None
    N_CIV: float | None = None
    trident_spectrum_path: str | None = None
    extra: dict = field(default_factory=dict)

    def summary(self) -> str:
        def lg(v):
            return f"{np.log10(v):.1f}" if v and v > 0 else "-"

        return (f"{self.label}: log NH={lg(self.N_total_H)} "
                f"NHI={lg(self.NHI)} OVI={lg(self.N_OVI)} "
                f"CIV={lg(self.N_CIV)} MgII={lg(self.N_MgII)}")


def _point_segment_geometry(pos, a, b):
    """Perpendicular distance of points to segment a-b plus the
    along-segment parameter t in [0, 1]."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 <= 0:
        d = np.linalg.norm(pos - a[None, :], axis=1)
        return d, np.zeros(len(pos))
    t = np.clip(((pos - a[None, :]) @ ab) / L2, 0.0, 1.0)
    closest = a[None, :] + t[:, None] * ab[None, :]
    return np.linalg.norm(pos - closest, axis=1), t


def compute_los_column_densities(
    sightline: Sightline,
    data,
    n_h_field: str = "NeutralHydrogenAbundance",
) -> Sightline:
    """Kernel-integrated column densities along the sightline.

    N = sum_i (m_i X_H / m_p) * x_i * c(b_i/h_i) / h_i^2  [cm^-2]
    where x_i is the species fraction per hydrogen atom: 1 for total H,
    NeutralHydrogenAbundance for HI, and abundance * f_CIE(T) for the
    metal ions (crude — see module docstring).

    Populates the sightline in place and returns it.
    """
    units = UnitSystem(data.header)
    pos = data.positions
    a = np.asarray(sightline.start, dtype=np.float64)
    b = np.asarray(sightline.end, dtype=np.float64)

    h = data.hsml.astype(np.float64)
    # Cheap prefilter box around the segment.
    lo = np.minimum(a, b) - h.max()
    hi = np.maximum(a, b) + h.max()
    cand = np.flatnonzero(np.all((pos >= lo) & (pos <= hi), axis=1))
    if cand.size == 0:
        sightline.NHI = sightline.N_total_H = 0.0
        sightline.N_OVI = sightline.N_CIV = sightline.N_MgII = 0.0
        return sightline

    d, _ = _point_segment_geometry(pos[cand], a, b)
    hit = d < h[cand]
    cand = cand[hit]
    if cand.size == 0:
        sightline.NHI = sightline.N_total_H = 0.0
        sightline.N_OVI = sightline.N_CIV = sightline.N_MgII = 0.0
        return sightline
    d = d[hit]

    h_cm = h[cand] * units.unit_length_cgs * units.a / units.h
    m_g = (data.masses[cand].astype(np.float64)
           * units.unit_mass_cgs / units.h)
    # Per-particle coefficient: H atoms per cm^2 before the kernel
    # geometry factor c(q)/h^2.
    coeff = m_g * XH_DEFAULT / PROTONMASS_CGS

    from . import fast_ops

    if fast_ops.HAVE_NUMBA:
        # numba path: kernel-column geometry recomputed per species in
        # code units, then rescaled to cm^-2 (h_code^2 -> h_cm^2).
        len_cm = units.unit_length_cgs * units.a / units.h

        def _col(vals):
            raw = fast_ops.los_column_density(a, b, pos[cand], vals,
                                              h[cand])
            return float(raw / len_cm**2)

        sightline.N_total_H = _col(coeff)
        n_h_col = None
    else:
        q = d / h[cand]
        ck = column_kernel(q)
        n_h_col = coeff * ck / h_cm**2
        sightline.N_total_H = float(n_h_col.sum())

    avail = set(data.available_fields())
    if n_h_field in avail:
        x_hi = np.clip(np.asarray(data.get_field(n_h_field),
                                  dtype=np.float64)[cand], 0.0, 1.0)
        sightline.NHI = (_col(coeff * x_hi) if n_h_col is None
                         else float((n_h_col * x_hi).sum()))
    else:
        sightline.NHI = None

    try:
        t = np.asarray(data.get_field("Temperature"),
                       dtype=np.float64)[cand]
        if "MetallicityZsun" in data.available_fields_with_derived():
            zfac = np.clip(np.asarray(
                data.get_field("MetallicityZsun"),
                dtype=np.float64)[cand], 0.0, None)
        else:
            zfac = np.ones_like(t)
        for ion, attr in (("OVI", "N_OVI"), ("CIV", "N_CIV"),
                          ("MgII", "N_MgII")):
            frac = cie_ion_fraction(ion, t)
            abund = CIE_IONS[ion]["abund"]
            if n_h_col is None:
                setattr(sightline, attr,
                        _col(coeff * abund * zfac * frac))
            else:
                setattr(sightline, attr,
                        float((n_h_col * abund * zfac * frac).sum()))
    except Exception:
        sightline.N_OVI = sightline.N_CIV = sightline.N_MgII = None

    center = data.get_view_center()
    d_c, _ = _point_segment_geometry(center[None, :], a, b)
    sightline.impact_b_kpc = float(d_c[0] * units.length_to_kpc)
    return sightline


# ---------------------------------------------------------------------------
# Trident subprocess launcher
# ---------------------------------------------------------------------------

_TRIDENT_SCRIPT = '''
import sys
import numpy as np
import yt
import trident

snapshot, x0, y0, z0, x1, y1, z1, outbase = sys.argv[1:9]
ions = sys.argv[9].split(",")
ds = yt.load(snapshot)
ray = trident.make_simple_ray(
    ds,
    start_position=np.array([float(x0), float(y0), float(z0)]),
    end_position=np.array([float(x1), float(y1), float(z1)]),
    lines=ions,
    ftype="PartType0",
)
sg = trident.SpectrumGenerator(lambda_min=1000, lambda_max=1600,
                               dlambda=0.01)
sg.make_spectrum(ray, lines=ions)
sg.save_spectrum(outbase + ".h5")
sg.plot_spectrum(outbase + ".png")
open(outbase + ".done", "w").write("ok")
'''


def launch_trident_spectrum(
    sightline: Sightline,
    snapshot_path: str,
    output_dir: str,
    ion_list=("H I 1216", "O VI 1032", "C IV 1548", "Mg II 2796"),
) -> subprocess.Popen:
    """Launch Trident in a background subprocess.

    Writes a small script to a temp file and runs it with the current
    interpreter. Completion is signalled by '<outbase>.done'; the app
    polls for it. Raises ImportError immediately when trident is not
    importable in this environment.
    """
    try:
        import trident  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "trident is not installed — pip install trident "
            "(needs yt and ~400 MB of ion tables on first run)") from e

    os.makedirs(output_dir, exist_ok=True)
    outbase = os.path.join(output_dir, sightline.label.replace(" ", "_"))
    script = tempfile.NamedTemporaryFile(
        "w", suffix="_vizmo_trident.py", delete=False)
    script.write(_TRIDENT_SCRIPT)
    script.close()
    args = [sys.executable, script.name, snapshot_path,
            *[f"{v:.10g}" for v in sightline.start],
            *[f"{v:.10g}" for v in sightline.end],
            outbase, ",".join(ion_list)]
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    sightline.trident_spectrum_path = outbase + ".h5"
    sightline.extra["trident_done_sentinel"] = outbase + ".done"
    sightline.extra["trident_started"] = time.time()
    return proc


def sightlines_to_csv(path: str, sightlines: list) -> str:
    """Write all sightline column densities to CSV."""
    import csv

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "impact_b_kpc", "log_NH", "log_NHI",
                    "log_NOVI", "log_NCIV", "log_NMgII"])

        def lg(v):
            return f"{np.log10(v):.3f}" if v and v > 0 else ""

        for s in sightlines:
            w.writerow([s.label, f"{s.impact_b_kpc:.2f}"
                        if s.impact_b_kpc is not None else "",
                        lg(s.N_total_H), lg(s.NHI), lg(s.N_OVI),
                        lg(s.N_CIV), lg(s.N_MgII)])
    return path


# ---------------------------------------------------------------------------
# Spectrum viewing support (Section 3.D)
# ---------------------------------------------------------------------------

def load_spectrum(path):
    """Load a Trident spectrum (.h5 from SpectrumGenerator or a 2-col
    FITS) -> (velocity_kms or wavelength, flux) float arrays."""
    import numpy as np

    if str(path).endswith((".h5", ".hdf5")):
        import h5py

        with h5py.File(path, "r") as f:
            wav = np.asarray(f["wavelength"])
            flux = np.asarray(f["flux"])
        return wav, flux
    from astropy.io import fits as pyfits

    with pyfits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is None:
                continue
            d = np.asarray(hdu.data)
            if d.ndim == 2 and d.shape[0] == 2:
                return d[0].astype(float), d[1].astype(float)
            if hasattr(hdu, "columns") and len(hdu.columns) >= 2:
                names = [c.name for c in hdu.columns]
                return (np.asarray(hdu.data[names[0]], dtype=float),
                        np.asarray(hdu.data[names[1]], dtype=float))
    raise ValueError(f"no spectrum table found in {path}")


# VoigtFit summary lines look like:
#   "HI : logN = 14.23 +/- 0.05, b = 25.4 +/- 2.1, v = -45.2"
VOIGT_LINE_RE = __import__("re").compile(
    r"logN\s*=\s*([-\d.]+).*?b\s*=\s*([-\d.]+).*?v\s*=\s*([-\d.]+)")


def parse_voigtfit_components(text):
    """Extract (logN, b_kms, v_kms) tuples from VoigtFit output text."""
    out = []
    for line in str(text).splitlines():
        m = VOIGT_LINE_RE.search(line)
        if m:
            out.append((float(m.group(1)), float(m.group(2)),
                        float(m.group(3))))
    return out


COMPARE_COLORS = [(0.42, 0.72, 1.0), (1.0, 0.62, 0.25),
                  (0.45, 0.9, 0.5), (0.95, 0.4, 0.75),
                  (0.95, 0.9, 0.4), (0.7, 0.6, 1.0)]


def compare_color(i):
    """Distinct overlay color for sightline index i (cycles)."""
    return COMPARE_COLORS[i % len(COMPARE_COLORS)]


_VOIGTFIT_SCRIPT = '''
import json
import sys

import numpy as np

spec_path, ion, z, outbase = sys.argv[1:5]
import VoigtFit

ds = VoigtFit.DataSet(float(z))
import h5py

with h5py.File(spec_path, "r") as f:
    wl = np.asarray(f["wavelength"])
    flux = np.asarray(f["flux"])
ds.add_data(wl, flux, 299792.458 / np.median(wl), err=np.full_like(flux, 0.02))
ds.add_line(ion)
ds.prepare_dataset()
popt, chi2 = ds.fit()
comps = []
for line in str(ds.print_results()).splitlines():
    comps.append(line)
open(outbase + ".log", "w").write("\\n".join(comps))
open(outbase + ".done", "w").write("ok")
'''


def launch_voigtfit(sightline, ion="HI 1216", redshift=0.0,
                    output_dir="spectra"):
    """Launch VoigtFit on a sightline's Trident spectrum in a
    subprocess (sentinel-file completion, same pattern as the Trident
    launcher). Raises ImportError when VoigtFit isn't installed and
    ValueError when no spectrum exists yet."""
    try:
        import VoigtFit  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "VoigtFit is not installed — pip install VoigtFit") from e
    if not (sightline.trident_spectrum_path
            and os.path.exists(sightline.trident_spectrum_path)):
        raise ValueError("run Trident first: no spectrum on disk")
    os.makedirs(output_dir, exist_ok=True)
    outbase = os.path.join(
        output_dir, sightline.label.replace(" ", "_") + "_voigt")
    script = tempfile.NamedTemporaryFile(
        "w", suffix="_vizmo_voigt.py", delete=False)
    script.write(_VOIGTFIT_SCRIPT)
    script.close()
    proc = subprocess.Popen(
        [sys.executable, script.name, sightline.trident_spectrum_path,
         ion, f"{redshift:.6f}", outbase],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sightline.extra["voigt_done_sentinel"] = outbase + ".done"
    sightline.extra["voigt_log"] = outbase + ".log"
    return proc


def voigt_gaussian_profile(v_kms, logN, b_kms, v0_kms,
                           tau_scale=1e-13):
    """Display-only Gaussian optical-depth profile for one fitted
    component: flux = exp(-tau), tau = tau_scale*10^logN/b *
    exp(-(v-v0)^2/b^2). Used to overlay fit components on the
    spectrum plot — not for quantitative radiative transfer."""
    tau = (tau_scale * 10.0**logN / max(b_kms, 1.0)
           * np.exp(-((np.asarray(v_kms) - v0_kms) / b_kms) ** 2))
    return np.exp(-np.clip(tau, 0, 50))


def save_spectrum_pdf(sightline, out_path, snapshot_path="",
                      redshift=0.0):
    """Publication PDF of the sightline's Trident spectrum with
    continuum, trough shading, fitted components (when present), and
    a metadata footer."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    w, flux = load_spectrum(sightline.trident_spectrum_path)
    fig = Figure(figsize=(8, 4.5), dpi=150)
    ax = fig.add_subplot(111)
    ax.plot(w, flux, lw=0.9, color="k")
    ax.axhline(1.0, ls="--", lw=0.8, color="gray")
    ax.fill_between(w, flux, 1.0, where=flux < 0.9, color="C0",
                    alpha=0.25, linewidth=0)
    comps = sightline.extra.get("voigt_components") or []
    for k, (logN, b, v0) in enumerate(comps):
        ax.axvline(np.median(w) * (1 + v0 / 299792.458), ls=":",
                   lw=0.8, color=f"C{k + 1}",
                   label=f"logN={logN:.2f} b={b:.0f} v={v0:+.0f}")
    if comps:
        ax.legend(fontsize=7)
    ax.set_xlabel("wavelength [A]")
    ax.set_ylabel("F / F_continuum")
    ax.set_ylim(0, 1.25)
    b_txt = (f"b = {sightline.impact_b_kpc:.0f} kpc"
             if sightline.impact_b_kpc is not None else "")
    fig.text(0.01, 0.01,
             f"{os.path.basename(snapshot_path)}  z={redshift:.3f}  "
             f"{sightline.label}  {b_txt}", fontsize=6, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig)
    return out_path
