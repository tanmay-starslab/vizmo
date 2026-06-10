"""Publication-figure annotation and data-subset export.

annotate_screenshot() burns a colorbar, physical scale bar, and a
metadata caption into a rendered frame so the PNG is paper/talk-ready
with zero post-processing. export_region() writes the particles inside
a sphere to a standalone Gadget-style HDF5 (or CSV) cutout.
"""

import os
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _font(size):
    try:
        import matplotlib.font_manager as fm

        path = fm.findfont(fm.FontProperties(family="sans-serif"))
        return ImageFont.truetype(path, size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:
            return ImageFont.load_default()


def _ticks(lo, hi, n=5):
    """~n nicely-rounded tick values spanning [lo, hi]."""
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo]
    raw = (hi - lo) / max(n - 1, 1)
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            step = m * mag
            break
    t0 = np.ceil(lo / step) * step
    out = []
    t = t0
    while t <= hi + 1e-9 * step:
        out.append(round(t, 10))
        t += step
    return out or [lo]


def annotate_screenshot(
    src_path,
    out_path,
    cmap_name="magma",
    qty_min=0.0,
    qty_max=1.0,
    log_scale=True,
    field_label="Masses",
    unit_label="",
    kpc_per_px=None,
    caption="",
):
    """Burn colorbar + scale bar + caption into a rendered frame.

    Args:
        src_path: rendered PNG (no UI) straight from the renderer.
        out_path: annotated output PNG.
        qty_min/qty_max: color range as used by the shader (already in
            log10 when log_scale).
        kpc_per_px: physical width of one image pixel at the reference
            depth; scale bar is omitted when None.
        caption: free-form line drawn bottom-left above the scale bar.
    """
    import matplotlib

    img = Image.open(src_path).convert("RGBA")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    fs = max(14, H // 54)
    font = _font(fs)
    font_small = _font(max(11, int(fs * 0.8)))

    # --- colorbar (right edge) ---
    bar_w = max(14, W // 90)
    bar_h = int(H * 0.42)
    bx1 = W - int(W * 0.025)
    bx0 = bx1 - bar_w
    by0 = (H - bar_h) // 2
    cmap = matplotlib.colormaps[cmap_name]
    for j in range(bar_h):
        t = 1.0 - j / max(bar_h - 1, 1)
        r, g, b, _ = cmap(t)
        draw.rectangle(
            [(bx0, by0 + j), (bx1, by0 + j)],
            fill=(int(r * 255), int(g * 255), int(b * 255), 255),
        )
    draw.rectangle([(bx0, by0), (bx1, by0 + bar_h)],
                   outline=(255, 255, 255, 220), width=2)

    for tval in _ticks(qty_min, qty_max, 5):
        frac = (tval - qty_min) / max(qty_max - qty_min, 1e-30)
        ty = by0 + bar_h - int(frac * bar_h)
        draw.line([(bx0 - 6, ty), (bx0, ty)], fill=(255, 255, 255, 220), width=2)
        label = f"{tval:g}"
        bb = draw.textbbox((0, 0), label, font=font_small)
        draw.text((bx0 - 10 - (bb[2] - bb[0]), ty - (bb[3] - bb[1]) // 2 - 2),
                  label, fill=(245, 246, 250, 255), font=font_small)

    flabel = f"{field_label} [{unit_label}]" if unit_label else field_label
    if log_scale:
        flabel = f"log10 {flabel}"
    txt = Image.new("RGBA", (bar_h, fs + 8), (0, 0, 0, 0))
    tdraw = ImageDraw.Draw(txt)
    bb = tdraw.textbbox((0, 0), flabel, font=font)
    tdraw.text(((bar_h - bb[2] + bb[0]) // 2, 0), flabel,
               fill=(245, 246, 250, 255), font=font)
    txt = txt.rotate(90, expand=True)
    img.alpha_composite(txt, (bx1 + 6 - txt.width + (fs + 8), by0))

    # --- scale bar (bottom-left) ---
    if kpc_per_px is not None and kpc_per_px > 0:
        from .analysis import nice_scale_bar

        target = kpc_per_px * W * 0.25
        val_kpc, label = nice_scale_bar(target)
        bar_px = int(val_kpc / kpc_per_px)
        if 10 < bar_px < W:
            sx0, sy = int(W * 0.04), H - int(H * 0.05)
            draw.line([(sx0, sy), (sx0 + bar_px, sy)],
                      fill=(255, 255, 255, 235), width=max(2, H // 300))
            for x in (sx0, sx0 + bar_px):
                draw.line([(x, sy - fs // 2), (x, sy)],
                          fill=(255, 255, 255, 235), width=max(2, H // 300))
            bb = draw.textbbox((0, 0), label, font=font)
            draw.text((sx0 + (bar_px - bb[2] + bb[0]) // 2, sy - fs - 10),
                      label, fill=(245, 246, 250, 255), font=font)

    # --- caption (top-left) ---
    if caption:
        draw.text((int(W * 0.04) + 1, int(H * 0.035) + 1), caption,
                  fill=(0, 0, 0, 180), font=font_small)
        draw.text((int(W * 0.04), int(H * 0.035)), caption,
                  fill=(235, 238, 245, 255), font=font_small)

    img.convert("RGB").save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# FITS map export (meshoid kernel-weighted projection)
# ---------------------------------------------------------------------------

def export_fits_map(data, field="Masses", center=None, radius_kpc=None,
                    npix=512, path=None, axis="z"):
    """Project a spherical region onto an axis-aligned grid and write a
    FITS image (+ PNG quicklook).

    Uses meshoid's kernel-weighted GridSurfaceDensity, so the map is a
    proper SPH/MFM projection rather than a 2D histogram:

      field == "Masses": surface density Sigma in Msun/kpc^2
      otherwise:         mass-weighted projection of `field`
                         (Sigma_{m*f} / Sigma_m) in the field's units

    `axis` ("x" | "y" | "z") is the line-of-sight axis. Returns
    (fits_path, png_path).
    """
    from astropy.io import fits as pyfits
    from meshoid.grid_deposition import GridSurfaceDensity

    from .physics import UnitSystem, field_unit_label

    units = UnitSystem(data.header)
    if center is None:
        center = data.get_view_center()
    center = np.asarray(center, dtype=np.float64)
    r_all = np.linalg.norm(data.positions - center[None, :], axis=1)
    if radius_kpc is None:
        radius_kpc = float(np.percentile(r_all, 25.0)) * units.length_to_kpc
    r_code = radius_kpc / max(units.length_to_kpc, 1e-30)

    keep = r_all <= r_code
    if keep.sum() < 8:
        raise ValueError("Fewer than 8 particles inside the aperture")
    # Permute coordinates so the LOS is the last axis.
    order = {"x": (1, 2, 0), "y": (2, 0, 1), "z": (0, 1, 2)}[axis]
    pos = np.ascontiguousarray(
        data.positions[keep][:, order], dtype=np.float64)
    ctr = center[list(order)]
    h = np.ascontiguousarray(data.hsml[keep], dtype=np.float64)
    m = data.masses[keep].astype(np.float64)

    size = 2.0 * r_code
    sigma_m = GridSurfaceDensity(m, pos, h, ctr, size, res=npix)
    if field in (None, "Masses"):
        # code mass / code length^2 -> Msun / kpc^2
        img = (sigma_m * units.mass_to_msun / units.length_to_kpc**2)
        bunit = "Msun/kpc^2"
        fname = "SurfaceDensity"
    else:
        vals = np.asarray(data.get_field(field), dtype=np.float64)[keep]
        sigma_mf = GridSurfaceDensity(m * vals, pos, h, ctr, size, res=npix)
        with np.errstate(invalid="ignore", divide="ignore"):
            img = np.where(sigma_m > 0, sigma_mf / sigma_m, np.nan)
        bunit = field_unit_label(field) or "code units"
        fname = field

    if path is None:
        path = f"vizmo_map_{fname}_{int(time.time())}.fits"
    dkpc = 2.0 * radius_kpc / npix

    hdu = pyfits.PrimaryHDU(img.T.astype(np.float32))
    hd = hdu.header
    hd["BUNIT"] = bunit
    hd["FIELD"] = fname
    hd["CTYPE1"] = "LINEAR"
    hd["CTYPE2"] = "LINEAR"
    hd["CUNIT1"] = "kpc"
    hd["CUNIT2"] = "kpc"
    hd["CRPIX1"] = npix / 2 + 0.5
    hd["CRPIX2"] = npix / 2 + 0.5
    hd["CRVAL1"] = 0.0
    hd["CRVAL2"] = 0.0
    hd["CDELT1"] = dkpc
    hd["CDELT2"] = dkpc
    hd["LOSAXIS"] = axis
    hd["RADKPC"] = radius_kpc
    hd["CENX"] = center[0]
    hd["CENY"] = center[1]
    hd["CENZ"] = center[2]
    hd["REDSHIFT"] = float(getattr(units, "redshift", 0.0))
    hd["ORIGIN"] = "vizmo export_fits_map (meshoid GridSurfaceDensity)"
    hd["SRCFILE"] = os.path.basename(getattr(data, "path", ""))
    hdu.writeto(path, overwrite=True)

    # PNG quicklook
    png_path = os.path.splitext(path)[0] + ".png"
    try:
        from matplotlib.figure import Figure
        from matplotlib.colors import LogNorm
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        fig = Figure(figsize=(6.4, 5.6), dpi=120)
        ax = fig.add_subplot(111)
        finite = np.isfinite(img)
        pos_ok = finite & (img > 0)
        if pos_ok.sum() > 0.5 * finite.sum():
            vmin = np.percentile(img[pos_ok], 1)
            vmax = np.percentile(img[pos_ok], 99.9)
            norm = LogNorm(vmin=max(vmin, vmax * 1e-8), vmax=vmax)
        else:
            norm = None
        ext = [-radius_kpc, radius_kpc, -radius_kpc, radius_kpc]
        im = ax.imshow(img.T, origin="lower", extent=ext, cmap="magma",
                       norm=norm, interpolation="nearest")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(f"{fname} [{bunit}]")
        ax.set_xlabel("kpc")
        ax.set_ylabel("kpc")
        ax.set_title(f"LOS={axis}  R={radius_kpc:.0f} kpc")
        fig.tight_layout()
        FigureCanvasAgg(fig).print_png(png_path)
    except Exception:
        png_path = None
    return os.path.abspath(path), png_path


# ---------------------------------------------------------------------------
# Region export
# ---------------------------------------------------------------------------

def export_region(data, center=None, radius_kpc=None, path=None):
    """Write all particles within a sphere to a Gadget-style HDF5 cutout
    (or a flat CSV when `path` ends in .csv).

    Field values are copied raw (code units) so the cutout round-trips
    through vizmo and standard analysis tools; Header attrs are copied
    with corrected NumPart counts, and a VizmoCutout group records the
    selection sphere.

    Returns (path, total_particles_written).
    """
    import h5py

    from .physics import UnitSystem

    units = UnitSystem(data.header)
    if center is None:
        center = data.get_view_center()
    center = np.asarray(center, dtype=np.float64)
    if radius_kpc is None:
        r_code_all = np.linalg.norm(data.positions - center[None, :], axis=1)
        radius_kpc = float(np.percentile(r_code_all, 25.0)) * units.length_to_kpc
    r_code = radius_kpc / max(units.length_to_kpc, 1e-30)

    if path is None:
        path = f"vizmo_region_{int(time.time())}.hdf5"

    is_csv = str(path).lower().endswith(".csv")
    n_written = 0

    if is_csv:
        # Flat table: ptype, x, y, z, mass (code units) — for quick
        # spreadsheet/topcat inspection of modest regions.
        rows = []
        for p, sl in sorted(getattr(data, "_type_slices", {}).items()):
            pos = data.positions[sl]
            keep = np.linalg.norm(pos - center[None, :], axis=1) <= r_code
            pk = pos[keep]
            mk = data.masses[sl][keep]
            for (x, y, z), m in zip(pk, mk):
                rows.append(f"{p},{x:.8g},{y:.8g},{z:.8g},{m:.8g}")
            n_written += int(keep.sum())
        with open(path, "w") as f:
            f.write("ptype,x,y,z,mass\n")
            f.write("\n".join(rows))
            f.write("\n")
        return os.path.abspath(path), n_written

    with h5py.File(path, "w") as out:
        hdr = out.create_group("Header")
        npart = [0] * 6
        for p, sl in sorted(getattr(data, "_type_slices", {}).items()):
            pos = data.positions[sl]
            keep = np.linalg.norm(pos - center[None, :], axis=1) <= r_code
            nk = int(keep.sum())
            if nk == 0:
                continue
            n_written += nk
            if p < len(npart):
                npart[p] = nk
            grp = out.create_group(f"PartType{p}")
            src = data._file.get(f"PartType{p}")
            if src is None:
                grp.create_dataset("Coordinates", data=pos[keep])
                grp.create_dataset("Masses", data=data.masses[sl][keep])
                continue
            idx = np.flatnonzero(keep)
            for name in src.keys():
                try:
                    ds = src[name]
                    if not hasattr(ds, "shape") or ds.shape[0] != len(pos):
                        continue
                    grp.create_dataset(name, data=np.asarray(ds)[idx])
                except Exception:
                    continue
            if "Masses" not in grp:
                grp.create_dataset("Masses", data=data.masses[sl][keep])

        for k, v in data.header.items():
            try:
                hdr.attrs[k] = v
            except Exception:
                continue
        hdr.attrs["NumPart_ThisFile"] = np.array(npart, dtype=np.int64)
        hdr.attrs["NumPart_Total"] = np.array(npart, dtype=np.uint64)
        hdr.attrs["NumFilesPerSnapshot"] = 1
        info = out.create_group("VizmoCutout")
        info.attrs["center_code_units"] = center
        info.attrs["radius_kpc"] = radius_kpc
        info.attrs["source"] = os.path.abspath(getattr(data, "path", ""))
    return os.path.abspath(path), n_written


# ---------------------------------------------------------------------------
# LaTeX table export
# ---------------------------------------------------------------------------

LATEX_SYMBOLS = {
    "M200c": (r"$M_{200c}$", r"$\mathrm{M_\odot}$"),
    "R200c": (r"$R_{200c}$", "kpc"),
    "M500c": (r"$M_{500c}$", r"$\mathrm{M_\odot}$"),
    "R500c": (r"$R_{500c}$", "kpc"),
    "c_NFW": (r"$c_{\mathrm{NFW}}$", "--"),
    "lambda_spin": (r"$\lambda$", "--"),
    "beta_anisotropy": (r"$\beta$", "--"),
    "f_gas": (r"$f_{\mathrm{gas}}$", "--"),
    "f_cold(T<2e4K)": (r"$f_{\mathrm{cold}}$", "--"),
    "f_baryon": (r"$f_{\mathrm{b}}$", "--"),
    "dM/dt boundary": (r"$\dot{M}_{\mathrm{boundary}}$",
                       r"$\mathrm{M_\odot\,yr^{-1}}$"),
    "D/T (|eps|>0.7)": (r"$D/T$", "--"),
    "Total mass": (r"$M_{\mathrm{tot}}$", r"$\mathrm{M_\odot}$"),
    "Half-mass r": (r"$r_{1/2}$", "kpc"),
    "SFR": (r"$\mathrm{SFR}$", r"$\mathrm{M_\odot\,yr^{-1}}$"),
    "<T> (mass-wtd)": (r"$\langle T \rangle$", "K"),
    "<Z>": (r"$\langle Z \rangle$", r"$\mathrm{Z_\odot}$"),
    "<v_r>": (r"$\langle v_r \rangle$", r"$\mathrm{km\,s^{-1}}$"),
    "Radius": (r"$R_{\mathrm{ap}}$", "kpc"),
    "Particles": (r"$N_{\mathrm{part}}$", "--"),
}


def _latex_number(text):
    """'4.109e+10 Msun' -> '$4.11 \\times 10^{10}$' (3 sig figs)."""
    tok = str(text).split()[0].replace(",", "")
    try:
        v = float(tok)
    except ValueError:
        return str(text).replace("_", r"\_")
    if v == 0:
        return "$0$"
    exp = int(np.floor(np.log10(abs(v))))
    if -2 <= exp <= 3:
        return f"${v:.3g}$"
    mant = v / 10.0**exp
    return rf"${mant:.2f} \times 10^{{{exp}}}$"


def export_latex_table(rows, output_path: str) -> str:
    """Write (label, value) stat rows as a bare LaTeX tabular block.

    Only the tabular environment is emitted (no preamble), ready for
    \\input{} into a paper. Labels found in LATEX_SYMBOLS get proper
    math symbols and units; unknown labels are escaped verbatim.
    """
    lines = [r"\begin{tabular}{lcc}", r"\hline",
             r"Quantity & Value & Unit \\", r"\hline"]
    for label, value in rows:
        if str(label).startswith("---"):
            lines.append(r"\hline")
            continue
        sym, unit = LATEX_SYMBOLS.get(
            label.strip(), (label.strip().replace("_", r"\_"), "--"))
        lines.append(f"{sym} & {_latex_number(value)} & {unit} \\\\")
    lines += [r"\hline", r"\end{tabular}", ""]
    tmp = str(output_path) + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines))
    os.replace(tmp, output_path)
    return output_path


# ---------------------------------------------------------------------------
# VTK (VTU) export
# ---------------------------------------------------------------------------

def export_vtk(pos, fields, output_path: str, aperture_region=None) -> str:
    """Export particles as a VTK UnstructuredGrid (.vtu, ASCII XML).

    No external dependency: writes the (well-documented) VTU XML by
    hand. `pos` is (N, 3) in kpc; `fields` maps names to (N,) scalar or
    (N, 3) vector arrays, written as PointData. Each particle is a
    VTK_VERTEX cell, so ParaView/VisIt load the cloud directly.
    """
    pos = np.asarray(pos, dtype=np.float64)
    if aperture_region is not None:
        keep = aperture_region.contains(pos)
        pos = pos[keep]
        fields = {k: np.asarray(v)[keep] for k, v in fields.items()}
    n = len(pos)

    def arr_to_text(a):
        return "\n".join(
            " ".join(f"{x:.7g}" for x in np.atleast_1d(row))
            for row in a)

    parts = []
    parts.append('<?xml version="1.0"?>')
    parts.append('<VTKFile type="UnstructuredGrid" version="0.1" '
                 'byte_order="LittleEndian">')
    parts.append("<UnstructuredGrid>")
    parts.append(f'<Piece NumberOfPoints="{n}" NumberOfCells="{n}">')
    parts.append("<Points>")
    parts.append('<DataArray type="Float64" NumberOfComponents="3" '
                 'format="ascii">')
    parts.append(arr_to_text(pos))
    parts.append("</DataArray></Points>")
    parts.append("<PointData>")
    for name, arr in fields.items():
        arr = np.asarray(arr)
        ncomp = 3 if arr.ndim == 2 else 1
        parts.append(f'<DataArray type="Float64" Name="{name}" '
                     f'NumberOfComponents="{ncomp}" format="ascii">')
        parts.append(arr_to_text(arr))
        parts.append("</DataArray>")
    parts.append("</PointData>")
    parts.append("<Cells>")
    parts.append('<DataArray type="Int64" Name="connectivity" format="ascii">')
    parts.append(" ".join(str(i) for i in range(n)))
    parts.append("</DataArray>")
    parts.append('<DataArray type="Int64" Name="offsets" format="ascii">')
    parts.append(" ".join(str(i + 1) for i in range(n)))
    parts.append("</DataArray>")
    parts.append('<DataArray type="UInt8" Name="types" format="ascii">')
    parts.append(" ".join("1" for _ in range(n)))  # VTK_VERTEX
    parts.append("</DataArray>")
    parts.append("</Cells>")
    parts.append("</Piece></UnstructuredGrid></VTKFile>")
    tmp = str(output_path) + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(parts))
    os.replace(tmp, output_path)
    return output_path


# ---------------------------------------------------------------------------
# ZIP bundle export
# ---------------------------------------------------------------------------

def export_all_data(output_dir: str, profiles=None, stats_rows=None,
                    halo_rows=None, phase=None, sightlines=None,
                    metadata=None) -> str:
    """Bundle every computed analysis product into one ZIP.

    Writes profiles as CSVs, stats as JSON + LaTeX, the phase histogram
    as FITS, sightline columns as CSV, plus a README describing each
    file and the provenance metadata. Returns the ZIP path.
    """
    import json
    import zipfile

    ts = int(time.time())
    zpath = os.path.join(output_dir, f"vizmo_analysis_{ts}.zip")
    os.makedirs(output_dir, exist_ok=True)
    readme = ["vizmo analysis bundle", "=" * 30, ""]
    if metadata:
        for k, v in metadata.items():
            readme.append(f"{k}: {v}")
        readme.append("")

    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        if profiles:
            from .analysis import profile_to_csv
            import tempfile

            for name, (r, prof, fld, unit) in profiles.items():
                tmp = tempfile.NamedTemporaryFile(
                    "w", suffix=".csv", delete=False)
                tmp.close()
                profile_to_csv(tmp.name, r, prof, fld, unit)
                z.write(tmp.name, f"profiles/{name}.csv")
                os.unlink(tmp.name)
                readme.append(f"profiles/{name}.csv — radial profile")
        if stats_rows:
            payload = {"region_stats": dict(stats_rows),
                       "halo_properties": dict(halo_rows or []),
                       "metadata": metadata or {}}
            z.writestr("stats.json", json.dumps(payload, indent=1))
            readme.append("stats.json — region statistics + metadata")
            import tempfile

            tmp = tempfile.NamedTemporaryFile("w", suffix=".tex",
                                              delete=False)
            tmp.close()
            export_latex_table(
                list(stats_rows) + list(halo_rows or []), tmp.name)
            z.write(tmp.name, "stats_table.tex")
            os.unlink(tmp.name)
            readme.append("stats_table.tex — LaTeX tabular block")
        if phase is not None:
            import io

            from astropy.io import fits as pyfits

            buf = io.BytesIO()
            hdu = pyfits.PrimaryHDU(phase["H"].T.astype(np.float32))
            hdu.header["XFIELD"] = phase["xfield"]
            hdu.header["YFIELD"] = phase["yfield"]
            hdu.header["WEIGHT"] = phase.get("weighting", "mass")
            hdu.writeto(buf)
            z.writestr("phase_diagram.fits", buf.getvalue())
            readme.append("phase_diagram.fits — 2D phase histogram")
        if sightlines:
            import tempfile

            from .spectro import sightlines_to_csv

            tmp = tempfile.NamedTemporaryFile("w", suffix=".csv",
                                              delete=False)
            tmp.close()
            sightlines_to_csv(tmp.name, sightlines)
            z.write(tmp.name, "sightlines/columns.csv")
            os.unlink(tmp.name)
            readme.append("sightlines/columns.csv — LOS column densities")
        z.writestr("README.txt", "\n".join(readme) + "\n")
    return zpath
