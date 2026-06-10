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
