"""Physical unit handling and derived (computed) fields.

UnitSystem converts code units -> physical (CGS / astro) units using the
snapshot header, handling cosmological factors (a, h) when the snapshot
is detected to be cosmological (Gadget/AREPO/TNG conventions).

DERIVED_FIELDS registers scientifically useful fields computed from raw
snapshot fields (Temperature, number density, pressure, entropy, radial
velocity, ...). SnapshotData exposes them through available_fields() /
get_field() so every render mode (surface density, weighted average,
variance, composite) works on them transparently.
"""

import numpy as np

# --- CGS constants ---
KPC_CGS = 3.0856775814913673e21
MSUN_CGS = 1.988409870698051e33
PROTONMASS_CGS = 1.67262192369e-24
BOLTZMANN_CGS = 1.380649e-16
GAMMA = 5.0 / 3.0
XH_DEFAULT = 0.76  # hydrogen mass fraction when no abundance field exists
ZSUN = 0.0127  # solar metal mass fraction (TNG/GFM convention)
SEC_PER_GYR = 3.15576e16
KM_CGS = 1.0e5


class UnitSystem:
    """Code-unit -> physical-unit conversions from a snapshot header.

    Cosmological snapshots (detected via ComovingIntegrationOn, or the
    HubbleParam/OmegaLambda/Time<=1 heuristic) get the standard Gadget
    comoving-coordinate corrections:

        length_phys   = code * UnitLength * a / h
        mass_phys     = code * UnitMass / h
        density_phys  = code * UnitMass/UnitLength^3 * h^2 / a^3
        velocity_phys = code * UnitVelocity * sqrt(a)   (peculiar)
        u (internal energy) is physical already (Gadget convention).
    """

    def __init__(self, header):
        h = header or {}
        self.unit_length_cgs = float(h.get("UnitLength_in_cm", KPC_CGS) or KPC_CGS)
        self.unit_mass_cgs = float(h.get("UnitMass_in_g", 1.989e43) or 1.989e43)
        self.unit_velocity_cgs = float(
            h.get("UnitVelocity_in_cm_per_s", KM_CGS) or KM_CGS
        )
        self.unit_time_cgs = self.unit_length_cgs / self.unit_velocity_cgs

        hub = float(h.get("HubbleParam", 1.0) or 1.0)
        self.omega_m = float(h.get("Omega0", 0.0) or 0.0)
        self.omega_l = float(h.get("OmegaLambda", 0.0) or 0.0)
        time = float(h.get("Time", 1.0) or 1.0)

        if "ComovingIntegrationOn" in h:
            self.cosmological = bool(h["ComovingIntegrationOn"])
        else:
            # Heuristic: a cosmological run has 0 < h < 1, a dark-energy
            # density, and Time used as the scale factor a <= ~1.
            self.cosmological = (
                0.0 < hub < 1.0 and self.omega_l > 0.0 and 0.0 < time <= 1.01
            )

        self.h = hub if (self.cosmological and hub > 0) else 1.0
        self.a = time if self.cosmological else 1.0
        zr = h.get("Redshift", None)
        self.redshift = float(zr) if zr is not None else (1.0 / self.a - 1.0)

    # --- scalar conversion factors (multiply code values) ---
    @property
    def length_to_kpc(self):
        return self.unit_length_cgs * self.a / self.h / KPC_CGS

    @property
    def mass_to_msun(self):
        return self.unit_mass_cgs / self.h / MSUN_CGS

    @property
    def velocity_to_kms(self):
        return self.unit_velocity_cgs * np.sqrt(self.a) / KM_CGS

    @property
    def density_to_cgs(self):
        return (
            self.unit_mass_cgs
            / self.unit_length_cgs**3
            * self.h**2
            / self.a**3
        )

    @property
    def u_to_cgs(self):
        """Specific internal energy: physical in the Gadget convention."""
        return self.unit_velocity_cgs**2

    @property
    def bfield_to_gauss(self):
        """code B -> Gauss: sqrt(UnitPressure) * h / a^2 (AREPO/TNG)."""
        unit_pressure = self.unit_mass_cgs / (
            self.unit_length_cgs * self.unit_time_cgs**2
        )
        return np.sqrt(unit_pressure) * self.h / self.a**2

    @property
    def time_to_gyr(self):
        return self.unit_time_cgs / self.h / SEC_PER_GYR

    def lookback_gyr(self, a_form):
        """Age of stars formed at scale factor a_form, in Gyr (flat LCDM).

        Uses the closed form t(a) = 2/(3 H0 sqrt(OL)) asinh(sqrt(OL/Om) a^1.5).
        Returns zeros for non-cosmological snapshots (caller should prefer
        Time - formation_time there).
        """
        if not self.cosmological or self.omega_l <= 0 or self.omega_m <= 0:
            return np.zeros_like(np.asarray(a_form, dtype=np.float64))
        H0_cgs = self.h * 100.0 * KM_CGS / (1000.0 * KPC_CGS)  # 100h km/s/Mpc
        pref = 2.0 / (3.0 * H0_cgs * np.sqrt(self.omega_l))
        rat = np.sqrt(self.omega_l / self.omega_m)

        def t_of_a(a):
            a = np.clip(np.asarray(a, dtype=np.float64), 1e-8, None)
            return pref * np.arcsinh(rat * a**1.5)

        return (t_of_a(self.a) - t_of_a(a_form)) / SEC_PER_GYR


def mean_molecular_weight(electron_abundance=None, x_h=XH_DEFAULT):
    """mu for a H/He plasma: 4 / (1 + 3 X_H + 4 X_H x_e)."""
    if electron_abundance is None:
        electron_abundance = 1.0 + 0.5 * (1.0 - x_h) / x_h  # fully ionized
    return 4.0 / (1.0 + 3.0 * x_h + 4.0 * x_h * electron_abundance)


def _xh(get, ptype_fields):
    """Hydrogen mass fraction array or scalar default."""
    if "GFM_Metals[0]" in ptype_fields:
        return np.clip(get("GFM_Metals[0]"), 0.1, 1.0)
    if "Metallicity[0]" in ptype_fields:  # GIZMO: He fraction at [1]
        z = get("Metallicity[0]")
        return np.clip(1.0 - z - 0.25, 0.1, 1.0)
    return XH_DEFAULT


def _temperature(get, units, fields):
    u_cgs = get("InternalEnergy").astype(np.float64) * units.u_to_cgs
    xe = get("ElectronAbundance") if "ElectronAbundance" in fields else None
    mu = mean_molecular_weight(xe, _xh(get, fields))
    t = (GAMMA - 1.0) * u_cgs * mu * PROTONMASS_CGS / BOLTZMANN_CGS
    return t.astype(np.float32)


def _number_density(get, units, fields):
    rho = get("Density").astype(np.float64) * units.density_to_cgs
    nh = rho * _xh(get, fields) / PROTONMASS_CGS
    return nh.astype(np.float32)


def _pressure(get, units, fields):
    """Thermal pressure as P/k_B in K cm^-3."""
    rho = get("Density").astype(np.float64) * units.density_to_cgs
    u_cgs = get("InternalEnergy").astype(np.float64) * units.u_to_cgs
    p = (GAMMA - 1.0) * rho * u_cgs / BOLTZMANN_CGS
    return p.astype(np.float32)


def _entropy(get, units, fields):
    """Pseudo-entropy K = T / n_H^(2/3) in K cm^2."""
    t = _temperature(get, units, fields).astype(np.float64)
    nh = _number_density(get, units, fields).astype(np.float64)
    k = t / np.maximum(nh, 1e-30) ** (2.0 / 3.0)
    return k.astype(np.float32)


def _sound_speed(get, units, fields):
    u_cgs = get("InternalEnergy").astype(np.float64) * units.u_to_cgs
    cs = np.sqrt(GAMMA * (GAMMA - 1.0) * u_cgs) / KM_CGS
    return cs.astype(np.float32)


def _velocity_magnitude(get, units, fields):
    v = get("@vec:Velocities").astype(np.float64) * units.velocity_to_kms
    return np.linalg.norm(v, axis=1).astype(np.float32)


def _radial_velocity(get, units, fields):
    """v . r_hat relative to the current view center, in km/s.

    No bulk-flow subtraction: for a halo at rest in the box this is the
    usual infall/outflow diagnostic; sign convention: positive = outflow.
    """
    v = get("@vec:Velocities").astype(np.float64) * units.velocity_to_kms
    pos = get("@vec:Coordinates").astype(np.float64)
    center = get("@center")
    d = pos - center[None, :]
    r = np.linalg.norm(d, axis=1)
    rhat = d / np.maximum(r, 1e-30)[:, None]
    return (v * rhat).sum(axis=1).astype(np.float32)


def _radius_from_center(get, units, fields):
    pos = get("@vec:Coordinates").astype(np.float64)
    center = get("@center")
    r = np.linalg.norm(pos - center[None, :], axis=1) * units.length_to_kpc
    return r.astype(np.float32)


def _bfield_magnitude(get, units, fields):
    b = get("@vec:MagneticField").astype(np.float64) * units.bfield_to_gauss
    return (np.linalg.norm(b, axis=1) * 1e6).astype(np.float32)  # microGauss


def _metallicity_zsun(get, units, fields):
    if "GFM_Metallicity" in fields:
        z = get("GFM_Metallicity")
    else:
        z = get("Metallicity[0]")
    return (np.asarray(z, dtype=np.float64) / ZSUN).astype(np.float32)


def _stellar_age(get, units, fields):
    aform = np.asarray(get("GFM_StellarFormationTime"), dtype=np.float64)
    age = units.lookback_gyr(np.clip(aform, 1e-8, None))
    # Negative formation time marks wind-phase cells in TNG: not stars.
    age[aform <= 0] = 0.0
    return age.astype(np.float32)


class DerivedField:
    def __init__(self, fn, requires, unit, description, any_of=None):
        self.fn = fn
        self.requires = set(requires)
        self.any_of = [set(g) for g in (any_of or [])]
        self.unit = unit
        self.description = description

    def available(self, raw_fields, raw_vectors):
        pool = set(raw_fields) | {f"@vec:{v}" for v in raw_vectors}
        pool.add("@vec:Coordinates")
        pool.add("@center")  # always derivable
        if not self.requires <= pool:
            return False
        for group in self.any_of:
            if not (group & pool):
                return False
        return True


DERIVED_FIELDS = {
    "Temperature": DerivedField(
        _temperature, {"InternalEnergy"}, "K",
        "Gas temperature from u, x_e (H/He plasma)"),
    "NumberDensity": DerivedField(
        _number_density, {"Density"}, "cm^-3",
        "Hydrogen number density n_H"),
    "Pressure": DerivedField(
        _pressure, {"Density", "InternalEnergy"}, "K cm^-3",
        "Thermal pressure P/k_B"),
    "Entropy": DerivedField(
        _entropy, {"Density", "InternalEnergy"}, "K cm^2",
        "Pseudo-entropy T / n_H^(2/3)"),
    "SoundSpeed": DerivedField(
        _sound_speed, {"InternalEnergy"}, "km/s",
        "Adiabatic sound speed"),
    "VelocityMagnitude": DerivedField(
        _velocity_magnitude, {"@vec:Velocities"}, "km/s",
        "Speed |v|"),
    "RadialVelocity": DerivedField(
        _radial_velocity, {"@vec:Velocities", "@center"}, "km/s",
        "v . r_hat about the view center (positive = outflow)"),
    "RadiusFromCenter": DerivedField(
        _radius_from_center, {"@center"}, "kpc",
        "Physical distance from the view center"),
    "MagneticFieldMagnitude": DerivedField(
        _bfield_magnitude, {"@vec:MagneticField"}, "uG",
        "Magnetic field strength |B|"),
    "MetallicityZsun": DerivedField(
        _metallicity_zsun, set(), "Zsun",
        "Metal mass fraction in solar units",
        any_of=[{"GFM_Metallicity", "Metallicity[0]"}]),
    "StellarAge": DerivedField(
        _stellar_age, {"GFM_StellarFormationTime"}, "Gyr",
        "Stellar age (flat LCDM lookback)"),
}

# Display units for raw fields after UnitSystem conversion (used by the
# inspector and colorbar labels; raw render values stay in code units).
RAW_FIELD_UNITS = {
    "Masses": ("Msun", "mass_to_msun"),
    "Density": ("g/cm^3", "density_to_cgs"),
    "StarFormationRate": ("Msun/yr", None),  # already physical in TNG
    "Velocities": ("km/s", "velocity_to_kms"),
    "Coordinates": ("kpc", "length_to_kpc"),
    "MagneticField": ("G", "bfield_to_gauss"),
}


def field_unit_label(name):
    """Axis/colorbar unit label for a field name ('' when unknown)."""
    df = DERIVED_FIELDS.get(name)
    if df is not None:
        return df.unit
    base = name.split("[")[0]
    if base in RAW_FIELD_UNITS:
        return "code units"
    return ""


def available_derived_fields(raw_fields, raw_vectors):
    """Names of derived fields computable from the given raw field set."""
    return [
        name
        for name, df in DERIVED_FIELDS.items()
        if df.available(raw_fields, raw_vectors)
    ]


def compute_derived_field(name, data):
    """Compute a derived field over all selected particle types.

    `data` is a SnapshotData; raw fields are fetched through its
    concatenated accessors so the result aligns with data.positions.
    """
    df = DERIVED_FIELDS[name]
    units = UnitSystem(data.header)
    fields = set(data.available_fields())

    def get(key):
        if key == "@center":
            return np.asarray(data.get_view_center(), dtype=np.float64)
        if key.startswith("@vec:"):
            return data.get_vector_field(key[5:])
        return data.get_field(key)

    return df.fn(get, units, fields)
