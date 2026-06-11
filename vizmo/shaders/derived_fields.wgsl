// GPU-side derived fields (Item 2): Temperature [K] and hydrogen
// number density n_H [cm^-3] from raw snapshot fields, matching
// physics.py's CPU formulas exactly (X_H = 0.76, gamma = 5/3,
// fully-general electron abundance).

struct DerivedParams {
    n_particles: u32,
    row_stride: u32,  // gid.x range = workgroups_x * 256 (2D dispatch)
    unit_velocity2: f32,  // (UnitVelocity_in_cm_per_s)^2 incl. sqrt(a) factors
    unit_density: f32,    // code density -> g/cm^3 (incl. h^2/a^3)
};

const X_H: f32 = 0.76;
const GAMMA_M1: f32 = 0.6666666667;       // gamma - 1
const M_H_CGS: f32 = 1.67262192e-24;
const K_B_CGS: f32 = 1.380649e-16;

@group(0) @binding(0) var<uniform> params: DerivedParams;
@group(0) @binding(1) var<storage, read> u_int: array<f32>;
@group(0) @binding(2) var<storage, read> xe: array<f32>;
@group(0) @binding(3) var<storage, read> rho: array<f32>;
@group(0) @binding(4) var<storage, read_write> temperature: array<f32>;
@group(0) @binding(5) var<storage, read_write> number_density: array<f32>;

@compute @workgroup_size(256)
fn main_derived_fields(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * params.row_stride + gid.x;
    if (i >= params.n_particles) { return; }
    let mu = 4.0 / (1.0 + 3.0 * X_H + 4.0 * X_H * xe[i]);
    let u_cgs = u_int[i] * params.unit_velocity2;
    temperature[i] = GAMMA_M1 * u_cgs * mu * M_H_CGS / K_B_CGS;
    number_density[i] = X_H * rho[i] * params.unit_density / M_H_CGS;
}
