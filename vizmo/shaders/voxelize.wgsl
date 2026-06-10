// CIC voxelization compute shader (Section 5.E): splats particle
// masses into an n^3 grid held as a u32 atomic storage buffer in
// fixed-point (value * FIXED_SCALE), because WGSL has no float
// atomics. A second tiny pass (or the CPU) converts to float.
// Out-of-box particles are dropped.

struct VoxParams {
    center: vec3<f32>,
    half_size: f32,
    n_grid: u32,
    n_particles: u32,
    fixed_scale: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> params: VoxParams;
@group(0) @binding(1) var<storage, read> s_pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> s_mass: array<f32>;
@group(0) @binding(3) var<storage, read_write> out_grid: array<atomic<u32>>;

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_particles) { return; }
    let n = params.n_grid;
    let nf = f32(n);
    let size = 2.0 * params.half_size;
    let p = (s_pos[i].xyz - (params.center - vec3<f32>(params.half_size)))
            / size * nf;
    if (any(p < vec3<f32>(0.0)) || any(p >= vec3<f32>(nf))) { return; }
    let pc = p - vec3<f32>(0.5);
    let i0 = vec3<i32>(floor(pc));
    let f = pc - floor(pc);
    let m = s_mass[i] * params.fixed_scale;
    for (var dx = 0; dx < 2; dx++) {
        let wx = select(1.0 - f.x, f.x, dx == 1);
        let ix = clamp(i0.x + dx, 0, i32(n) - 1);
        for (var dy = 0; dy < 2; dy++) {
            let wy = select(1.0 - f.y, f.y, dy == 1);
            let iy = clamp(i0.y + dy, 0, i32(n) - 1);
            for (var dz = 0; dz < 2; dz++) {
                let wz = select(1.0 - f.z, f.z, dz == 1);
                let iz = clamp(i0.z + dz, 0, i32(n) - 1);
                let idx = u32(ix) * n * n + u32(iy) * n + u32(iz);
                atomicAdd(&out_grid[idx], u32(m * wx * wy * wz));
            }
        }
    }
}
