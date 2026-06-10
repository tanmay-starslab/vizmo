// Slice-plane reconstruction (meshoid Slice(), 0th order, on the GPU).
//
// One invocation per output pixel. Each pixel sits at a world-space
// point on the slice plane; the shader gathers every slab-culled
// particle (the CPU pre-filters to |dist_to_plane| < hsml and caps the
// count), evaluates the M4 cubic-spline kernel at the 3D distance, and
// accumulates kernel-weighted numerator/denominator:
//
//   value(p) = sum_i f_i W(|p - x_i|/h_i)/h_i^3 / sum_i W(...)/h_i^3
//
// which is meshoid's kernel-weighted (0th-order) slice reconstruction.
// Empty pixels (no kernel support) output NaN-coded -1e30 so the CPU
// can mask them.
//
// Particle positions arrive already transformed into plane coordinates
// (e1, e2 in-plane, e3 = normal), relative to the plane center, so the
// pixel grid is axis-aligned in shader space.

struct SliceParams {
    half_size: f32,   // half extent of the grid in world units
    res: u32,         // grid resolution (res x res)
    n_particles: u32, // slab-culled particle count
    _pad: u32,
};

@group(0) @binding(0) var<uniform> params: SliceParams;
// xyz = plane-frame position (e1, e2, normal-dist), w = hsml
@group(0) @binding(1) var<storage, read> s_pos_h: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> s_val: array<f32>;
// Output: res*res floats, row-major (y * res + x)
@group(0) @binding(3) var<storage, read_write> out_grid: array<f32>;

fn kernel_m4(u: f32) -> f32 {
    // M4 cubic spline, normalized to 8/pi * ... (constant factors
    // cancel in the ratio, kept for numerical sanity).
    if (u >= 1.0) {
        return 0.0;
    }
    if (u < 0.5) {
        return (8.0 / 3.14159265) * (1.0 - 6.0 * u * u + 6.0 * u * u * u);
    }
    let t = 1.0 - u;
    return (8.0 / 3.14159265) * 2.0 * t * t * t;
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.res;
    if (gid.x >= res || gid.y >= res) {
        return;
    }
    // Pixel center in plane coordinates.
    let cell = 2.0 * params.half_size / f32(res);
    let px = -params.half_size + (f32(gid.x) + 0.5) * cell;
    let py = -params.half_size + (f32(gid.y) + 0.5) * cell;

    var num: f32 = 0.0;
    var den: f32 = 0.0;
    for (var i: u32 = 0u; i < params.n_particles; i = i + 1u) {
        let ph = s_pos_h[i];
        let h = ph.w;
        let dx = ph.x - px;
        let dy = ph.y - py;
        let dz = ph.z; // distance to the plane (precomputed component)
        let r = sqrt(dx * dx + dy * dy + dz * dz);
        let w = kernel_m4(r / h) / (h * h * h);
        num = num + s_val[i] * w;
        den = den + w;
    }
    let idx = gid.y * res + gid.x;
    if (den > 0.0) {
        out_grid[idx] = num / den;
    } else {
        out_grid[idx] = -1.0e30; // empty-pixel sentinel
    }
}
