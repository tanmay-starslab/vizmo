// Volume ray marching (Section 5.E): fullscreen pass; each fragment
// casts a ray from the camera through the voxel cube and integrates
// the emission-absorption transport equation
//     dI/ds = j - k I
// front-to-back:  C += T * tf_color * a;  T *= (1 - a)
// with a 1D RGBA transfer function (alpha = opacity per unit step,
// rgb = emission color). mode = 1 switches to maximum-intensity
// projection. Field values are normalized to [0, 1] with (vmin, vmax)
// (log-scaled on the CPU side via the params).

struct VolParams {
    cam_pos: vec3<f32>,
    fov_y: f32,
    cam_fwd: vec3<f32>,
    aspect: f32,
    cam_right: vec3<f32>,
    step_mult: f32,
    cam_up: vec3<f32>,
    vmin: f32,
    box_center: vec3<f32>,
    vmax: f32,
    half_size: f32,
    n_grid: f32,
    mode: u32,        // 0 = emission-absorption, 1 = MIP
    max_steps: u32,
};

@group(0) @binding(0) var<uniform> p: VolParams;
@group(0) @binding(1) var t_vol: texture_3d<f32>;
@group(0) @binding(2) var s_vol: sampler;
@group(0) @binding(3) var t_tf: texture_1d<f32>;
@group(0) @binding(4) var s_tf: sampler;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    // Fullscreen triangle
    var out: VSOut;
    let x = f32((vi << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vi & 2u) * 2.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.ndc = vec2<f32>(x, y);
    return out;
}

fn box_intersect(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    let lo = p.box_center - vec3<f32>(p.half_size);
    let hi = p.box_center + vec3<f32>(p.half_size);
    let inv = 1.0 / rd;
    let t0 = (lo - ro) * inv;
    let t1 = (hi - ro) * inv;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let near = max(max(tmin.x, tmin.y), tmin.z);
    let far = min(min(tmax.x, tmax.y), tmax.z);
    return vec2<f32>(max(near, 0.0), far);
}

@fragment
fn fs_main(@location(0) ndc: vec2<f32>) -> @location(0) vec4<f32> {
    let tan_half = tan(p.fov_y * 0.5);
    let rd = normalize(p.cam_fwd
                       + ndc.x * tan_half * p.aspect * p.cam_right
                       + ndc.y * tan_half * p.cam_up);
    let hit = box_intersect(p.cam_pos, rd);
    if (hit.y <= hit.x) {
        return vec4<f32>(0.0);
    }
    let step = (2.0 * p.half_size / p.n_grid) * 0.5 * p.step_mult;
    var t = hit.x + step * 0.5;
    var color = vec3<f32>(0.0);
    var transmit = 1.0;
    var mip = 0.0;
    let inv_span = 1.0 / max(p.vmax - p.vmin, 1.0e-30);
    for (var s: u32 = 0u; s < p.max_steps; s = s + 1u) {
        if (t >= hit.y || transmit < 0.01) { break; }
        let wp = p.cam_pos + t * rd;
        let uvw = (wp - (p.box_center - vec3<f32>(p.half_size)))
                  / (2.0 * p.half_size);
        let v = textureSampleLevel(t_vol, s_vol, uvw, 0.0).r;
        let x = clamp((v - p.vmin) * inv_span, 0.0, 1.0);
        if (p.mode == 1u) {
            mip = max(mip, x);
        } else {
            let tf = textureSampleLevel(t_tf, s_tf, x, 0.0);
            let a = clamp(tf.a * p.step_mult, 0.0, 1.0);
            color += transmit * tf.rgb * a;
            transmit *= (1.0 - a);
        }
        t += step;
    }
    if (p.mode == 1u) {
        let tf = textureSampleLevel(t_tf, s_tf, mip, 0.0);
        return vec4<f32>(tf.rgb, mip);
    }
    return vec4<f32>(color, 1.0 - transmit);
}
