// Isosurface mesh rendering with Phong shading (Section 5.B).
// Directional light from the camera direction + ambient term; the
// surface uses a single solid color with opacity (the threshold is
// already encoded by the surface itself).

struct IsoUniform {
    mvp: mat4x4<f32>,
    color: vec4<f32>,     // rgb + opacity
    light_dir: vec4<f32>, // xyz = direction TOWARD the light (unit)
};

@group(0) @binding(0) var<uniform> u: IsoUniform;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) normal: vec3<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>,
           @location(1) nrm: vec3<f32>) -> VSOut {
    var out: VSOut;
    out.position = u.mvp * vec4<f32>(pos, 1.0);
    out.normal = nrm;
    return out;
}

@fragment
fn fs_main(@location(0) nrm: vec3<f32>) -> @location(0) vec4<f32> {
    let n = normalize(nrm);
    let l = normalize(u.light_dir.xyz);
    // Two-sided diffuse (mesh normals from marching cubes can point
    // either way) + ambient + a small Blinn specular.
    let diff = abs(dot(n, l));
    let spec = pow(max(abs(dot(n, l)), 0.0), 24.0) * 0.25;
    let shade = 0.25 + 0.65 * diff + spec;
    return vec4<f32>(u.color.rgb * shade, u.color.a);
}
