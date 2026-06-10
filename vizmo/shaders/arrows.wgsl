// Instanced 3D arrow glyphs (Section 5.D): one instance per particle,
// procedural shaft + cone tip generated in the vertex shader from
// lookup tables (30 vertices per instance, triangle list). Instance
// attributes: position (relative to the CPU-folded reference), the
// arrow vector (direction * length in world units), and an RGBA color
// (field value through the colormap, computed CPU-side).

struct ArrowUniform {
    mvp: mat4x4<f32>,
    radius: f32,    // shaft radius in world units
    _p0: f32,
    _p1: f32,
    _p2: f32,
};

@group(0) @binding(0) var<uniform> u: ArrowUniform;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) shade: f32,
};

// Unit-arrow template along +Z: shaft = 3 rectangles (6 tris) from
// z=0 to z=0.7, tip cone = 4 tris from z=0.7 to z=1. Cross-sections
// use a triangle (3 spokes at 120 deg).
const N_VERTS: u32 = 30u;

fn spoke(i: u32) -> vec2<f32> {
    let a = f32(i) * 2.0943951; // 120 degrees
    return vec2<f32>(cos(a), sin(a));
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32,
           @location(0) ipos: vec3<f32>,
           @location(1) ivec: vec3<f32>,
           @location(2) icolor: vec4<f32>) -> VSOut {
    let len = length(ivec);
    var out: VSOut;
    if (len < 1.0e-20) {
        out.position = vec4<f32>(0.0, 0.0, -2.0, 1.0); // degenerate
        out.color = vec4<f32>(0.0);
        out.shade = 0.0;
        return out;
    }
    let dir = ivec / len;
    // Orthonormal frame around dir.
    var up_ref = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(dir.z) > 0.9) { up_ref = vec3<f32>(0.0, 1.0, 0.0); }
    let e1 = normalize(cross(up_ref, dir));
    let e2 = cross(dir, e1);

    // Template vertex in (radial spoke s, radial scale r, axial z).
    // Shaft tris (18 verts): quad k between spokes k and k+1.
    var s0: u32; var r0: f32; var z0: f32;
    if (vi < 18u) {
        let quad = vi / 6u;          // 0..2
        let corner = vi % 6u;        // two tris of the quad
        var sk = quad;
        var zk = 0.0;
        // corners: (s,0) (s+1,0) (s,0.7) / (s+1,0) (s+1,0.7) (s,0.7)
        if (corner == 1u || corner == 3u || corner == 4u) { sk = quad + 1u; }
        if (corner >= 2u && corner != 3u) { zk = 0.7; }
        if (corner == 4u) { zk = 0.7; }
        s0 = sk % 3u; r0 = 1.0; z0 = zk;
    } else {
        // Cone tip (12 verts): 4 tris — 3 side faces + 1 base fan
        let tri = (vi - 18u) / 3u;   // 0..3
        let c = (vi - 18u) % 3u;
        if (tri < 3u) {
            if (c == 0u) { s0 = tri % 3u; r0 = 2.2; z0 = 0.7; }
            else if (c == 1u) { s0 = (tri + 1u) % 3u; r0 = 2.2; z0 = 0.7; }
            else { s0 = 0u; r0 = 0.0; z0 = 1.0; } // apex
        } else {
            // base ring fan to close the cone bottom
            if (c == 0u) { s0 = 0u; r0 = 2.2; z0 = 0.7; }
            else if (c == 1u) { s0 = 1u; r0 = 2.2; z0 = 0.7; }
            else { s0 = 2u; r0 = 2.2; z0 = 0.7; }
        }
    }
    let sp = spoke(s0) * u.radius * r0;
    let world = ipos + dir * (z0 * len) + e1 * sp.x + e2 * sp.y;
    out.position = u.mvp * vec4<f32>(world, 1.0);
    out.color = icolor;
    // Cheap shading: dim the shaft, brighten the tip.
    out.shade = 0.6 + 0.4 * z0;
    return out;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>,
           @location(1) shade: f32) -> @location(0) vec4<f32> {
    return vec4<f32>(color.rgb * shade, color.a);
}
