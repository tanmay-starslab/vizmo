// Streamline polylines (Section 5.C): line_strip with per-vertex
// color computed on the CPU from the color-by field through the
// active colormap. MVP folds the float64 reference translation on the
// CPU (same precision strategy as the isosurface mesh path).

struct LineUniform {
    mvp: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> u: LineUniform;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>,
           @location(1) color: vec4<f32>) -> VSOut {
    var out: VSOut;
    out.position = u.mvp * vec4<f32>(pos, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
    return color;
}
