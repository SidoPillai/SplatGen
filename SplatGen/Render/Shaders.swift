/// Metal source for the splat pipeline. Compiled at launch so the project builds without
/// the separately installed Metal Toolchain.
enum Shaders {
	static let source = #"""
#include <metal_stdlib>
using namespace metal;

// Must match GPUSplat in GaussianCloud.swift.
struct Splat {
    float4 position;
    float4 color;
    float4 covA; // xx, xy, xz, yy
    float4 covB; // yz, zz
};

// Must match SplatUniforms in SplatRenderer.swift.
struct Uniforms {
    float4x4 view;         // scene -> camera, OpenCV convention (y down, z forward)
    float4 focalViewport;  // fx, fy, viewport width, viewport height (pixels)
    float4 reveal;         // progress 0...1, near depth, far depth, unused
};

struct VertexOut {
    float4 position [[position]];
    float4 color;
    float2 uv;
};

vertex VertexOut splatVertex(uint vertexID [[vertex_id]],
                             uint instanceID [[instance_id]],
                             const device Splat *splats [[buffer(0)]],
                             const device uint *order [[buffer(1)]],
                             constant Uniforms &u [[buffer(2)]]) {
    VertexOut out;
    out.position = float4(0, 0, 2, 1); // outside the clip volume unless we finish below
    out.color = float4(0);
    out.uv = float2(0);

    Splat s = splats[order[instanceID]];
    float3 cam = (u.view * float4(s.position.xyz, 1)).xyz;
    if (cam.z < 0.02) { return out; }

    float2 focal = u.focalViewport.xy;
    float2 viewport = u.focalViewport.zw;

    // Grow splats in from front to back while the scene is being revealed.
    float depthT = saturate((cam.z - u.reveal.y) / max(u.reveal.z - u.reveal.y, 1e-4));
    float grow = smoothstep(0.0, 1.0, saturate(u.reveal.x * 2.0 - depthT));
    if (grow <= 0.0) { return out; }

    float3x3 sigma = float3x3(float3(s.covA.x, s.covA.y, s.covA.z),
                              float3(s.covA.y, s.covA.w, s.covB.x),
                              float3(s.covA.z, s.covB.x, s.covB.y));
    float3x3 rotation = float3x3(u.view[0].xyz, u.view[1].xyz, u.view[2].xyz);

    // Keep the linearised projection stable for splats far outside the frustum.
    float2 limit = 1.3 * 0.5 * viewport / focal;
    float2 t = clamp(cam.xy / cam.z, -limit, limit) * cam.z;
    float3x3 jacobian = float3x3(float3(focal.x / cam.z, 0, 0),
                                 float3(0, focal.y / cam.z, 0),
                                 float3(-focal.x * t.x / (cam.z * cam.z), -focal.y * t.y / (cam.z * cam.z), 0));
    float3x3 m = jacobian * rotation;
    float3x3 cov = m * sigma * transpose(m);

    // 2D covariance in pixels, with a small low-pass so thin splats don't alias.
    float a = cov[0][0] + 0.3;
    float b = cov[0][1];
    float d = cov[1][1] + 0.3;
    float mid = 0.5 * (a + d);
    float radius = length(float2(0.5 * (a - d), b));
    float lambda1 = mid + radius;
    float lambda2 = max(mid - radius, 0.1);

    float2 axis = float2(b, lambda1 - a);
    axis = dot(axis, axis) > 1e-12 ? normalize(axis) : (a >= d ? float2(1, 0) : float2(0, 1));
    float2 major = min(sqrt(2.0 * lambda1), 1024.0) * axis;
    float2 minor = min(sqrt(2.0 * lambda2), 1024.0) * float2(axis.y, -axis.x);

    float2 corner = float2(vertexID & 1 ? 2.0 : -2.0, vertexID & 2 ? 2.0 : -2.0);
    float2 pixel = focal * cam.xy / cam.z + (corner.x * major + corner.y * minor) * grow;

    out.position = float4(pixel.x * 2.0 / viewport.x, -pixel.y * 2.0 / viewport.y, 0.5, 1);
    out.color = float4(s.color.rgb, s.color.a * grow);
    out.uv = corner;
    return out;
}

fragment float4 splatFragment(VertexOut in [[stage_in]]) {
    float r2 = dot(in.uv, in.uv);
    if (r2 > 4.0) { discard_fragment(); }
    float alpha = in.color.a * exp(-r2);
    return float4(in.color.rgb * alpha, alpha);
}
"""#
}
