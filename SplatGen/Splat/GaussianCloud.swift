import Foundation
import simd

/// One Gaussian as the GPU sees it. Layout must match `Splat` in Shaders.swift.
struct GPUSplat {
	/// xyz in scene space (OpenCV convention: x right, y down, z forward).
	var position: SIMD4<Float>
	/// Linear RGB + opacity.
	var color: SIMD4<Float>
	/// Upper triangle of the 3D covariance: (xx, xy, xz, yy) and (yz, zz, -, -).
	var covA: SIMD4<Float>
	var covB: SIMD4<Float>

	init(position: SIMD3<Float>, color: SIMD3<Float>, opacity: Float, covariance c: simd_float3x3) {
		self.position = SIMD4(position, 1)
		self.color = SIMD4(color, opacity)
		covA = SIMD4(c[0][0], c[1][0], c[2][0], c[1][1])
		covB = SIMD4(c[2][1], c[2][2], 0, 0)
	}

	var covariance: simd_float3x3 {
		simd_float3x3(
			SIMD3(covA.x, covA.y, covA.z),
			SIMD3(covA.y, covA.w, covB.x),
			SIMD3(covA.z, covB.x, covB.y))
	}
}

/// The camera the scene was captured (or predicted) from, sitting at the origin looking down +z.
struct SourceCamera {
	var imageSize: SIMD2<Float>
	var focalPx: Float
	/// False when the camera is a guess rather than a real photo frame.
	var isFromPhoto = true

	/// Used for splat files that carry no camera: a 50° vertical field of view.
	static let fallback = SourceCamera(
		imageSize: SIMD2(1600, 1200), focalPx: 600 / tan(25 * .pi / 180), isFromPhoto: false)
}

/// An immutable set of Gaussians ready for rendering.
final class GaussianCloud: @unchecked Sendable {
	let splats: ContiguousArray<GPUSplat>
	/// Positions duplicated into a tight array so depth sorting stays cache friendly.
	let positions: ContiguousArray<SIMD4<Float>>
	let camera: SourceCamera
	/// Depth (z) quantiles at 0%, 1% ... 100%, opacity-filtered.
	let depthQuantiles: [Float]

	var count: Int { splats.count }

	init(splats: ContiguousArray<GPUSplat>, camera: SourceCamera) {
		self.splats = splats
		self.camera = camera
		positions = ContiguousArray(splats.lazy.map(\.position))

		var sample: [Float] = []
		let step = max(1, splats.count / 20_000)
		for i in stride(from: 0, to: splats.count, by: step) where splats[i].color.w > 0.2 {
			sample.append(splats[i].position.z)
		}
		if sample.isEmpty { sample = [1] }
		sample.sort()
		depthQuantiles = (0...100).map { sample[min(sample.count - 1, $0 * sample.count / 100)] }
	}

	func depth(atQuantile q: Float) -> Float {
		let x = max(0, min(1, q)) * 100
		let i = Int(x)
		guard i < 100 else { return depthQuantiles[100] }
		return depthQuantiles[i] + (depthQuantiles[i + 1] - depthQuantiles[i]) * (x - Float(i))
	}
}

enum ColorSpace {
	static func srgbToLinear(_ c: Float) -> Float {
		c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4)
	}

	static func linearToSRGB(_ c: Float) -> Float {
		c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1 / 2.4) - 0.055
	}
}
