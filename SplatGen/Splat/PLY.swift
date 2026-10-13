import Foundation
import simd

/// Reads and writes the standard 3D Gaussian Splatting `.ply` layout
/// (x y z, f_dc_*, opacity logit, log scales, wxyz rotation), including the
/// camera elements that Apple's SHARP tooling appends after the vertices.
enum PLY {
	enum Failure: LocalizedError {
		case malformed(String)

		var errorDescription: String? {
			switch self {
			case .malformed(let why): "This isn't a Gaussian splat .ply file (\(why))."
			}
		}
	}

	private static let shC0: Float = 0.28209479177387814
	/// Files this app writes never exceed this size.
	static let maxFileBytes = 50_000_000
	private static let fieldsPerVertex = 14
	/// The most splats a file can hold within `maxFileBytes`, leaving room for the header and camera.
	static let maxSplats = (maxFileBytes - 4096) / (fieldsPerVertex * MemoryLayout<Float>.stride)
	/// Header line marking files whose camera is just a starting viewpoint, not the frame of a source photo.
	private static let viewpointComment = "comment SplatGen camera is a viewpoint, not a source photo"

	// MARK: Reading

	private struct Element {
		var name: String
		var count: Int
		var properties: [(name: String, offset: Int, type: String)] = []
		var stride = 0
		var dataOffset = 0
	}

	private static func size(of type: String) -> Int? {
		switch type {
		case "char", "uchar", "int8", "uint8": 1
		case "short", "ushort", "int16", "uint16": 2
		case "int", "uint", "float", "int32", "uint32", "float32": 4
		case "double", "float64": 8
		default: nil
		}
	}

	static func read(_ url: URL) throws -> GaussianCloud {
		let data = try Data(contentsOf: url, options: .alwaysMapped)
		let marker = Data("end_header\n".utf8)
		guard let end = data.range(of: marker, in: 0..<min(data.count, 1 << 16)),
			  let header = String(data: data[0..<end.lowerBound], encoding: .ascii) else {
			throw Failure.malformed("no header")
		}
		guard header.contains("format binary_little_endian") else {
			throw Failure.malformed("only binary little-endian files are supported")
		}

		var elements: [Element] = []
		for line in header.split(whereSeparator: \.isNewline) {
			let parts = line.split(separator: " ").map(String.init)
			if parts.first == "element", parts.count == 3, let count = Int(parts[2]) {
				elements.append(Element(name: parts[1], count: count))
			} else if parts.first == "property", parts.count == 3, !elements.isEmpty {
				guard let size = size(of: parts[1]) else { throw Failure.malformed("unsupported property \(parts[1])") }
				let i = elements.count - 1
				elements[i].properties.append((parts[2], elements[i].stride, parts[1]))
				elements[i].stride += size
			} else if parts.first == "property" {
				throw Failure.malformed("list properties are not supported")
			}
		}
		var offset = end.upperBound
		for i in elements.indices {
			elements[i].dataOffset = offset
			offset += elements[i].count * elements[i].stride
		}
		guard offset <= data.count, let vertex = elements.first(where: { $0.name == "vertex" }) else {
			throw Failure.malformed("missing vertex data")
		}

		func floatOffset(_ name: String) throws -> Int {
			guard let p = vertex.properties.first(where: { $0.name == name }), p.type == "float" || p.type == "float32" else {
				throw Failure.malformed("missing \(name)")
			}
			return p.offset
		}
		let names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
					 "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
		let o = try names.map(floatOffset)

		var splats = ContiguousArray<GPUSplat>()
		splats.reserveCapacity(vertex.count)
		data.withUnsafeBytes { raw in
			for i in 0..<vertex.count {
				let base = vertex.dataOffset + i * vertex.stride
				func f(_ k: Int) -> Float { raw.loadUnaligned(fromByteOffset: base + o[k], as: Float.self) }

				let opacity = 1 / (1 + exp(-f(6)))
				guard opacity > 1 / 255 else { continue }
				let srgb = SIMD3(f(3), f(4), f(5)) * shC0 + 0.5
				let color = SIMD3(
					ColorSpace.srgbToLinear(max(0, min(1, srgb.x))),
					ColorSpace.srgbToLinear(max(0, min(1, srgb.y))),
					ColorSpace.srgbToLinear(max(0, min(1, srgb.z))))
				let scale = SIMD3(exp(f(7)), exp(f(8)), exp(f(9)))
				let q = simd_normalize(simd_quatf(ix: f(11), iy: f(12), iz: f(13), r: f(10)))
				let m = simd_float3x3(q) * simd_float3x3(diagonal: scale)
				splats.append(GPUSplat(position: SIMD3(f(0), f(1), f(2)), color: color, opacity: opacity, covariance: m * m.transpose))
			}
		}
		guard !splats.isEmpty else { throw Failure.malformed("no visible splats") }

		var camera = SourceCamera.fallback
		if let intrinsic = elements.first(where: { $0.name == "intrinsic" && $0.count == 9 && $0.stride == 4 }),
		   let size = elements.first(where: { $0.name == "image_size" && $0.count == 2 && $0.stride == 4 }) {
			data.withUnsafeBytes { raw in
				let focal = raw.loadUnaligned(fromByteOffset: intrinsic.dataOffset, as: Float.self)
				let w = raw.loadUnaligned(fromByteOffset: size.dataOffset, as: UInt32.self)
				let h = raw.loadUnaligned(fromByteOffset: size.dataOffset + 4, as: UInt32.self)
				if focal > 0, w > 0, h > 0 {
					camera = SourceCamera(
						imageSize: SIMD2(Float(w), Float(h)), focalPx: focal, isFromPhoto: !header.contains(viewpointComment))
				}
			}
		}
		let cloud = GaussianCloud(splats: splats, camera: camera)
		// A file written facing the other way says so in its extrinsic (world to camera, row-major).
		if let extrinsic = elements.first(where: { $0.name == "extrinsic" && $0.count == 16 && $0.stride == 4 }) {
			let m = data.withUnsafeBytes { raw in
				(0..<12).map { raw.loadUnaligned(fromByteOffset: extrinsic.dataOffset + $0 * 4, as: Float.self) }
			}
			let rotation = simd_float3x3(rows: [SIMD3(m[0], m[1], m[2]), SIMD3(m[4], m[5], m[6]), SIMD3(m[8], m[9], m[10])])
			let translation = SIMD3(m[3], m[7], m[11])
			let skew = rotation * rotation.transpose - matrix_identity_float3x3
			let isRigid = max(simd_length(skew[0]), simd_length(skew[1]), simd_length(skew[2])) < 1e-3
				&& simd_determinant(rotation) > 0
			if isRigid, rotation != matrix_identity_float3x3 || translation != .zero {
				return cloud.transformed(rotation: rotation, translation: translation, camera: camera)
			}
		}
		return cloud
	}

	// MARK: Writing

	/// Writes `cloud`, first dropping its least visible splats if it holds more than `maxSplats`.
	/// `facingBackward` turns the scene half a turn about its vertical axis, so it lies down -z instead of +z:
	/// viewers that look down -z (visionOS) then show it in front of the viewer rather than behind.
	static func write(_ cloud: GaussianCloud, to url: URL, facingBackward: Bool = false) throws {
		var cloud = cloud.pruned(to: maxSplats)
		let turn = simd_float3x3(diagonal: SIMD3(-1, 1, -1))
		if facingBackward { cloud = cloud.transformed(rotation: turn, translation: .zero, camera: cloud.camera) }
		let n = cloud.count
		let w = cloud.camera.imageSize.x, h = cloud.camera.imageSize.y, f = cloud.camera.focalPx
		var header = "ply\nformat binary_little_endian 1.0\n"
		if !cloud.camera.isFromPhoto { header += viewpointComment + "\n" }
		header += "element vertex \(n)\n"
		for name in ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
					 "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"] {
			header += "property float \(name)\n"
		}
		header += "element extrinsic 16\nproperty float extrinsic\n"
		header += "element intrinsic 9\nproperty float intrinsic\n"
		header += "element image_size 2\nproperty uint image_size\n"
		header += "end_header\n"

		var vertices = [Float](repeating: 0, count: n * fieldsPerVertex)
		vertices.withUnsafeMutableBufferPointer { out in
			let out = UnsafeMutablePointer<Float>(out.baseAddress!)
			cloud.splats.withUnsafeBufferPointer { splats in
				let chunk = 16_384
				DispatchQueue.concurrentPerform(iterations: (n + chunk - 1) / chunk) { c in
					for i in (c * chunk)..<min(n, (c + 1) * chunk) {
						let s = splats[i]
						let (scale, q) = decompose(s.covariance)
						let v = out + i * fieldsPerVertex
						v[0] = s.position.x; v[1] = s.position.y; v[2] = s.position.z
						for k in 0..<3 {
							v[3 + k] = (ColorSpace.linearToSRGB(max(0, min(1, s.color[k]))) - 0.5) / shC0
							v[7 + k] = log(scale[k])
						}
						let a = max(1e-6, min(1 - 1e-6, s.color.w))
						v[6] = log(a / (1 - a))
						v[10] = q.real; v[11] = q.imag.x; v[12] = q.imag.y; v[13] = q.imag.z
					}
				}
			}
		}

		var data = Data(header.utf8)
		vertices.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
		// World to camera: the turn is its own inverse.
		let extrinsic: [Float] = facingBackward
			? [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1]
			: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
		let intrinsic: [Float] = [f, 0, w / 2, 0, f, h / 2, 0, 0, 1]
		let imageSize: [UInt32] = [UInt32(w), UInt32(h)]
		extrinsic.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
		intrinsic.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
		imageSize.withUnsafeBufferPointer { data.append(Data(buffer: $0)) }
		try data.write(to: url, options: .atomic)
	}

	/// Splits a covariance back into per-axis scales and a rotation (Jacobi eigen-decomposition).
	private static func decompose(_ covariance: simd_float3x3) -> (SIMD3<Float>, simd_quatf) {
		var a = simd_double3x3(
			SIMD3<Double>(covariance[0]), SIMD3<Double>(covariance[1]), SIMD3<Double>(covariance[2]))
		var v = matrix_identity_double3x3
		for _ in 0..<8 {
			let off = abs(a[1][0]) + abs(a[2][0]) + abs(a[2][1])
			if off < 1e-18 * (abs(a[0][0]) + abs(a[1][1]) + abs(a[2][2])) { break }
			for (p, q) in [(0, 1), (0, 2), (1, 2)] where abs(a[q][p]) > 0 {
				let theta = (a[q][q] - a[p][p]) / (2 * a[q][p])
				let t = (theta >= 0 ? 1 : -1) / (abs(theta) + (theta * theta + 1).squareRoot())
				let c = 1 / (t * t + 1).squareRoot(), s = t * c
				var j = matrix_identity_double3x3
				j[p][p] = c; j[q][q] = c; j[q][p] = s; j[p][q] = -s
				a = j.transpose * a * j
				v = v * j
			}
		}
		if simd_determinant(v) < 0 { v[2] = -v[2] }
		let scale = SIMD3(Float(max(a[0][0], 1e-14).squareRoot()),
						  Float(max(a[1][1], 1e-14).squareRoot()),
						  Float(max(a[2][2], 1e-14).squareRoot()))
		let rotation = simd_float3x3(SIMD3<Float>(v[0]), SIMD3<Float>(v[1]), SIMD3<Float>(v[2]))
		return (scale, simd_normalize(simd_quatf(rotation)))
	}
}
