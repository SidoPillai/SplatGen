import CoreGraphics
import CoreML
import Foundation
import ImageIO
import simd

/// Turns one photo into Gaussians with Apple's SHARP network, running through Core ML on this Mac.
actor SharpGenerator {
	enum Failure: LocalizedError {
		case unreadableImage
		case modelMissing
		case unexpectedOutput(String)

		var errorDescription: String? {
			switch self {
			case .unreadableImage: "That file couldn't be read as an image."
			case .modelMissing: "The SHARP model isn't installed yet."
			case .unexpectedOutput(let detail): "The model returned something unexpected (\(detail))."
			}
		}
	}

	/// SHARP works on a fixed square input; the photo is stretched to it and un-stretched afterwards.
	private static let inputSide = 1536
	private var model: MLModel?

	// MARK: Model

	private func loadModel(status: @Sendable (String) -> Void) async throws -> MLModel {
		if let model { return model }
		let fm = FileManager.default
		if !fm.fileExists(atPath: ModelStore.compiledURL.path) {
			guard fm.fileExists(atPath: ModelStore.packageURL.path) else { throw Failure.modelMissing }
			status("Preparing the model for this Mac (first run only)…")
			let compiled = try await MLModel.compileModel(at: ModelStore.packageURL)
			try? fm.removeItem(at: ModelStore.compiledURL)
			try fm.moveItem(at: compiled, to: ModelStore.compiledURL)
		}
		status("Loading model…")
		let configuration = MLModelConfiguration()
		configuration.computeUnits = .cpuAndGPU
		let loaded = try MLModel(contentsOf: ModelStore.compiledURL, configuration: configuration)
		model = loaded
		return loaded
	}

	// MARK: Generation

	func generate(from imageURL: URL, status: @Sendable (String) -> Void) async throws -> GaussianCloud {
		let photo = try Self.loadPhoto(imageURL)
		let model = try await loadModel(status: status)

		status("Imagining depth and detail…")
		let disparityFactor = try MLMultiArray(shape: [1], dataType: .float32)
		disparityFactor[0] = NSNumber(value: photo.focalPx / Float(photo.width))
		let input = try MLDictionaryFeatureProvider(dictionary: [
			"image": try Self.imageTensor(photo.image),
			"disparity_factor": disparityFactor,
		])
		let output = try await model.prediction(from: input)

		status("Building splats…")
		func tensor(_ name: String, components: Int) throws -> MLShapedArray<Float> {
			guard let array = output.featureValue(for: name)?.multiArrayValue else {
				throw Failure.unexpectedOutput("missing \(name); got \(output.featureNames.sorted())")
			}
			let shaped = MLShapedArray<Float>(converting: array)
			guard shaped.scalarCount % components == 0 else { throw Failure.unexpectedOutput("shape of \(name)") }
			return shaped
		}
		let means = try tensor("mean_vectors_3d_positions", components: 3)
		let scales = try tensor("singular_values_scales", components: 3)
		let rotations = try tensor("quaternions_rotations", components: 4)
		let colors = try tensor("colors_rgb_linear", components: 3)
		let opacities = try tensor("opacities_alpha_channel", components: 1)
		let count = opacities.scalarCount
		guard means.scalarCount == count * 3, scales.scalarCount == count * 3,
			  rotations.scalarCount == count * 4, colors.scalarCount == count * 3 else {
			throw Failure.unexpectedOutput("mismatched Gaussian counts")
		}

		// The network predicts in a normalised frustum where x/z and y/z span -1...1 across the
		// square input. Stretch x and y back to the photo's real field of view and aspect ratio.
		let stretch = SIMD3(Float(photo.width) / 2 / photo.focalPx, Float(photo.height) / 2 / photo.focalPx, 1)
		let unproject = simd_float3x3(diagonal: stretch)

		var splats = ContiguousArray<GPUSplat>()
		splats.reserveCapacity(count)
		means.withUnsafeShapedBufferPointer { m, _, _ in
		scales.withUnsafeShapedBufferPointer { s, _, _ in
		rotations.withUnsafeShapedBufferPointer { r, _, _ in
		colors.withUnsafeShapedBufferPointer { c, _, _ in
		opacities.withUnsafeShapedBufferPointer { o, _, _ in
			for i in 0..<count {
				let opacity = o[i]
				let position = SIMD3(m[i * 3], m[i * 3 + 1], m[i * 3 + 2])
				guard opacity > 1 / 255, position.z > 0, position.z.isFinite else { continue }
				let q = simd_normalize(simd_quatf(ix: r[i * 4 + 1], iy: r[i * 4 + 2], iz: r[i * 4 + 3], r: r[i * 4]))
				let basis = unproject * simd_float3x3(q)
					* simd_float3x3(diagonal: SIMD3(s[i * 3], s[i * 3 + 1], s[i * 3 + 2]))
				let color = simd_clamp(SIMD3(c[i * 3], c[i * 3 + 1], c[i * 3 + 2]), .zero, .one)
				splats.append(GPUSplat(
					position: position * stretch, color: color, opacity: min(opacity, 1),
					covariance: basis * basis.transpose))
			}
		}}}}}
		guard !splats.isEmpty else { throw Failure.unexpectedOutput("no visible Gaussians") }

		let camera = SourceCamera(imageSize: SIMD2(Float(photo.width), Float(photo.height)), focalPx: photo.focalPx)
		return GaussianCloud(splats: splats, camera: camera)
	}

	// MARK: Image handling

	private struct Photo {
		var image: CGImage
		var width: Int
		var height: Int
		/// Focal length in pixels of `image`.
		var focalPx: Float
	}

	private static func loadPhoto(_ url: URL) throws -> Photo {
		guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { throw Failure.unreadableImage }
		// Decoding through the thumbnail path applies the EXIF orientation for us.
		let options: [CFString: Any] = [
			kCGImageSourceCreateThumbnailFromImageAlways: true,
			kCGImageSourceCreateThumbnailWithTransform: true,
			kCGImageSourceThumbnailMaxPixelSize: inputSide * 2,
		]
		guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
			throw Failure.unreadableImage
		}

		// Same fallback as Apple's reference code: assume a 30 mm (full-frame equivalent) lens.
		var focal35mm: Float = 30
		if let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
		   let exif = properties[kCGImagePropertyExifDictionary] as? [CFString: Any],
		   let value = (exif[kCGImagePropertyExifFocalLenIn35mmFilm] as? NSNumber)?.floatValue, value >= 1 {
			focal35mm = value
		}
		let diagonal = (Float(image.width * image.width + image.height * image.height)).squareRoot()
		let focalPx = focal35mm * diagonal / (36 * 36 + 24 * 24 as Float).squareRoot()
		return Photo(image: image, width: image.width, height: image.height, focalPx: focalPx)
	}

	/// Stretches the photo to the network's square input and packs it as planar RGB in 0...1.
	private static func imageTensor(_ image: CGImage) throws -> MLMultiArray {
		let side = inputSide
		guard let context = CGContext(
			data: nil, width: side, height: side, bitsPerComponent: 8, bytesPerRow: side * 4,
			space: CGColorSpace(name: CGColorSpace.sRGB)!,
			bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue), let pixels = context.data else {
			throw Failure.unreadableImage
		}
		context.interpolationQuality = .high
		context.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
		let rgba = pixels.bindMemory(to: UInt8.self, capacity: side * side * 4)

		let tensor = try MLMultiArray(shape: [1, 3, side, side] as [NSNumber], dataType: .float32)
		tensor.withUnsafeMutableBufferPointer(ofType: Float.self) { out, _ in
			let plane = side * side
			for i in 0..<plane {
				out[i] = Float(rgba[i * 4]) / 255
				out[plane + i] = Float(rgba[i * 4 + 1]) / 255
				out[plane * 2 + i] = Float(rgba[i * 4 + 2]) / 255
			}
		}
		return tensor
	}
}
