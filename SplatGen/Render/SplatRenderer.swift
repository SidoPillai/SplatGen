import MetalKit
import simd

/// Must match `Uniforms` in Shaders.swift.
private struct SplatUniforms {
	var view: simd_float4x4
	var focalViewport: SIMD4<Float>
	var reveal: SIMD4<Float>
}

/// Draws a `GaussianCloud` as depth-sorted, alpha-blended screen-space quads.
final class SplatRenderer: NSObject, MTKViewDelegate {
	static let pixelFormat = MTLPixelFormat.bgra8Unorm_srgb
	static let clearColor = MTLClearColor(red: 0.004, green: 0.004, blue: 0.005, alpha: 1)

	let device: MTLDevice
	private let commandQueue: MTLCommandQueue
	private let pipeline: MTLRenderPipelineState
	private let state: ViewerState

	private var cloud: GaussianCloud?
	private var splatBuffer: MTLBuffer?

	// Sorted draw order. Three buffers rotate so the sorter never writes into one the GPU may still be reading.
	private var orderBuffers: [MTLBuffer] = []
	private var frontOrder = 0
	private var isSorting = false
	private var sortedDepthAxis = SIMD4<Float>(0, 0, 0, 0)
	private var sortGeneration = 0
	private let sortQueue = DispatchQueue(label: "SplatGen.sort", qos: .userInteractive)

	private var lastFrameTime = CACurrentMediaTime()
	private var revealProgress: Float = 1
	private var lastFrame: (viewMatrix: simd_float4x4, viewport: SIMD2<Float>)?

	init(device: MTLDevice, state: ViewerState) throws {
		guard let queue = device.makeCommandQueue() else { throw CocoaError(.featureUnsupported) }
		let library = try device.makeLibrary(source: Shaders.source, options: nil)
		let descriptor = MTLRenderPipelineDescriptor()
		descriptor.vertexFunction = library.makeFunction(name: "splatVertex")
		descriptor.fragmentFunction = library.makeFunction(name: "splatFragment")
		let attachment = descriptor.colorAttachments[0]!
		attachment.pixelFormat = Self.pixelFormat
		// Premultiplied alpha, composited back to front.
		attachment.isBlendingEnabled = true
		attachment.sourceRGBBlendFactor = .one
		attachment.sourceAlphaBlendFactor = .one
		attachment.destinationRGBBlendFactor = .oneMinusSourceAlpha
		attachment.destinationAlphaBlendFactor = .oneMinusSourceAlpha
		let pipeline = try device.makeRenderPipelineState(descriptor: descriptor)

		self.device = device
		self.commandQueue = queue
		self.pipeline = pipeline
		self.state = state
	}

	func setCloud(_ newCloud: GaussianCloud?) {
		cloud = newCloud
		sortGeneration += 1
		isSorting = false
		lastFrame = nil
		splatBuffer = nil
		orderBuffers = []
		guard let newCloud else { return }

		splatBuffer = newCloud.splats.withUnsafeBytes {
			device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
		}
		let identity = ContiguousArray<UInt32>(0..<UInt32(newCloud.count))
		orderBuffers = identity.withUnsafeBytes { bytes in
			(0..<3).compactMap { _ in device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared) }
		}
		frontOrder = 0
		sortedDepthAxis = .zero
		revealProgress = 0
		state.resetCamera()
	}

	func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

	func draw(in view: MTKView) {
		let now = CACurrentMediaTime()
		let deltaTime = Float(min(now - lastFrameTime, 0.1))
		lastFrameTime = now

		guard let descriptor = view.currentRenderPassDescriptor,
			  let drawable = view.currentDrawable,
			  let commandBuffer = commandQueue.makeCommandBuffer(),
			  let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: descriptor) else { return }

		if let cloud, view.drawableSize.width > 0 {
			let viewport = SIMD2(Float(view.drawableSize.width), Float(view.drawableSize.height))
			state.viewFocal = focal(for: cloud, viewport: viewport) / Float(view.window?.backingScaleFactor ?? 2)
			let viewMatrix = state.viewMatrix(for: cloud, deltaTime: deltaTime)
			revealProgress = min(1, revealProgress + deltaTime / 1.6)
			sortIfNeeded(cloud: cloud, viewMatrix: viewMatrix)
			encode(cloud, with: encoder, viewMatrix: viewMatrix, viewport: viewport)
			lastFrame = (viewMatrix, viewport)
		}

		encoder.endEncoding()
		commandBuffer.present(drawable)
		commandBuffer.commit()
	}

	/// Fits the original photo's frame inside the viewport.
	private func focal(for cloud: GaussianCloud, viewport: SIMD2<Float>) -> Float {
		cloud.camera.focalPx * min(viewport.x / cloud.camera.imageSize.x, viewport.y / cloud.camera.imageSize.y)
	}

	/// Variance, per unit of squared source depth, of the 3D smoothing filter applied in the shader.
	/// Sized from the spacing between splats as the source camera saw them, as in Mip-Splatting.
	private func smoothing(for cloud: GaussianCloud) -> Float {
		guard cloud.camera.isFromPhoto, cloud.count > 0 else { return 0 }
		let spacingPx = (cloud.camera.imageSize.x * cloud.camera.imageSize.y / Float(cloud.count)).squareRoot()
		let spacing = spacingPx / cloud.camera.focalPx
		return 0.2 * spacing * spacing
	}

	private func encode(_ cloud: GaussianCloud, with encoder: MTLRenderCommandEncoder,
						viewMatrix: simd_float4x4, viewport: SIMD2<Float>) {
		guard let splatBuffer, orderBuffers.count == 3 else { return }
		let focal = focal(for: cloud, viewport: viewport)
		var uniforms = SplatUniforms(
			view: viewMatrix,
			focalViewport: SIMD4(focal, focal, viewport.x, viewport.y),
			reveal: SIMD4(revealProgress, cloud.depth(atQuantile: 0.02), cloud.depth(atQuantile: 0.98), smoothing(for: cloud)))
		if state.cropToFrame, !state.flyMode, cloud.camera.isFromPhoto {
			let fit = focal / cloud.camera.focalPx
			let width = min(Int(viewport.x), Int((cloud.camera.imageSize.x * fit).rounded()))
			let height = min(Int(viewport.y), Int((cloud.camera.imageSize.y * fit).rounded()))
			encoder.setScissorRect(MTLScissorRect(
				x: (Int(viewport.x) - width) / 2, y: (Int(viewport.y) - height) / 2, width: width, height: height))
		}
		encoder.setRenderPipelineState(pipeline)
		encoder.setVertexBuffer(splatBuffer, offset: 0, index: 0)
		encoder.setVertexBuffer(orderBuffers[frontOrder], offset: 0, index: 1)
		encoder.setVertexBytes(&uniforms, length: MemoryLayout<SplatUniforms>.stride, index: 2)
		encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: cloud.count)
	}

	/// Renders the view as it was last shown into an image.
	func snapshot() -> CGImage? {
		guard let cloud, let (viewMatrix, viewport) = lastFrame else { return nil }
		let width = Int(viewport.x), height = Int(viewport.y)
		let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
			pixelFormat: Self.pixelFormat, width: width, height: height, mipmapped: false)
		textureDescriptor.usage = [.renderTarget, .shaderRead]
		let pass = MTLRenderPassDescriptor()
		guard let texture = device.makeTexture(descriptor: textureDescriptor),
			  let commandBuffer = commandQueue.makeCommandBuffer() else { return nil }
		pass.colorAttachments[0].texture = texture
		pass.colorAttachments[0].loadAction = .clear
		pass.colorAttachments[0].clearColor = Self.clearColor
		pass.colorAttachments[0].storeAction = .store
		guard let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: pass) else { return nil }
		encode(cloud, with: encoder, viewMatrix: viewMatrix, viewport: viewport)
		encoder.endEncoding()
		commandBuffer.commit()
		commandBuffer.waitUntilCompleted()

		var pixels = [UInt8](repeating: 0, count: width * height * 4)
		texture.getBytes(&pixels, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
		let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue
		guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
		return CGImage(
			width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
			space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: bitmapInfo),
			provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
	}

	// MARK: Depth sorting

	private func sortIfNeeded(cloud: GaussianCloud, viewMatrix: simd_float4x4) {
		// Camera-space depth of a point is dot(axis.xyz, p) + axis.w.
		let axis = SIMD4(viewMatrix[0][2], viewMatrix[1][2], viewMatrix[2][2], viewMatrix[3][2])
		guard !isSorting, simd_distance(axis, sortedDepthAxis) > 1e-4 else { return }
		isSorting = true

		let generation = sortGeneration
		let target = (frontOrder + 1) % 3
		let buffer = orderBuffers[target]
		sortQueue.async { [weak self] in
			Self.sortBackToFront(cloud: cloud, axis: axis, into: buffer)
			DispatchQueue.main.async {
				guard let self, self.sortGeneration == generation else { return }
				self.frontOrder = target
				self.sortedDepthAxis = axis
				self.isSorting = false
			}
		}
	}

	/// 16-bit counting sort of splat indices by camera depth, farthest first. Keys are spaced evenly in
	/// log depth so a few distant splats don't starve the foreground of buckets.
	private static func sortBackToFront(cloud: GaussianCloud, axis: SIMD4<Float>, into buffer: MTLBuffer) {
		let count = cloud.count
		let out = buffer.contents().bindMemory(to: UInt32.self, capacity: count)
		cloud.positions.withUnsafeBufferPointer { positions in
			var minDepth = Float.greatestFiniteMagnitude, maxDepth = -Float.greatestFiniteMagnitude
			for i in 0..<count {
				let d = simd_dot(positions[i], axis)
				minDepth = min(minDepth, d)
				maxDepth = max(maxDepth, d)
			}
			// Splats nearer than this are culled by the shader, so their order doesn't matter.
			let near = max(minDepth, 0.02)
			let far = max(maxDepth, near)
			let logFar = log(far)
			let buckets = 1 << 16
			let scale = Float(buckets - 1) / max(logFar - log(near), 1e-9)
			let keys = UnsafeMutablePointer<UInt16>.allocate(capacity: count)
			let offsets = UnsafeMutablePointer<UInt32>.allocate(capacity: buckets + 1)
			defer { keys.deallocate(); offsets.deallocate() }
			offsets.update(repeating: 0, count: buckets + 1)

			for i in 0..<count {
				let depth = max(near, min(far, simd_dot(positions[i], axis)))
				let key = UInt16(max(0, min(Float(buckets - 1), (logFar - log(depth)) * scale)))
				keys[i] = key
				offsets[Int(key) + 1] += 1
			}
			for b in 1...buckets { offsets[b] += offsets[b - 1] }
			for i in 0..<count {
				let key = Int(keys[i])
				out[Int(offsets[key])] = UInt32(i)
				offsets[key] += 1
			}
		}
	}
}
