import Accelerate
import AVFoundation
import CoreImage
import Foundation

/// Sharp, evenly spaced stills pulled from a video, ready for camera tracking and splat training.
struct FrameSet: Sendable {
	struct Frame: Sendable {
		var url: URL
		/// Seconds from the start of the video.
		var time: Double
		var sharpness: Float
	}

	var directory: URL
	var frames: [Frame]
	/// How many video frames were looked at.
	var examined: Int
	/// Picked frames thrown away for being much blurrier than their neighbours.
	var droppedAsBlurry: Int
}

/// Picks the sharpest frame from each short stretch of a video and saves it as a JPEG.
enum FrameExtractor {
	enum Failure: LocalizedError {
		case noVideoTrack
		case unreadable(String)
		case noFrames

		var errorDescription: String? {
			switch self {
			case .noVideoTrack: "That file doesn't contain any video."
			case .unreadable(let detail): "The video couldn't be read (\(detail))."
			case .noFrames: "No usable frames were found in the video."
			}
		}
	}

	static let directory: URL = {
		let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
		return support.appendingPathComponent("SplatGen/Captures", isDirectory: true)
	}()

	/// Never pick frames closer together than this, in seconds.
	private static let minimumInterval = 0.25
	/// Longer videos are sampled more sparsely to stay around this many frames.
	private static let frameBudget = 300.0
	/// Saved frames are scaled down so their longer side is at most this many pixels.
	private static let maxSavedSide: CGFloat = 1920
	/// A picked frame is dropped when its sharpness falls below this fraction of its neighbours' median.
	private static let blurryFraction: Float = 0.4

	/// Extracts frames into a new folder under `directory`, reporting progress in 0...1.
	static func extract(from videoURL: URL, progress: @Sendable (Double) -> Void) async throws -> FrameSet {
		let asset = AVURLAsset(url: videoURL)
		guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw Failure.noVideoTrack }
		let duration = try await asset.load(.duration).seconds
		guard duration.isFinite, duration > 0 else { throw Failure.unreadable("unknown duration") }

		// The composition turns frames upright and converts HDR footage to standard dynamic range.
		let composition = try await AVMutableVideoComposition.videoComposition(withPropertiesOf: asset)
		composition.colorPrimaries = AVVideoColorPrimaries_ITU_R_709_2
		composition.colorTransferFunction = AVVideoTransferFunction_ITU_R_709_2
		composition.colorYCbCrMatrix = AVVideoYCbCrMatrix_ITU_R_709_2
		// High-frame-rate footage has more frames than are worth examining.
		if composition.frameDuration.seconds < 1.0 / 30 {
			composition.frameDuration = CMTime(value: 1, timescale: 30)
		}

		let reader = try AVAssetReader(asset: asset)
		let output = AVAssetReaderVideoCompositionOutput(
			videoTracks: [track],
			videoSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
		output.videoComposition = composition
		output.alwaysCopiesSampleData = false
		reader.add(output)
		guard reader.startReading() else {
			throw Failure.unreadable(reader.error?.localizedDescription ?? "couldn't start decoding")
		}

		let stamp = Date().formatted(Date.ISO8601FormatStyle(timeZone: .current).year().month().day().time(includingFractionalSeconds: false)
			.dateSeparator(.dash).timeSeparator(.omitted))
		let folder = directory.appendingPathComponent(
			"\(videoURL.deletingPathExtension().lastPathComponent) \(stamp)", isDirectory: true)
		let fm = FileManager.default
		try fm.createDirectory(at: folder, withIntermediateDirectories: true)

		do {
			let interval = max(minimumInterval, duration / frameBudget)
			let context = CIContext()
			let meter = SharpnessMeter()
			var frames: [FrameSet.Frame] = []
			var examined = 0
			var window = 0
			var best: (pixels: CVPixelBuffer, time: Double, sharpness: Float)?
			var lastPercent = -1

			func saveBest() throws {
				guard let picked = best else { return }
				best = nil
				let url = folder.appendingPathComponent(String(format: "frame_%04d.jpg", frames.count + 1))
				try save(picked.pixels, to: url, context: context)
				frames.append(FrameSet.Frame(url: url, time: picked.time, sharpness: picked.sharpness))
			}

			while let sample = output.copyNextSampleBuffer() {
				try Task.checkCancellation()
				guard let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
				let time = CMSampleBufferGetPresentationTimeStamp(sample).seconds
				examined += 1

				let sampleWindow = Int(time / interval)
				if sampleWindow != window { try saveBest() }
				window = sampleWindow

				let sharpness = meter.sharpness(of: pixels)
				if best == nil || sharpness > best!.sharpness { best = (pixels, time, sharpness) }

				let percent = Int(min(1, time / duration) * 100)
				if percent != lastPercent {
					lastPercent = percent
					progress(Double(percent) / 100)
				}
			}
			if reader.status == .failed {
				throw Failure.unreadable(reader.error?.localizedDescription ?? "decoding stopped early")
			}
			try saveBest()

			let kept = sharpEnough(frames)
			for frame in frames where !kept.contains(where: { $0.url == frame.url }) {
				try? fm.removeItem(at: frame.url)
			}
			guard !kept.isEmpty else { throw Failure.noFrames }
			return FrameSet(directory: folder, frames: kept, examined: examined, droppedAsBlurry: frames.count - kept.count)
		} catch {
			reader.cancelReading()
			try? fm.removeItem(at: folder)
			throw error
		}
	}

	/// Drops frames that are much softer than the frames picked just before and after them.
	/// Sharpness depends on what is in view, so frames are only ever compared with their neighbours.
	private static func sharpEnough(_ frames: [FrameSet.Frame]) -> [FrameSet.Frame] {
		frames.indices.compactMap { i in
			var neighbours = (max(0, i - 10)...min(frames.count - 1, i + 10)).filter { $0 != i }.map { frames[$0].sharpness }
			guard !neighbours.isEmpty else { return frames[i] }
			neighbours.sort()
			let median = neighbours[neighbours.count / 2]
			return frames[i].sharpness >= blurryFraction * median ? frames[i] : nil
		}
	}

	private static func save(_ pixels: CVPixelBuffer, to url: URL, context: CIContext) throws {
		var image = CIImage(cvPixelBuffer: pixels)
		let scale = min(1, maxSavedSide / max(image.extent.width, image.extent.height))
		if scale < 1 {
			image = image.applyingFilter("CILanczosScaleTransform", parameters: [kCIInputScaleKey: scale])
			let size = CGSize(width: image.extent.width.rounded(.down), height: image.extent.height.rounded(.down))
			image = image.cropped(to: CGRect(origin: image.extent.origin, size: size))
		}
		try context.writeJPEGRepresentation(
			of: image, to: url, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
			options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.92])
	}
}

/// Scores how crisp a frame is: the variance of its Laplacian, measured on a reduced greyscale copy.
/// Motion blur and missed focus flatten edges, which lowers the score.
private final class SharpnessMeter {
	/// Frames are reduced so their longer side is at most this before measuring.
	private static let maxSide = 960

	private var fullSize = (width: 0, height: 0)
	private var smallSize = (width: 0, height: 0)
	private var fullLuma: UnsafeMutablePointer<UInt8>?
	private var smallLuma: UnsafeMutablePointer<UInt8>?
	private var smallFloat: UnsafeMutablePointer<Float>?
	private var laplacian: UnsafeMutablePointer<Float>?

	deinit { release() }

	private func release() {
		fullLuma?.deallocate()
		smallLuma?.deallocate()
		smallFloat?.deallocate()
		laplacian?.deallocate()
	}

	private func prepare(width: Int, height: Int) {
		guard width != fullSize.width || height != fullSize.height else { return }
		release()
		fullSize = (width, height)
		let scale = min(1, Double(Self.maxSide) / Double(max(width, height)))
		smallSize = (max(3, Int(Double(width) * scale)), max(3, Int(Double(height) * scale)))
		fullLuma = .allocate(capacity: width * height)
		smallLuma = .allocate(capacity: smallSize.width * smallSize.height)
		smallFloat = .allocate(capacity: smallSize.width * smallSize.height)
		laplacian = .allocate(capacity: smallSize.width * smallSize.height)
	}

	/// `pixels` must be 32-bit BGRA.
	func sharpness(of pixels: CVPixelBuffer) -> Float {
		CVPixelBufferLockBaseAddress(pixels, .readOnly)
		defer { CVPixelBufferUnlockBaseAddress(pixels, .readOnly) }
		guard let base = CVPixelBufferGetBaseAddress(pixels) else { return 0 }
		let width = CVPixelBufferGetWidth(pixels), height = CVPixelBufferGetHeight(pixels)
		prepare(width: width, height: height)

		var source = vImage_Buffer(
			data: base, height: vImagePixelCount(height), width: vImagePixelCount(width),
			rowBytes: CVPixelBufferGetBytesPerRow(pixels))
		var full = vImage_Buffer(
			data: fullLuma, height: vImagePixelCount(height), width: vImagePixelCount(width), rowBytes: width)
		let smallHeight = vImagePixelCount(smallSize.height), smallWidth = vImagePixelCount(smallSize.width)
		var small = vImage_Buffer(data: smallLuma, height: smallHeight, width: smallWidth, rowBytes: smallSize.width)
		var float = vImage_Buffer(
			data: smallFloat, height: smallHeight, width: smallWidth, rowBytes: smallSize.width * MemoryLayout<Float>.stride)
		var edges = vImage_Buffer(
			data: laplacian, height: smallHeight, width: smallWidth, rowBytes: smallSize.width * MemoryLayout<Float>.stride)

		// Rec. 709 luma weights in B, G, R, A order, out of 256.
		let weights: [Int16] = [18, 183, 55, 0]
		let kernel: [Float] = [0, 1, 0, 1, -4, 1, 0, 1, 0]
		let flags = vImage_Flags(kvImageNoFlags)
		guard vImageMatrixMultiply_ARGB8888ToPlanar8(&source, &full, weights, 256, nil, 0, flags) == kvImageNoError,
			  vImageScale_Planar8(&full, &small, nil, flags) == kvImageNoError,
			  vImageConvert_Planar8toPlanarF(&small, &float, 255, 0, flags) == kvImageNoError,
			  vImageConvolve_PlanarF(&float, &edges, nil, 0, 0, kernel, 3, 3, 0, vImage_Flags(kvImageEdgeExtend)) == kvImageNoError
		else { return 0 }

		var mean: Float = 0, deviation: Float = 0
		vDSP_normalize(laplacian!, 1, nil, 1, &mean, &deviation, vDSP_Length(smallSize.width * smallSize.height))
		return deviation * deviation
	}
}
