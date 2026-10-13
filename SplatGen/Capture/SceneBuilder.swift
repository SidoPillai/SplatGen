import AppKit
import Foundation
import simd

/// Turns frames from a video into a splat you can move through, using two external tools:
/// COLMAP works out where the camera was for each frame, then Brush trains the Gaussians.
enum SceneBuilder {
	enum Failure: LocalizedError {
		case toolsMissing([String])
		case toolFailed(tool: String, detail: String)
		case noCameraPath
		case noSplats

		var errorDescription: String? {
			switch self {
			case .toolsMissing(let names):
				"Building a scene needs \(names.joined(separator: " and ")), which couldn't be found on this Mac."
			case .toolFailed(let tool, let detail):
				"\(tool) stopped with an error:\n\(detail)"
			case .noCameraPath:
				"The camera's path couldn't be worked out from this video. "
					+ "It works best when the camera moves sideways past things that have plenty of detail."
			case .noSplats:
				"Training finished without producing a splat file."
			}
		}
	}

	struct Scene: Sendable {
		var cloud: GaussianCloud
		/// The finished splat, saved inside the scene's folder.
		var file: URL
		/// How many frames COLMAP managed to place.
		var framesUsed: Int
	}

	private static let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
		.appendingPathComponent("SplatGen", isDirectory: true)
	static let directory = support.appendingPathComponent("Scenes", isDirectory: true)
	/// Tools that aren't installed system-wide can be dropped anywhere inside this folder.
	static let toolsDirectory = support.appendingPathComponent("Tools", isDirectory: true)

	private static let trainingSteps = 10_000
	/// Every frame is matched against every other up to this many; beyond it only against its neighbours in time.
	private static let exhaustiveMatchingLimit = 100

	// MARK: Tools

	private static func find(_ names: [String]) -> URL? {
		let fm = FileManager.default
		let folders = ["/opt/homebrew/bin", "/usr/local/bin", NSHomeDirectory() + "/.cargo/bin"]
		for folder in folders {
			for name in names where fm.isExecutableFile(atPath: folder + "/" + name) {
				return URL(fileURLWithPath: folder + "/" + name)
			}
		}
		let bundled = fm.enumerator(at: toolsDirectory, includingPropertiesForKeys: nil)
		while let url = bundled?.nextObject() as? URL {
			if names.contains(url.lastPathComponent), fm.isExecutableFile(atPath: url.path),
			   (try? url.resourceValues(forKeys: [.isRegularFileKey]))?.isRegularFile == true {
				return url
			}
		}
		return nil
	}

	private static var colmap: URL? { find(["colmap"]) }
	private static var brush: URL? { find(["brush_app", "brush"]) }

	/// Names of the required tools that aren't installed.
	static var missingTools: [String] {
		(colmap == nil ? ["COLMAP"] : []) + (brush == nil ? ["Brush"] : [])
	}

	// MARK: Building

	static func build(from frames: FrameSet, status: @escaping @Sendable (String) -> Void) async throws -> Scene {
		guard let colmap, let brush else { throw Failure.toolsMissing(missingTools) }
		let fm = FileManager.default
		let name = frames.directory.lastPathComponent
		let scene = directory.appendingPathComponent(name, isDirectory: true)
		try? fm.removeItem(at: scene)
		try fm.createDirectory(at: scene, withIntermediateDirectories: true)
		let log = scene.appendingPathComponent("log.txt")
		fm.createFile(atPath: log.path, contents: nil)

		let database = scene.appendingPathComponent("database.db").path
		let sparse = scene.appendingPathComponent("sparse", isDirectory: true)
		let dense = scene.appendingPathComponent("dense", isDirectory: true)
		let text = scene.appendingPathComponent("text", isDirectory: true)
		let exports = scene.appendingPathComponent("exports", isDirectory: true)
		for folder in [sparse, text, exports] { try fm.createDirectory(at: folder, withIntermediateDirectories: true) }

		do {
			status("Finding landmarks in \(frames.frames.count) frames…")
			try await run(colmap, [
				"feature_extractor", "--database_path", database, "--image_path", frames.directory.path,
				"--ImageReader.single_camera", "1", "--ImageReader.camera_model", "SIMPLE_RADIAL",
			], in: scene, log: log)

			status("Matching frames…")
			if frames.frames.count <= exhaustiveMatchingLimit {
				try await run(colmap, ["exhaustive_matcher", "--database_path", database], in: scene, log: log)
			} else {
				try await run(colmap, [
					"sequential_matcher", "--database_path", database,
					"--SequentialMatching.overlap", "20", "--SequentialMatching.loop_detection", "0",
				], in: scene, log: log)
			}

			status("Working out the camera's path…")
			try await run(colmap, [
				"mapper", "--database_path", database, "--image_path", frames.directory.path, "--output_path", sparse.path,
			], in: scene, log: log)
			// COLMAP can split a video into several unconnected pieces; use the one that placed the most frames.
			func imagesSize(_ model: URL) -> Int {
				(try? model.appendingPathComponent("images.bin").resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
			}
			let models = (try? fm.contentsOfDirectory(at: sparse, includingPropertiesForKeys: nil)) ?? []
			guard let model = models.max(by: { imagesSize($0) < imagesSize($1) }), imagesSize(model) > 0 else {
				throw Failure.noCameraPath
			}

			status("Straightening frames…")
			try await run(colmap, [
				"image_undistorter", "--image_path", frames.directory.path, "--input_path", model.path,
				"--output_path", dense.path, "--output_type", "COLMAP",
			], in: scene, log: log)
			// Brush expects the model one level down, in sparse/0.
			let denseSparse = dense.appendingPathComponent("sparse", isDirectory: true)
			let denseModel = denseSparse.appendingPathComponent("0", isDirectory: true)
			let modelFiles = try fm.contentsOfDirectory(at: denseSparse, includingPropertiesForKeys: nil)
			try fm.createDirectory(at: denseModel, withIntermediateDirectories: true)
			for file in modelFiles { try fm.moveItem(at: file, to: denseModel.appendingPathComponent(file.lastPathComponent)) }
			try await run(colmap, [
				"model_converter", "--input_path", denseModel.path, "--output_path", text.path, "--output_type", "TXT",
			], in: scene, log: log)
			let viewpoint = try readViewpoint(in: text)

			status("Training the splat…")
			let poller = Task {
				var reported = 0
				while !Task.isCancelled {
					try? await Task.sleep(for: .seconds(2))
					if let step = latestExport(in: exports)?.step, step != reported, !Task.isCancelled {
						reported = step
						status("Training the splat… \(min(100, step * 100 / trainingSteps))%")
					}
				}
			}
			defer { poller.cancel() }
			// The viewer shows one colour per splat, so train without view-dependent colour (SH degree 0).
			try await run(brush, [
				dense.path, "--total-steps", String(trainingSteps), "--sh-degree", "0",
				"--export-every", String(trainingSteps / 10), "--export-path", exports.path,
				"--export-name", "splat_{iter}.ply",
			], in: scene, log: log)
			poller.cancel()
			guard let trained = latestExport(in: exports)?.url else { throw Failure.noSplats }

			status("Loading the scene…")
			let cloud = try PLY.read(trained).transformed(
				rotation: viewpoint.rotation, translation: viewpoint.translation, camera: viewpoint.camera)
			let file = scene.appendingPathComponent(name + ".ply")
			try PLY.write(cloud, to: file)
			// "target" is a tuning cache Brush leaves in its working folder.
			for folder in [dense, exports, text, scene.appendingPathComponent("target")] { try? fm.removeItem(at: folder) }
			try? fm.removeItem(atPath: database)
			return Scene(cloud: cloud, file: file, framesUsed: viewpoint.framesUsed)
		} catch {
			if error is CancellationError { try? fm.removeItem(at: scene) }
			throw error
		}
	}

	/// The newest `splat_<step>.ply` Brush has written.
	private static func latestExport(in folder: URL) -> (step: Int, url: URL)? {
		let files = (try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)) ?? []
		return files.compactMap { url -> (step: Int, url: URL)? in
			let name = url.deletingPathExtension().lastPathComponent
			guard url.pathExtension == "ply", name.hasPrefix("splat_"), let step = Int(name.dropFirst(6)) else { return nil }
			return (step, url)
		}.max { $0.step < $1.step }
	}

	/// Reads COLMAP's text export: the pose of the earliest placed frame, which becomes the viewer's
	/// starting viewpoint, and the lens shared by all frames.
	private static func readViewpoint(in text: URL) throws
		-> (rotation: simd_float3x3, translation: SIMD3<Float>, camera: SourceCamera, framesUsed: Int) {
		// Image lines: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME. The point lines between them hold triples.
		let images = try String(contentsOf: text.appendingPathComponent("images.txt"), encoding: .utf8)
			.split(whereSeparator: \.isNewline)
			.filter { !$0.hasPrefix("#") }
			.map { $0.split(separator: " ") }
			.filter { $0.count == 10 && $0[9].contains(".") }
		// Camera line: CAMERA_ID PINHOLE WIDTH HEIGHT FX FY CX CY.
		let lens = try String(contentsOf: text.appendingPathComponent("cameras.txt"), encoding: .utf8)
			.split(whereSeparator: \.isNewline)
			.map { $0.split(separator: " ") }
			.first { $0.count >= 6 && $0[1] == "PINHOLE" }
		guard let first = images.min(by: { $0[9] < $1[9] }), let lens,
			  let width = Float(lens[2]), let height = Float(lens[3]), let focal = Float(lens[4]) else {
			throw Failure.noCameraPath
		}
		let v = first[1...7].compactMap { Float($0) }
		guard v.count == 7 else { throw Failure.noCameraPath }
		let rotation = simd_float3x3(simd_normalize(simd_quatf(ix: v[1], iy: v[2], iz: v[3], r: v[0])))
		let camera = SourceCamera(imageSize: SIMD2(width, height), focalPx: focal, isFromPhoto: false)
		return (rotation, SIMD3(v[4], v[5], v[6]), camera, images.count)
	}

	// MARK: Running tools

	/// Runs a command-line tool to completion, appending its output to `log`.
	/// Cancelling the task, or quitting the app, stops the tool.
	private static func run(_ tool: URL, _ arguments: [String], in folder: URL, log: URL) async throws {
		try Task.checkCancellation()
		let output = try FileHandle(forWritingTo: log)
		try output.seekToEnd()
		defer { try? output.close() }

		let process = Process()
		process.executableURL = tool
		process.arguments = arguments
		process.currentDirectoryURL = folder
		process.standardOutput = output
		process.standardError = output
		let quitObserver = NotificationCenter.default.addObserver(
			forName: NSApplication.willTerminateNotification, object: nil, queue: nil
		) { _ in
			if process.isRunning { process.terminate() }
		}
		defer { NotificationCenter.default.removeObserver(quitObserver) }

		try await withTaskCancellationHandler {
			try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
				process.terminationHandler = { _ in continuation.resume() }
				do {
					try process.run()
				} catch {
					process.terminationHandler = nil
					continuation.resume(throwing: error)
				}
			}
		} onCancel: {
			if process.isRunning { process.terminate() }
		}
		try Task.checkCancellation()
		guard process.terminationStatus == 0 else {
			let lines = (try? String(contentsOf: log, encoding: .utf8))?.split(whereSeparator: \.isNewline) ?? []
			throw Failure.toolFailed(tool: tool.lastPathComponent, detail: lines.suffix(4).joined(separator: "\n"))
		}
	}
}

extension GaussianCloud {
	/// The same Gaussians moved into another frame (`rotation * p + translation`), seen through `camera`.
	func transformed(rotation: simd_float3x3, translation: SIMD3<Float>, camera: SourceCamera) -> GaussianCloud {
		var moved = ContiguousArray<GPUSplat>()
		moved.reserveCapacity(count)
		for splat in splats {
			let position = rotation * SIMD3(splat.position.x, splat.position.y, splat.position.z) + translation
			moved.append(GPUSplat(
				position: position, color: SIMD3(splat.color.x, splat.color.y, splat.color.z), opacity: splat.color.w,
				covariance: rotation * splat.covariance * rotation.transpose))
		}
		return GaussianCloud(splats: moved, camera: camera)
	}
}
