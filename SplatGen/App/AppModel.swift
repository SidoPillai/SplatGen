import AppKit
import Observation
import UniformTypeIdentifiers

@MainActor
@Observable
final class AppModel {
	enum Phase: Equatable {
		case idle
		case working(String)
		case ready
	}

	var phase: Phase = .idle
	var cloud: GaussianCloud?
	var title = "SplatGen"
	var summary = ""
	var errorMessage: String?

	var modelInstalled = ModelStore.isInstalled
	/// 0...1 while the model is downloading.
	var downloadProgress: Double?

	let viewer = ViewerState()
	private let generator = SharpGenerator()
	private var jobID = 0

	static let photoTypes: [UTType] = [.jpeg, .png, .heic, .heif, .tiff, .webP, .bmp, .gif]
	static let plyType = UTType(filenameExtension: "ply") ?? .data

	var isWorking: Bool {
		if case .working = phase { return true }
		return false
	}

	// MARK: Opening

	func chooseFile() {
		let panel = NSOpenPanel()
		panel.allowedContentTypes = Self.photoTypes + [Self.plyType]
		panel.message = "Choose a photo to turn into a 3D scene, or a .ply splat file to explore."
		if panel.runModal() == .OK, let url = panel.url { open(url) }
	}

	func open(_ url: URL) {
		guard !isWorking else { return }
		if url.pathExtension.lowercased() == "ply" {
			loadSplats(url)
		} else if modelInstalled {
			generate(from: url)
		} else {
			errorMessage = "Install the SHARP model first, then drop the photo again."
		}
	}

	private func generate(from url: URL) {
		phase = .working("Reading photo…")
		jobID += 1
		let job = jobID
		Task {
			let start = Date()
			do {
				let cloud = try await generator.generate(from: url) { message in
					// Progress hops to the main actor and can land after the job has finished; drop it then.
					Task { @MainActor in
						if self.jobID == job, self.isWorking { self.phase = .working(message) }
					}
				}
				show(cloud, named: url, detail: String(format: "generated in %.1f s", Date().timeIntervalSince(start)))
			} catch {
				fail(error)
			}
		}
	}

	private func loadSplats(_ url: URL) {
		phase = .working("Loading splats…")
		Task {
			do {
				let cloud = try await Task.detached { try PLY.read(url) }.value
				show(cloud, named: url, detail: "loaded from file")
			} catch {
				fail(error)
			}
		}
	}

	private func show(_ cloud: GaussianCloud, named url: URL, detail: String) {
		jobID += 1
		self.cloud = cloud
		title = url.deletingPathExtension().lastPathComponent
		summary = "\(cloud.count.formatted()) splats · \(detail)"
		phase = .ready
	}

	private func fail(_ error: Error) {
		jobID += 1
		errorMessage = error.localizedDescription
		phase = cloud == nil ? .idle : .ready
	}

	// MARK: Exporting

	func exportSplats() {
		guard let cloud, !isWorking else { return }
		let panel = NSSavePanel()
		panel.allowedContentTypes = [Self.plyType]
		panel.nameFieldStringValue = title + ".ply"
		guard panel.runModal() == .OK, let url = panel.url else { return }
		phase = .working("Exporting…")
		Task {
			do {
				try await Task.detached { try PLY.write(cloud, to: url) }.value
				phase = .ready
			} catch {
				fail(error)
			}
		}
	}

	func saveImage() {
		guard let image = viewer.snapshot() else { return }
		let panel = NSSavePanel()
		panel.allowedContentTypes = [.png]
		panel.nameFieldStringValue = title + ".png"
		guard panel.runModal() == .OK, let url = panel.url else { return }
		do {
			try Self.writePNG(image, to: url)
		} catch {
			fail(error)
		}
	}

	static func writePNG(_ image: CGImage, to url: URL) throws {
		guard let data = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:]) else {
			throw CocoaError(.fileWriteUnknown)
		}
		try data.write(to: url, options: .atomic)
	}

	// MARK: Model setup

	func downloadModel() {
		guard downloadProgress == nil else { return }
		downloadProgress = 0
		Task {
			do {
				try await ModelStore.download { fraction in
					Task { @MainActor in self.downloadProgress = fraction }
				}
			} catch {
				errorMessage = "The model download didn't finish: \(error.localizedDescription)"
			}
			downloadProgress = nil
			modelInstalled = ModelStore.isInstalled
		}
	}

	func chooseModel() {
		let panel = NSOpenPanel()
		panel.message = "Choose a SHARP Core ML model (sharp.mlpackage or sharp.mlmodelc)."
		panel.canChooseDirectories = true
		panel.treatsFilePackagesAsDirectories = false
		guard panel.runModal() == .OK, let url = panel.url else { return }
		guard ["mlpackage", "mlmodelc"].contains(url.pathExtension) else {
			errorMessage = "That isn't a Core ML model package."
			return
		}
		do {
			try ModelStore.install(from: url)
		} catch {
			errorMessage = error.localizedDescription
		}
		modelInstalled = ModelStore.isInstalled
	}
}
