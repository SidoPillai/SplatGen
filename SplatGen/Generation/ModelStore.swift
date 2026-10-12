import Foundation

/// Where the SHARP Core ML model lives on disk, and how to fetch it.
enum ModelStore {
	static let directory: URL = {
		let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
		return support.appendingPathComponent("SplatGen/Models", isDirectory: true)
	}()

	static let packageURL = directory.appendingPathComponent("sharp.mlpackage", isDirectory: true)
	/// Core ML compiles the package once; the result is cached here.
	static let compiledURL = directory.appendingPathComponent("sharp.mlmodelc", isDirectory: true)

	/// Community Core ML conversion of Apple's SHARP weights (Apple ML Research Model License).
	private static let remoteBase = URL(string: "https://huggingface.co/pearsonkyle/Sharp-coreml/resolve/main/sharp.mlpackage/")!
	private static let files: [(path: String, bytes: Int64)] = [
		("Manifest.json", 617),
		("Data/com.apple.CoreML/model.mlmodel", 938_769),
		("Data/com.apple.CoreML/weights/weight.bin", 2_672_576_384),
	]

	static var isInstalled: Bool {
		let fm = FileManager.default
		return fm.fileExists(atPath: compiledURL.path)
			|| files.allSatisfy { fm.fileExists(atPath: packageURL.appendingPathComponent($0.path).path) }
	}

	/// Copies a user-supplied `.mlpackage` or `.mlmodelc` into place.
	static func install(from url: URL) throws {
		let fm = FileManager.default
		try fm.createDirectory(at: directory, withIntermediateDirectories: true)
		let destination = url.pathExtension == "mlmodelc" ? compiledURL : packageURL
		try? fm.removeItem(at: compiledURL)
		try? fm.removeItem(at: packageURL)
		try fm.copyItem(at: url, to: destination)
	}

	/// Downloads the model package, reporting overall progress in 0...1.
	static func download(progress: @escaping @Sendable (Double) -> Void) async throws {
		let fm = FileManager.default
		let total = files.reduce(0) { $0 + $1.bytes }
		var completed: Int64 = 0
		for file in files {
			let destination = packageURL.appendingPathComponent(file.path)
			try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
			if !fm.fileExists(atPath: destination.path) {
				let base = completed
				let tracker = DownloadProgress { progress(Double(base + $0) / Double(total)) }
				let (temporary, response) = try await URLSession.shared.download(
					from: remoteBase.appendingPathComponent(file.path), delegate: tracker)
				guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
				try fm.moveItem(at: temporary, to: destination)
			}
			completed += file.bytes
			progress(Double(completed) / Double(total))
		}
	}
}

private final class DownloadProgress: NSObject, URLSessionDownloadDelegate {
	private let onBytes: @Sendable (Int64) -> Void

	init(onBytes: @escaping @Sendable (Int64) -> Void) {
		self.onBytes = onBytes
	}

	func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
					totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
		onBytes(totalBytesWritten)
	}

	func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
}
