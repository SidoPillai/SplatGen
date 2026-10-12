import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
	@Environment(AppModel.self) private var model
	@State private var isDropTargeted = false

	var body: some View {
		@Bindable var model = model
		ZStack {
			Color(red: 0.05, green: 0.05, blue: 0.06).ignoresSafeArea()

			if let cloud = model.cloud {
				SplatView(cloud: cloud, state: model.viewer) { message in
					Task { @MainActor in model.errorMessage = message }
				}
					.ignoresSafeArea()
				VStack {
					Spacer()
					ViewerControls(viewer: model.viewer, summary: model.summary)
						.padding(.bottom, 18)
				}
			} else {
				WelcomeView()
			}

			if case .working(let message) = model.phase {
				ProgressOverlay(message: message)
			}

			if isDropTargeted {
				RoundedRectangle(cornerRadius: 14)
					.strokeBorder(Color.accentColor, style: StrokeStyle(lineWidth: 3, dash: [10, 6]))
					.padding(10)
					.allowsHitTesting(false)
			}
		}
		.navigationTitle(model.title)
		.toolbar {
			ToolbarItemGroup {
				Button("Open", systemImage: "photo.badge.plus") { model.chooseFile() }
					.help("Open a photo or a .ply splat file")
					.disabled(model.isWorking)
				Button("Export", systemImage: "square.and.arrow.up") { model.exportSplats() }
					.help("Export the scene as a .ply splat file")
					.disabled(model.cloud == nil || model.isWorking)
				Button("Save Image", systemImage: "camera") { model.saveImage() }
					.help("Save the current view as a PNG")
					.disabled(model.cloud == nil)
			}
		}
		.dropDestination(for: URL.self) { urls, _ in
			guard let url = urls.first else { return false }
			model.open(url)
			return true
		} isTargeted: {
			isDropTargeted = $0
		}
		.alert("Something went wrong", isPresented: Binding(
			get: { model.errorMessage != nil },
			set: { if !$0 { model.errorMessage = nil } })
		) {
			Button("OK") { model.errorMessage = nil }
		} message: {
			Text(model.errorMessage ?? "")
		}
	}
}

private struct WelcomeView: View {
	@Environment(AppModel.self) private var model

	var body: some View {
		VStack(spacing: 18) {
			Image(systemName: "cube.transparent")
				.font(.system(size: 64, weight: .ultraLight))
				.foregroundStyle(.secondary)
			Text("Turn a photo into a living 3D view")
				.font(.title2.weight(.semibold))

			if model.modelInstalled {
				Text("Drop a photo here, or open one. Everything runs on this Mac.")
					.foregroundStyle(.secondary)
				Button("Open Photo…") { model.chooseFile() }
					.controlSize(.large)
					.buttonStyle(.borderedProminent)
			} else {
				ModelSetupView()
			}
		}
		.multilineTextAlignment(.center)
		.padding(40)
	}
}

private struct ModelSetupView: View {
	@Environment(AppModel.self) private var model

	var body: some View {
		VStack(spacing: 14) {
			Text("SplatGen needs Apple's SHARP model once (2.7 GB). After that it works fully offline.")
				.foregroundStyle(.secondary)
				.frame(maxWidth: 420)
			if let progress = model.downloadProgress {
				ProgressView(value: progress) {
					Text("Downloading… \(Int(progress * 100))%")
						.monospacedDigit()
				}
				.frame(width: 320)
			} else {
				HStack {
					Button("Download Model") { model.downloadModel() }
						.buttonStyle(.borderedProminent)
					Button("Choose Existing…") { model.chooseModel() }
				}
				.controlSize(.large)
			}
			Text("The SHARP weights are licensed by Apple for research use only.")
				.font(.footnote)
				.foregroundStyle(.tertiary)
		}
	}
}

private struct ProgressOverlay: View {
	let message: String

	var body: some View {
		VStack(spacing: 12) {
			ProgressView()
				.controlSize(.large)
			Text(message)
				.font(.callout)
		}
		.padding(28)
		.background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
	}
}

private struct ViewerControls: View {
	@Bindable var viewer: ViewerState
	let summary: String

	var body: some View {
		VStack(spacing: 8) {
			HStack(spacing: 16) {
				Picker("Motion", selection: $viewer.motion) {
					ForEach(MotionStyle.allCases) { Text($0.rawValue).tag($0) }
				}
				.pickerStyle(.segmented)
				.labelsHidden()
				.fixedSize()

				LabeledSlider(label: "Depth", systemImage: "move.3d", value: $viewer.intensity, range: 0...2)
					.help("How far the camera travels")
				LabeledSlider(label: "Focus", systemImage: "scope", value: $viewer.focus, range: 0.05...0.9)
					.help("Which depth stays still while the camera moves: near … far")

				Toggle(isOn: $viewer.cropToFrame) {
					Image(systemName: "crop")
				}
				.toggleStyle(.button)
				.help("Keep the view inside the original photo's frame")

				Toggle(isOn: $viewer.followPointer) {
					Image(systemName: "cursorarrow.motionlines")
				}
				.toggleStyle(.button)
				.help("Lean the view toward the pointer")

				Button("Reset", systemImage: "arrow.counterclockwise") { viewer.resetCamera() }
					.labelStyle(.iconOnly)
					.help("Return to the photo's viewpoint (double-click the view)")
			}
			Text(summary + " · drag to orbit, scroll to move in, ⌥-drag to pan")
				.font(.caption)
				.foregroundStyle(.secondary)
		}
		.padding(.horizontal, 18)
		.padding(.vertical, 12)
		.background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
	}
}

private struct LabeledSlider: View {
	let label: String
	let systemImage: String
	@Binding var value: Float
	let range: ClosedRange<Float>

	var body: some View {
		HStack(spacing: 6) {
			Image(systemName: systemImage)
				.foregroundStyle(.secondary)
				.accessibilityHidden(true)
			Slider(value: $value, in: range) { Text(label) }
				.labelsHidden()
				.frame(width: 90)
		}
	}
}
