import SwiftUI

@main
struct SplatGenApp: App {
	@State private var model = AppModel()

	var body: some Scene {
		Window("SplatGen", id: "main") {
			ContentView()
				.environment(model)
				.frame(minWidth: 720, minHeight: 520)
				.preferredColorScheme(.dark)
		}
		.defaultSize(width: 1100, height: 760)
		// Always start with the window open, whatever state the last session ended in.
		.defaultLaunchBehavior(.presented)
		.restorationBehavior(.disabled)
		.commands {
			CommandGroup(replacing: .newItem) {
				Button("Open…") { model.chooseFile() }
					.keyboardShortcut("o")
				Button("Export Splats…") { model.exportSplats() }
					.keyboardShortcut("e")
					.disabled(model.cloud == nil || model.isWorking)
				Button("Save Image…") { model.saveImage() }
					.keyboardShortcut("s")
					.disabled(model.cloud == nil)
			}
			CommandMenu("Camera") {
				Button("Reset Camera") { model.viewer.resetCamera() }
					.keyboardShortcut("0")
					.disabled(model.cloud == nil)
			}
		}
	}
}
