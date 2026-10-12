import MetalKit
import SwiftUI

/// SwiftUI wrapper around the Metal splat viewport.
struct SplatView: NSViewRepresentable {
	let cloud: GaussianCloud
	let state: ViewerState
	let onFailure: (String) -> Void

	final class Coordinator {
		var renderer: SplatRenderer?
		var cloud: GaussianCloud?
	}

	func makeCoordinator() -> Coordinator { Coordinator() }

	func makeNSView(context: Context) -> InteractiveMetalView {
		let view = InteractiveMetalView(frame: .zero, device: MTLCreateSystemDefaultDevice())
		view.state = state
		view.colorPixelFormat = SplatRenderer.pixelFormat
		view.clearColor = SplatRenderer.clearColor
		view.preferredFramesPerSecond = 60
		do {
			guard let device = view.device else { throw CocoaError(.featureUnsupported) }
			let renderer = try SplatRenderer(device: device, state: state)
			context.coordinator.renderer = renderer
			state.snapshot = { [weak renderer] in renderer?.snapshot() }
			view.delegate = renderer
		} catch {
			onFailure("The 3D view couldn't start on this Mac's GPU: \(error.localizedDescription)")
		}
		return view
	}

	func updateNSView(_ view: InteractiveMetalView, context: Context) {
		if context.coordinator.cloud !== cloud {
			context.coordinator.cloud = cloud
			context.coordinator.renderer?.setCloud(cloud)
		}
	}
}

/// MTKView that turns mouse and trackpad input into camera moves.
final class InteractiveMetalView: MTKView {
	var state: ViewerState?

	override var acceptsFirstResponder: Bool { true }

	override func updateTrackingAreas() {
		super.updateTrackingAreas()
		trackingAreas.forEach(removeTrackingArea)
		addTrackingArea(NSTrackingArea(
			rect: .zero,
			options: [.mouseMoved, .mouseEnteredAndExited, .activeInKeyWindow, .inVisibleRect],
			owner: self))
	}

	override func mouseDown(with event: NSEvent) {
		window?.makeFirstResponder(self)
		if event.clickCount == 2 {
			state?.resetCamera()
		} else {
			state?.isDragging = true
		}
	}

	override func mouseDragged(with event: NSEvent) {
		if event.modifierFlags.contains(.option) {
			state?.pan(dx: Float(event.deltaX), dy: Float(event.deltaY))
		} else {
			state?.orbit(dx: Float(event.deltaX), dy: Float(event.deltaY))
		}
	}

	override func mouseUp(with event: NSEvent) {
		state?.isDragging = false
		updatePointer(with: event)
	}

	override func rightMouseDown(with event: NSEvent) {
		state?.isDragging = true
	}

	override func rightMouseDragged(with event: NSEvent) {
		state?.pan(dx: Float(event.deltaX), dy: Float(event.deltaY))
	}

	override func rightMouseUp(with event: NSEvent) {
		state?.isDragging = false
	}

	override func scrollWheel(with event: NSEvent) {
		let scale: Float = event.hasPreciseScrollingDeltas ? 0.002 : 0.02
		state?.zoom(by: Float(event.scrollingDeltaY) * scale)
	}

	override func magnify(with event: NSEvent) {
		state?.zoom(by: Float(event.magnification) * 0.6)
	}

	override func mouseMoved(with event: NSEvent) {
		updatePointer(with: event)
	}

	override func mouseExited(with event: NSEvent) {
		state?.pointer = nil
	}

	private func updatePointer(with event: NSEvent) {
		guard bounds.width > 0, bounds.height > 0 else { return }
		let p = convert(event.locationInWindow, from: nil)
		state?.pointer = SIMD2(
			Float(p.x / bounds.width) * 2 - 1,
			1 - Float(p.y / bounds.height) * 2)
	}
}
