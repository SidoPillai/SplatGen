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
		// Movement keys only reach the viewport while it has keyboard focus.
		if state.flyMode, view.window?.firstResponder !== view {
			view.window?.makeFirstResponder(view)
		}
	}
}

/// MTKView that turns mouse and trackpad input into camera moves.
final class InteractiveMetalView: MTKView {
	var state: ViewerState?

	/// Fly-mode movement keys by key code, as (right, up, forward): WASD, Q/E and the arrow keys.
	private static let movementKeys: [UInt16: SIMD3<Float>] = [
		13: SIMD3(0, 0, 1), 1: SIMD3(0, 0, -1), 0: SIMD3(-1, 0, 0), 2: SIMD3(1, 0, 0),
		12: SIMD3(0, -1, 0), 14: SIMD3(0, 1, 0),
		126: SIMD3(0, 0, 1), 125: SIMD3(0, 0, -1), 123: SIMD3(-1, 0, 0), 124: SIMD3(1, 0, 0),
	]
	private var heldKeys: Set<UInt16> = []
	private var resignKeyObserver: NSObjectProtocol?

	override var acceptsFirstResponder: Bool { true }

	override func viewDidMoveToWindow() {
		super.viewDidMoveToWindow()
		if let resignKeyObserver { NotificationCenter.default.removeObserver(resignKeyObserver) }
		resignKeyObserver = nil
		guard let window else { return }
		window.makeFirstResponder(self)
		// Key-up events are lost when the window loses focus, so stop moving then.
		resignKeyObserver = NotificationCenter.default.addObserver(
			forName: NSWindow.didResignKeyNotification, object: window, queue: .main
		) { [weak self] _ in
			MainActor.assumeIsolated { self?.releaseKeys() }
		}
	}

	override func resignFirstResponder() -> Bool {
		releaseKeys()
		return super.resignFirstResponder()
	}

	override func keyDown(with event: NSEvent) {
		guard state?.flyMode == true, !event.modifierFlags.contains(.command),
			  Self.movementKeys[event.keyCode] != nil else { return super.keyDown(with: event) }
		heldKeys.insert(event.keyCode)
		updateFlyInput()
	}

	override func keyUp(with event: NSEvent) {
		guard heldKeys.remove(event.keyCode) != nil else { return super.keyUp(with: event) }
		updateFlyInput()
	}

	override func flagsChanged(with event: NSEvent) {
		state?.flyBoost = event.modifierFlags.contains(.shift)
		super.flagsChanged(with: event)
	}

	private func releaseKeys() {
		heldKeys.removeAll()
		state?.flyBoost = false
		updateFlyInput()
	}

	private func updateFlyInput() {
		state?.flyInput = heldKeys.reduce(SIMD3<Float>(0, 0, 0)) { $0 + (Self.movementKeys[$1] ?? .zero) }
	}

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
