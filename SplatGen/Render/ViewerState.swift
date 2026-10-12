import CoreGraphics
import Foundation
import Observation
import simd

enum MotionStyle: String, CaseIterable, Identifiable {
	case still = "Still"
	case sway = "Sway"
	case orbit = "Orbit"
	case breathe = "Breathe"

	var id: Self { self }
}

/// Camera and motion settings shared between the SwiftUI controls and the Metal renderer.
/// Only touched on the main thread.
@Observable
final class ViewerState {
	var motion: MotionStyle = .sway
	/// 0...2 multiplier on how far the idle motion and pointer parallax travel.
	var intensity: Float = 1
	var followPointer = true
	/// Depth quantile (0 = nearest, 1 = farthest) the camera pivots around.
	var focus: Float = 0.35
	/// Clip the view to the original photo's frame, hiding the rough splats the model extrapolates beyond it.
	var cropToFrame = true

	// User-driven camera offsets from the original photo viewpoint.
	@ObservationIgnored var yaw: Float = 0
	@ObservationIgnored var pitch: Float = 0
	/// Fraction of the focus distance travelled toward the pivot.
	@ObservationIgnored var dolly: Float = 0
	@ObservationIgnored var pan = SIMD2<Float>(0, 0)

	/// Renders the current view to an image; installed by the viewport.
	@ObservationIgnored var snapshot: () -> CGImage? = { nil }

	/// Focal length of the on-screen camera in view pixels, published by the renderer.
	@ObservationIgnored var viewFocal: Float = 1000

	/// Pointer position in the view, -1...1 with y pointing down; nil when outside.
	@ObservationIgnored var pointer: SIMD2<Float>?
	@ObservationIgnored var isDragging = false
	@ObservationIgnored private var smoothedPointer = SIMD2<Float>(0, 0)
	@ObservationIgnored private var motionWeight: Float = 1
	@ObservationIgnored private var clock: Float = 0

	func resetCamera() {
		yaw = 0; pitch = 0; dolly = 0; pan = .zero
	}

	func orbit(dx: Float, dy: Float) {
		yaw = max(-1.0, min(1.0, yaw + dx * 0.004))
		pitch = max(-0.7, min(0.7, pitch - dy * 0.004))
	}

	/// Slides the pivot so the content follows a drag of `dx`, `dy` view pixels.
	func pan(dx: Float, dy: Float) {
		pan.x = max(-1, min(1, pan.x - dx / viewFocal))
		pan.y = max(-1, min(1, pan.y - dy / viewFocal))
	}

	func zoom(by amount: Float) {
		dolly = max(-1.5, min(0.85, dolly + amount))
	}

	/// Advances the idle animation and returns the scene-to-camera transform.
	func viewMatrix(for cloud: GaussianCloud, deltaTime: Float) -> simd_float4x4 {
		let focusDepth = max(cloud.depth(atQuantile: focus), 0.05)
		let nearDepth = max(cloud.depth(atQuantile: 0.1), 0.05)

		// Angle that slides the camera sideways by a fixed fraction of the nearest content,
		// so parallax feels similar for a close-up portrait and a deep landscape.
		let amplitude = intensity * atan(0.07 * nearDepth / focusDepth)

		// Idle motion fades out while the user is dragging and eases back afterwards.
		let target: Float = isDragging ? 0 : 1
		motionWeight += (target - motionWeight) * min(1, deltaTime * 4)
		clock += deltaTime * motionWeight

		var yawOffset: Float = 0, pitchOffset: Float = 0, dollyOffset: Float = 0
		switch motion {
		case .still:
			break
		case .sway:
			yawOffset = sin(clock * 0.7)
			pitchOffset = 0.3 * sin(clock * 1.4 + 0.5)
		case .orbit:
			yawOffset = cos(clock * 0.9)
			pitchOffset = 0.6 * sin(clock * 0.9)
		case .breathe:
			dollyOffset = 0.06 * intensity * sin(clock * 0.8)
			yawOffset = 0.3 * sin(clock * 0.4)
		}

		let pointerTarget = (followPointer && !isDragging) ? (pointer ?? .zero) : .zero
		smoothedPointer += (pointerTarget - smoothedPointer) * min(1, deltaTime * 5)
		yawOffset = yawOffset * motionWeight - smoothedPointer.x * 1.2
		pitchOffset = pitchOffset * motionWeight + smoothedPointer.y * 0.8

		let totalYaw = yaw + yawOffset * amplitude
		let totalPitch = pitch + pitchOffset * amplitude
		let distance = focusDepth * (1 - dolly - dollyOffset * motionWeight)

		let rotation = simd_float3x3(simd_quatf(angle: totalYaw, axis: SIMD3(0, 1, 0)))
			* simd_float3x3(simd_quatf(angle: totalPitch, axis: SIMD3(1, 0, 0)))
		let pivot = SIMD3<Float>(0, 0, focusDepth) + rotation * SIMD3(pan.x, pan.y, 0) * focusDepth
		let eye = pivot + rotation * SIMD3(0, 0, -distance)

		let inverse = rotation.transpose
		let translation = -(inverse * eye)
		return simd_float4x4(
			SIMD4(inverse[0], 0), SIMD4(inverse[1], 0), SIMD4(inverse[2], 0), SIMD4(translation, 1))
	}
}
