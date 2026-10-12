# SplatGen

A macOS app that turns a single photo into a 3D Gaussian-splat scene you can move through. Generation and rendering both run entirely on device.

Drop in a photo; a few seconds later it becomes a scene with real parallax that drifts gently on its own and responds to the pointer.

## Requirements

- Apple silicon Mac, macOS 15 or later
- Xcode 16 or later (the project uses file-system-synchronized groups)
- About 3 GB of disk space for the model

## Build and run

Open `SplatGen.xcodeproj` in Xcode and run the `SplatGen` scheme, or build from the command line:

```bash
xcodebuild -project SplatGen.xcodeproj -scheme SplatGen -configuration Release -derivedDataPath build build
```

```bash
open build/Build/Products/Release/SplatGen.app
```

Use the Release configuration for normal use. Debug builds work but decode and depth-sort noticeably slower.

## The model

SplatGen uses Apple's [SHARP](https://github.com/apple/ml-sharp) network, which predicts a full set of 3D Gaussians from one image in a single forward pass. The app runs it through Core ML using the community conversion at [pearsonkyle/Sharp-coreml](https://huggingface.co/pearsonkyle/Sharp-coreml).

The model (2.7 GB) is not part of this repository. On first launch the app offers to download it, or you can point it at an existing `sharp.mlpackage` or `sharp.mlmodelc`. It is stored in:

```
~/Library/Application Support/SplatGen/Models
```

The first generation takes longer (around 30 seconds on an M2 Pro) while Core ML compiles the model for the machine; the compiled model is cached next to the package. After that a photo takes roughly 5–8 seconds.

**Licence:** the SHARP weights are released under the Apple Machine Learning Research Model License, which permits non-commercial research use only. An app that ships these weights, or depends on them, cannot be released commercially as-is.

## Using the app

| Action | Input |
| --- | --- |
| Open a photo or `.ply` | Drag onto the window, or ⌘O |
| Orbit | Drag |
| Move in and out | Scroll or pinch |
| Pan | ⌥-drag or right-drag |
| Reset to the photo's viewpoint | Double-click, or ⌘0 |
| Export splats as `.ply` | ⌘E |
| Save the current view as PNG | ⌘S |

The control bar at the bottom of the viewer sets:

- **Motion** — how the camera moves when left alone: Still, Sway, Orbit or Breathe.
- **Depth** — how far the camera travels.
- **Focus** — which depth stays fixed while the camera moves, from near to far.
- **Crop** — clip the view to the original photo's frame. SHARP emits rough splats beyond the photo's edges, so this is on by default.
- **Follow pointer** — lean the view toward the pointer.

A single photo only supports modest camera moves. Beyond roughly 30° of orbit, regions the photo never saw open up and the scene visibly tears.

## How it works

1. **Photo in.** The image is decoded with its EXIF orientation applied and stretched to SHARP's fixed 1536 × 1536 input. Focal length comes from EXIF when present, otherwise a 30 mm full-frame-equivalent lens is assumed, matching Apple's reference code.
2. **Prediction.** Core ML returns about 1.2 million Gaussians (position, scale, rotation, linear colour, opacity) in a normalised frustum.
3. **Un-projection.** Positions and covariances are scaled back to the photo's real field of view and aspect ratio, giving a metric scene with the camera at the origin looking down +z (OpenCV convention: x right, y down).
4. **Rendering.** Each Gaussian is drawn as an instanced screen-space quad. The vertex shader projects the 3D covariance to a 2D ellipse; fragments are alpha-blended back to front in linear colour. Draw order comes from a 16-bit counting sort that runs off the main thread whenever the camera moves.

Exported `.ply` files use the standard 3D Gaussian Splatting layout, plus the camera elements SHARP's own tools write, so they open in other splat viewers and reopen here with the original framing.

## Project layout

```
SplatGen/
  App/          SwiftUI app, window, controls and app state
  Generation/   Model storage/download and SHARP inference through Core ML
  Render/       Metal renderer, shader source, camera and input handling
  Splat/        Gaussian cloud data model and .ply reading/writing
  Assets.xcassets/  App icon
Tools/
  MakeAppIcon.swift  Script that draws the app icon (not part of the app target)
```

To regenerate the icon artwork, run `swiftc -O -o /tmp/mkicon Tools/MakeAppIcon.swift && /tmp/mkicon icon.png`, then resize the result into `AppIcon.appiconset`.

The Metal shader lives in `Render/Shaders.swift` as a source string compiled at launch, so the project builds without Xcode's separately installed Metal Toolchain component.

## License

The code in this repository is released under the [MIT License](LICENSE).

That license does not cover the SHARP model, which is downloaded separately and remains under the Apple Machine Learning Research Model License (non-commercial research use only).

## Notes

- The app is unsandboxed and signed to run locally; it is not set up for distribution.
- There are no automated tests. Changes are checked by building and exercising the app.
