import AppKit
import CoreGraphics

// Draws the SplatGen icon: a sunset photo, tilted in space, breaking apart into Gaussian splats.
let size = 1024
let cs = CGColorSpace(name: CGColorSpace.sRGB)!
let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
ctx.translateBy(x: 0, y: CGFloat(size)); ctx.scaleBy(x: 1, y: -1) // y down

var seed: UInt64 = 0x5EED1234
func rnd() -> Double { seed = seed &* 6364136223846793005 &+ 1442695040888963407; return Double((seed >> 33) & 0xFFFFFF) / Double(0xFFFFFF) }
func mix(_ a: [Double], _ b: [Double], _ t: Double) -> [Double] { zip(a, b).map { $0 + ($1 - $0) * max(0, min(1, t)) } }

// macOS icon plate
let plate = CGRect(x: 100, y: 100, width: 824, height: 824)
let platePath = CGPath(roundedRect: plate, cornerWidth: 186, cornerHeight: 186, transform: nil)
ctx.saveGState()
ctx.setShadow(offset: CGSize(width: 0, height: -12), blur: 28, color: CGColor(gray: 0, alpha: 0.45))
ctx.addPath(platePath); ctx.setFillColor(CGColor(gray: 0.05, alpha: 1)); ctx.fillPath()
ctx.restoreGState()
ctx.saveGState()
ctx.addPath(platePath); ctx.clip()
let bg = CGGradient(colorsSpace: cs, colors: [CGColor(red: 0.10, green: 0.11, blue: 0.20, alpha: 1), CGColor(red: 0.02, green: 0.02, blue: 0.05, alpha: 1)] as CFArray, locations: [0, 1])!
ctx.drawRadialGradient(bg, startCenter: CGPoint(x: 400, y: 330), startRadius: 0, endCenter: CGPoint(x: 512, y: 512), endRadius: 680, options: [.drawsAfterEndLocation])

func photo(_ u: Double, _ v: Double) -> [Double] {
    let horizon = 0.60
    var c: [Double]
    if v < horizon {
        let t = v / horizon
        c = t < 0.55 ? mix([0.13, 0.10, 0.42], [0.80, 0.24, 0.58], t / 0.55) : mix([0.80, 0.24, 0.58], [1.0, 0.66, 0.26], (t - 0.55) / 0.45)
        let d = ((u - 0.52) * (u - 0.52) + (v - 0.44) * (v - 0.44)).squareRoot()
        if d < 0.13 { c = [1.0, 0.95, 0.74] } else { c = mix([1.0, 0.86, 0.5], c, (d - 0.13) / 0.16) }
        let ridge = 0.17 * max(0, 1 - abs(u - 0.22) / 0.24) + 0.11 * max(0, 1 - abs(u - 0.80) / 0.22)
        if v > horizon - ridge { c = [0.16, 0.10, 0.34] }
    } else {
        let t = (v - horizon) / (1 - horizon)
        c = mix([0.10, 0.30, 0.50], [0.03, 0.08, 0.20], t)
        let glint = max(0, 1 - abs(u - 0.52) / (0.05 + 0.10 * t))
        c = mix(c, [1.0, 0.72, 0.40], glint * 0.75)
    }
    return c
}

struct Blob { var x, y, z, r: Double; var color: [Double]; var alpha: Double }
var blobs: [Blob] = []
let n = 26
for j in 0..<n { for i in 0..<n {
    let u = (Double(i) + 0.5) / Double(n), v = (Double(j) + 0.5) / Double(n)
    var x = (u - 0.5) * 2, y = (v - 0.5) * 2, z = 0.0
    var r = 1.25 / Double(n), alpha = 0.95
    // The right side of the photo lifts off and scatters toward the viewer.
    let t = max(0, (u - 0.50) / 0.50), k = t * t
    x += k * (rnd() - 0.30) * 0.75
    y += k * (rnd() - 0.5) * 0.95
    z -= k * (0.25 + rnd() * 1.5)
    r *= 1 + k * (rnd() * 1.6 - 0.25)
    alpha -= k * rnd() * 0.45
    if k > 0.2 && rnd() < k * 0.45 { continue }
    blobs.append(Blob(x: x, y: y, z: z, r: r, color: photo(u, v), alpha: alpha))
}}

// Rotate about the vertical axis, tilt slightly, then project.
let yaw = -0.62, pitch = 0.16, dist = 4.2, focal = 1180.0
func project(_ b: Blob) -> (CGPoint, Double, Double) {
    let x1 = b.x * cos(yaw) + b.z * sin(yaw), z1 = -b.x * sin(yaw) + b.z * cos(yaw)
    let y2 = b.y * cos(pitch) - z1 * sin(pitch), z2 = b.y * sin(pitch) + z1 * cos(pitch)
    let d = dist + z2
    return (CGPoint(x: 500 + focal * x1 / d, y: 512 + focal * y2 / d), focal * b.r / d, d)
}
let projected = blobs.map { ($0, project($0)) }.sorted { $0.1.2 > $1.1.2 }
for (b, (p, r, _)) in projected {
    let c = b.color
    let g = CGGradient(colorsSpace: cs, colors: [
        CGColor(red: c[0], green: c[1], blue: c[2], alpha: b.alpha),
        CGColor(red: c[0], green: c[1], blue: c[2], alpha: b.alpha * 0.85),
        CGColor(red: c[0], green: c[1], blue: c[2], alpha: 0)] as CFArray, locations: [0, 0.55, 1])!
    ctx.drawRadialGradient(g, startCenter: p, startRadius: 0, endCenter: p, endRadius: r * 1.5, options: [])
}
// Soft top highlight on the plate
let sheen = CGGradient(colorsSpace: cs, colors: [CGColor(gray: 1, alpha: 0.10), CGColor(gray: 1, alpha: 0)] as CFArray, locations: [0, 1])!
ctx.drawLinearGradient(sheen, start: CGPoint(x: 512, y: 100), end: CGPoint(x: 512, y: 420), options: [])
ctx.restoreGState()

let rep = NSBitmapImageRep(cgImage: ctx.makeImage()!)
try! rep.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
