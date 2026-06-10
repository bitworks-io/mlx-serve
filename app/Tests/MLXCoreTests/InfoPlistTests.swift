import XCTest

/// The release pipeline bundles Homebrew MLX dylibs + mlx.metallib built on the
/// macos-26 CI runner (LC_BUILD_VERSION minos 26.0). On older macOS the first
/// Metal call inside MLX throws and mlx-c's default handler calls exit(-1) —
/// the instant "Exit code 255" in issue #21. The declared LSMinimumSystemVersion
/// must match what the bundled libraries actually require, so LaunchServices
/// shows a proper "requires macOS 26" dialog instead of a silent crash.
final class InfoPlistTests: XCTestCase {
    func testMinimumSystemVersionCoversBundledMLXLibs() throws {
        let plistURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent() // MLXCoreTests
            .deletingLastPathComponent() // Tests
            .deletingLastPathComponent() // app
            .appendingPathComponent("Info.plist")
        let data = try Data(contentsOf: plistURL)
        let plist = try XCTUnwrap(
            PropertyListSerialization.propertyList(from: data, options: [], format: nil) as? [String: Any]
        )
        let minVersion = try XCTUnwrap(plist["LSMinimumSystemVersion"] as? String)
        let major = try XCTUnwrap(minVersion.split(separator: ".").first.flatMap { Int($0) })
        XCTAssertGreaterThanOrEqual(
            major, 26,
            "LSMinimumSystemVersion is \(minVersion), but the bundled MLX dylibs/metallib are built for macOS 26 and exit(255) on anything older (issue #21)"
        )
    }
}
