import XCTest
@testable import MLXCore

/// Unit tests for the pure layer of `SystemGrounding` — the LAN-IP selection
/// policy and the grounding sentence. The `getifaddrs` enumeration itself is the
/// untestable I/O shell; everything that decides *which* address to surface and
/// *how* to phrase it is pure and pinned here.
final class SystemGroundingTests: XCTestCase {

    func testDateLineStatesTheClock() {
        // 2026-06-19 — the line must contain the formatted moment, not a guess.
        let line = SystemGrounding.dateTimeLine()
        XCTAssertTrue(line.contains("current date and time"))
    }

    func testPickPrefersWiFiThenEthernetThenOther() {
        let picked = SystemGrounding.pickLanIPv4(from: [
            ("en1", "10.0.0.5"),
            ("en0", "192.168.1.42"),
            ("utun3", "172.16.0.9"),
        ])
        XCTAssertEqual(picked, "192.168.1.42", "en0 (Wi-Fi) wins when present")
    }

    func testPickFallsBackToEthernetThenAny() {
        XCTAssertEqual(
            SystemGrounding.pickLanIPv4(from: [("utun3", "172.16.0.9"), ("en1", "10.0.0.5")]),
            "10.0.0.5", "en1 (Ethernet) when no en0")
        XCTAssertEqual(
            SystemGrounding.pickLanIPv4(from: [("bridge0", "192.168.64.1")]),
            "192.168.64.1", "any remaining interface when no en0/en1")
    }

    func testPickSkipsLoopbackAndLinkLocal() {
        let picked = SystemGrounding.pickLanIPv4(from: [
            ("lo0", "127.0.0.1"),
            ("en0", "169.254.10.10"),   // self-assigned link-local — not reachable
            ("en1", "192.168.0.7"),
        ])
        XCTAssertEqual(picked, "192.168.0.7", "loopback + link-local are filtered out")
    }

    func testPickReturnsNilWhenNothingUsable() {
        XCTAssertNil(SystemGrounding.pickLanIPv4(from: []))
        XCTAssertNil(SystemGrounding.pickLanIPv4(from: [("lo0", "127.0.0.1")]))
    }

    func testIPLineIncludesAddressAndReachabilityGuidance() {
        let line = SystemGrounding.localIPLine(ip: "192.168.1.42")
        XCTAssertTrue(line.contains("192.168.1.42"))
        XCTAssertTrue(line.contains("0.0.0.0"), "must steer the agent to bind 0.0.0.0")
        XCTAssertTrue(line.contains("http://192.168.1.42:"), "must show the reachable URL shape")
    }

    func testIPLineEmptyWhenOffline() {
        XCTAssertEqual(SystemGrounding.localIPLine(ip: nil), "")
        XCTAssertEqual(SystemGrounding.localIPLine(ip: ""), "")
    }
}
