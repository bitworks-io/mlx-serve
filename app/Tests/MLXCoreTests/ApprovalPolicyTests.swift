import XCTest
@testable import MLXCore

/// The autonomy matrix is the security surface for unattended runs, so it gets the
/// most exhaustive unit coverage: 4 autonomy levels × read/write/shell tools ×
/// in-/out-of-workspace paths. A wrong `allow` here is how a "read-only" task ends
/// up running `rm`, so every cell is pinned.
final class ApprovalPolicyTests: XCTestCase {

    private let wd = "/tmp/run-folder"

    private func decide(_ tool: String, _ autonomy: TaskAutonomy,
                        args: [String: String] = [:],
                        wd: String? = "/tmp/run-folder") -> ApprovalDecision {
        ApprovalPolicy.decide(tool: tool, autonomy: autonomy, arguments: args,
                              rawArguments: "", workingDirectory: wd)
    }

    // MARK: yolo — everything allowed, even outside the folder

    func testYoloAllowsEverything() {
        for tool in ["readFile", "shell", "writeFile", "saveMemory", "frobnicate"] {
            XCTAssertEqual(decide(tool, .yolo, args: ["path": "/etc/passwd"], wd: nil), .allow,
                           "yolo should allow \(tool)")
        }
    }

    // MARK: readOnly — only observe; everything else asks

    func testReadOnlyAllowsReadTools() {
        for tool in ApprovalPolicy.readOnlyTools {
            XCTAssertEqual(decide(tool, .readOnly), .allow)
        }
    }

    func testReadOnlyAsksForWriteAndShell() {
        for tool in ["writeFile", "editFile", "shell", "saveMemory"] {
            guard case .ask = decide(tool, .readOnly, args: ["path": "out.txt"]) else {
                return XCTFail("readOnly should ask for \(tool)")
            }
        }
    }

    // MARK: workspace — confined writes auto, escapes + shell ask

    func testWorkspaceAllowsReads() {
        XCTAssertEqual(decide("searchFiles", .workspace), .allow)
    }

    func testWorkspaceAllowsConfinedWrite() {
        XCTAssertEqual(decide("writeFile", .workspace, args: ["path": "notes/out.txt"]), .allow)
    }

    func testWorkspaceAsksForEscapingWrite() {
        guard case .ask = decide("writeFile", .workspace, args: ["path": "/etc/hosts"]) else {
            return XCTFail("workspace should ask for an absolute write outside the folder")
        }
        // ".." traversal must also be caught.
        guard case .ask = decide("editFile", .workspace, args: ["path": "../../secrets.txt"]) else {
            return XCTFail("workspace should ask for a ..-escaping write")
        }
    }

    func testWorkspaceAsksForShellAndSaveMemory() {
        guard case .ask = decide("shell", .workspace, args: ["command": "ls"]) else {
            return XCTFail("workspace should ask before shell (uncconfinable)")
        }
        guard case .ask = decide("saveMemory", .workspace, args: ["memory": "x"]) else {
            return XCTFail("workspace should ask before saveMemory (writes outside folder)")
        }
    }

    // MARK: fullAuto — shell ok, file writes still confined

    func testFullAutoAllowsShellAndSaveMemory() {
        XCTAssertEqual(decide("shell", .fullAuto, args: ["command": "ls"]), .allow)
        XCTAssertEqual(decide("saveMemory", .fullAuto, args: ["memory": "x"]), .allow)
    }

    func testFullAutoStillConfinesFileWrites() {
        XCTAssertEqual(decide("writeFile", .fullAuto, args: ["path": "a/b.txt"]), .allow)
        guard case .ask = decide("writeFile", .fullAuto, args: ["path": "/etc/hosts"]) else {
            return XCTFail("fullAuto should still ask before writing outside the folder")
        }
    }

    // MARK: MCP-namespaced tools — allowed under fullAuto/yolo, ask otherwise

    func testMCPToolFullAutoAllowed() {
        XCTAssertEqual(decide("github__create_issue", .fullAuto, args: ["title": "x"]), .allow)
        XCTAssertEqual(decide("db__query", .yolo, args: [:], wd: nil), .allow)
    }

    func testMCPToolPausesAtLowerLevels() {
        for level in [TaskAutonomy.readOnly, .workspace] {
            guard case .ask = decide("slack__send_message", level, args: ["text": "hi"]) else {
                return XCTFail("\(level) should pause an MCP tool")
            }
        }
    }

    func testIsMCPTool() {
        XCTAssertTrue(ApprovalPolicy.isMCPTool("github__search"))
        XCTAssertFalse(ApprovalPolicy.isMCPTool("shell"))
    }

    // MARK: isConfined helper

    func testIsConfined() {
        XCTAssertTrue(ApprovalPolicy.isConfined("sub/f.txt", to: wd))
        XCTAssertTrue(ApprovalPolicy.isConfined(nil, to: wd))           // no path -> not an escape
        XCTAssertTrue(ApprovalPolicy.isConfined("\(wd)/x", to: wd))     // absolute but inside
        XCTAssertFalse(ApprovalPolicy.isConfined("/etc/passwd", to: wd))
        XCTAssertFalse(ApprovalPolicy.isConfined("../escape", to: wd))
        XCTAssertFalse(ApprovalPolicy.isConfined("f.txt", to: nil))     // can't prove confinement
    }
}
