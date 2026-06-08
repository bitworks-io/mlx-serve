import XCTest
@testable import MLXCore

/// Pure helpers on TaskScheduler. The full headless-run flow needs a live server
/// (covered by manual e2e + the Phase 3 fake-engine integration test); these pin
/// the small, server-free seams.
final class TaskSchedulerHelpersTests: XCTestCase {

    func testDeriveTitleUsesFirstLineAndTrims() {
        XCTAssertEqual(TaskScheduler.deriveTitle(from: "Check HN\nand email me"), "Check HN")
        XCTAssertEqual(TaskScheduler.deriveTitle(from: "   spaced out   "), "spaced out")
        XCTAssertEqual(TaskScheduler.deriveTitle(from: ""), "Untitled task")
    }

    func testDeriveTitleTruncatesLongGoal() {
        let long = String(repeating: "x", count: 80)
        let title = TaskScheduler.deriveTitle(from: long)
        XCTAssertTrue(title.hasSuffix("…"))
        XCTAssertLessThanOrEqual(title.count, 49)
    }

    func testLastAssistantTextPicksLatestNonEmpty() {
        let msgs = [
            ChatMessage(role: .user, content: "do it"),
            ChatMessage(role: .assistant, content: "first"),
            ChatMessage(role: .assistant, content: "   "),     // empty/whitespace skipped
        ]
        XCTAssertEqual(TaskScheduler.lastAssistantText(msgs), "first")
    }

    func testLastAssistantTextNilWhenNoAssistant() {
        XCTAssertNil(TaskScheduler.lastAssistantText([ChatMessage(role: .user, content: "hi")]))
    }
}
