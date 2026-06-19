import XCTest
@testable import MLXCore

/// Unit tests for `ChatRowBuilder` — folding the agent's separate tool-call and
/// tool-result summary messages into one collapsible transcript row.
@MainActor
final class ChatRowBuilderTests: XCTestCase {

    private func call(_ name: String = "dbhub__execute_sql",
                      _ args: String = "sql: SELECT 1") -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: "**\(name)**(\(args))")
        m.isAgentSummary = true
        return m
    }
    private func result(_ name: String = "dbhub__execute_sql",
                        _ out: String = "{\"ok\":true}") -> ChatMessage {
        var m = ChatMessage(role: .assistant, content: "**\(name)** → \(out)")
        m.isAgentSummary = true
        return m
    }
    private func normal(_ text: String) -> ChatMessage {
        ChatMessage(role: .assistant, content: text)
    }
    private func hiddenToolMsg() -> ChatMessage {
        var m = ChatMessage(role: .system, content: "raw tool output")
        m.toolCallId = "call_1"
        m.toolName = "dbhub__execute_sql"
        return m
    }

    func testCallAndResultFoldIntoOneRow() {
        let rows = ChatRowBuilder.rows(from: [call(), result()])
        XCTAssertEqual(rows.count, 1)
        guard case .toolCall(_, let results) = rows[0] else { return XCTFail("expected toolCall row") }
        XCTAssertEqual(results.count, 1)
    }

    func testRawToolResultMessageStaysHiddenButResultSummaryGroups() {
        // The role:.system message with a toolCallId is the hidden raw result; it
        // must be filtered out, and the call+result-summary still fold together.
        let rows = ChatRowBuilder.rows(from: [call(), hiddenToolMsg(), result()])
        XCTAssertEqual(rows.count, 1, "the hidden raw tool message must not produce a row")
        guard case .toolCall(_, let results) = rows[0] else { return XCTFail("expected toolCall row") }
        XCTAssertEqual(results.count, 1)
    }

    func testMultiCallRoundGroupsAllResultsUnderTheCall() {
        // One call summary (two tools) followed by two result summaries.
        var twoCalls = ChatMessage(role: .assistant,
            content: "**a**(x: 1)\n**b**(y: 2)")
        twoCalls.isAgentSummary = true
        let rows = ChatRowBuilder.rows(from: [twoCalls, result("a"), result("b")])
        XCTAssertEqual(rows.count, 1)
        guard case .toolCall(_, let results) = rows[0] else { return XCTFail("expected toolCall row") }
        XCTAssertEqual(results.count, 2)
    }

    func testNormalMessagesArePassedThroughAroundGroups() {
        let rows = ChatRowBuilder.rows(from: [normal("hi"), call(), result(), normal("done")])
        XCTAssertEqual(rows.count, 3)
        if case .message = rows[0] {} else { XCTFail("row 0 should be a message") }
        if case .toolCall = rows[1] {} else { XCTFail("row 1 should be a toolCall") }
        if case .message = rows[2] {} else { XCTFail("row 2 should be a message") }
    }

    func testLoneResultWithoutCallRendersAsMessage() {
        // Defensive: a result summary with no preceding call (e.g. a resumed task
        // run) must still render, not vanish.
        let rows = ChatRowBuilder.rows(from: [result()])
        XCTAssertEqual(rows.count, 1)
        if case .message = rows[0] {} else { XCTFail("lone result should fall back to a message row") }
    }

    func testCallWithNoResultYetIsAGroupWithEmptyResults() {
        // Mid-execution: the call summary exists, results not appended yet.
        var streaming = call()
        streaming.isStreaming = true
        let rows = ChatRowBuilder.rows(from: [streaming])
        XCTAssertEqual(rows.count, 1)
        guard case .toolCall(_, let results) = rows[0] else { return XCTFail("expected toolCall row") }
        XCTAssertTrue(results.isEmpty)
    }

    func testClassificationDiscriminatesCallVsResult() {
        XCTAssertTrue(ChatRowBuilder.isCallSummary(call()))
        XCTAssertFalse(ChatRowBuilder.isResultSummary(call()))
        XCTAssertTrue(ChatRowBuilder.isResultSummary(result()))
        XCTAssertFalse(ChatRowBuilder.isCallSummary(result()))
        // A denied result is still a result.
        var denied = ChatMessage(role: .assistant, content: "**shell** → denied by user")
        denied.isAgentSummary = true
        XCTAssertTrue(ChatRowBuilder.isResultSummary(denied))
        // Non-agent-summary content is neither.
        XCTAssertFalse(ChatRowBuilder.isCallSummary(normal("**a** → b")))
    }
}
