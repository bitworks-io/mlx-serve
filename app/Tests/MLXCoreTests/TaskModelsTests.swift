import XCTest
@testable import MLXCore

/// Codable round-trip + backfill for the scheduled-tasks catalog. These pin the
/// on-disk format so a future field addition can't silently break old tasks.json.
final class TaskModelsTests: XCTestCase {

    private func roundTrip<T: Codable & Equatable>(_ value: T) throws -> T {
        let enc = JSONEncoder()
        enc.dateEncodingStrategy = .iso8601
        let data = try enc.encode(value)
        let dec = JSONDecoder()
        dec.dateDecodingStrategy = .iso8601
        return try dec.decode(T.self, from: data)
    }

    // Whole-second date so `.iso8601` (which drops sub-second precision, exactly as
    // AppState persists) round-trips exactly.
    private let fixedDate = Date(timeIntervalSince1970: 1_700_000_000)

    func testScheduledTaskRoundTripsAllTriggers() throws {
        let triggers: [TaskTrigger] = [
            .interval(seconds: 900),
            .dailyAt(hour: 8, minute: 0),
            .weekly(weekdays: [2, 4, 6], hour: 18, minute: 30),
            .cron(expression: "*/5 * * * *"),
        ]
        for trigger in triggers {
            let task = ScheduledTask(title: "T", goal: "do a thing", trigger: trigger,
                                     autonomy: .workspace, createdAt: fixedDate)
            XCTAssertEqual(try roundTrip(task), task, "trigger \(trigger) did not round-trip")
        }
    }

    func testTaskRunRoundTripsWithPendingApproval() throws {
        let pending = PendingApproval(toolCallId: "call_1",
                                      toolName: "writeFile",
                                      arguments: ["path": "out.txt", "content": "hi"],
                                      rawArguments: "{\"path\":\"out.txt\"}",
                                      reason: "needs OK",
                                      requestedAt: Date(timeIntervalSince1970: 1_700_000_000))
        let run = TaskRun(taskId: UUID(), startedAt: fixedDate, status: .needsApproval,
                          triggerReason: "scheduled", summary: nil, pendingApproval: pending)
        XCTAssertEqual(try roundTrip(run), run)
    }

    func testScheduledTaskBackfillsMissingOptionalFields() throws {
        // Simulate an older catalog with only the required fields present.
        let json = """
        {"id":"\(UUID().uuidString)","title":"Old","goal":"g",
         "trigger":{"interval":{"seconds":3600}},
         "createdAt":"2026-01-01T00:00:00Z"}
        """
        let dec = JSONDecoder()
        dec.dateDecodingStrategy = .iso8601
        let task = try dec.decode(ScheduledTask.self, from: Data(json.utf8))
        XCTAssertEqual(task.autonomy, .workspace)   // default
        XCTAssertFalse(task.useMCP)                 // default
        XCTAssertTrue(task.enabled)                 // default
        XCTAssertTrue(task.catchUpMissed)           // default
        XCTAssertNil(task.modelPath)
        XCTAssertNil(task.nextFireAt)
    }

    func testTaskPathsLayout() {
        let taskId = UUID()
        let runId = UUID()
        XCTAssertTrue(TaskPaths.catalogFile.hasSuffix("tasks/tasks.json"))
        XCTAssertTrue(TaskPaths.runsFile(taskId).hasSuffix("\(taskId.uuidString)/runs.json"))
        XCTAssertTrue(TaskPaths.transcriptFile(taskId, runId)
            .hasSuffix("\(taskId.uuidString)/\(runId.uuidString)/transcript.json"))
    }
}
