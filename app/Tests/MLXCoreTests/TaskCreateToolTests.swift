import XCTest
@testable import MLXCore

/// Unit tests for the `createTask` agent tool's pure layer: schedule-string
/// classification (`TaskScheduler.scheduleIntent`) and the task-origin field that
/// routes a finished run's result back to the Telegram chat that created it.
final class TaskCreateToolTests: XCTestCase {

    // MARK: - scheduleIntent

    func testEmptyOrNowMeansRunOnce() {
        XCTAssertEqual(TaskScheduler.scheduleIntent(nil), .once)
        XCTAssertEqual(TaskScheduler.scheduleIntent(""), .once)
        XCTAssertEqual(TaskScheduler.scheduleIntent("   "), .once)
        for word in ["now", "Now", "ONCE", "immediately", "asap", "right now", "one-off"] {
            XCTAssertEqual(TaskScheduler.scheduleIntent(word), .once, "“\(word)” should run once")
        }
    }

    func testParseableScheduleIsRecurring() {
        // A 5-field cron is unambiguously a recurring schedule.
        guard case .recurring = TaskScheduler.scheduleIntent("0 9 * * *") else {
            return XCTFail("cron expression should classify as recurring")
        }
        guard case .recurring = TaskScheduler.scheduleIntent("every day at 9am") else {
            return XCTFail("'every day at 9am' should classify as recurring")
        }
    }

    func testUnparseableScheduleIsInvalid() {
        // Provided but meaningless → invalid, so the model gets a helpful error
        // instead of the schedule being silently dropped to a one-shot.
        XCTAssertEqual(TaskScheduler.scheduleIntent("xyzzy not a schedule"), .invalid)
    }

    // MARK: - ScheduledTask.originTelegramChatId

    func testTaskOriginRoundTrips() throws {
        let task = ScheduledTask(
            title: "t", goal: "do the thing",
            trigger: .dailyAt(hour: 9, minute: 0),
            autonomy: .fullAuto, enabled: false,
            originTelegramChatId: 7123456789)   // > Int32 to pin Int64
        let decoded = try JSONDecoder().decode(
            ScheduledTask.self, from: try JSONEncoder().encode(task))
        XCTAssertEqual(decoded.originTelegramChatId, 7123456789)
        XCTAssertEqual(decoded, task)
    }

    /// A task saved before this field existed must still decode (origin → nil) —
    /// the catalog `try?`-decodes, so a throw would wipe the user's whole task list.
    func testTaskFromOldCatalogDecodesWithoutOrigin() throws {
        let task = ScheduledTask(title: "t", goal: "g", trigger: .interval(seconds: 60))
        var obj = try JSONSerialization.jsonObject(
            with: try JSONEncoder().encode(task)) as! [String: Any]
        obj.removeValue(forKey: "originTelegramChatId")
        let decoded = try JSONDecoder().decode(
            ScheduledTask.self, from: try JSONSerialization.data(withJSONObject: obj))
        XCTAssertNil(decoded.originTelegramChatId)
        XCTAssertEqual(decoded.goal, "g")
    }
}
