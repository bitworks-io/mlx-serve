import XCTest
@testable import MLXCore

/// Recurrence math + catch-up policy + the pause-cleanup helper. All pure statics,
/// pinned against a fixed UTC calendar so they don't drift with the test machine's
/// timezone.
final class TaskSchedulingTests: XCTestCase {

    private var cal: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        return c
    }()

    private let t0 = Date(timeIntervalSince1970: 1_767_265_200) // 2026-01-01T11:00:00Z

    // MARK: nextFire

    func testIntervalNextFire() {
        let next = TaskScheduler.nextFire(for: .interval(seconds: 900), after: t0, calendar: cal)
        XCTAssertEqual(next, t0.addingTimeInterval(900))
    }

    func testDailyNextFireHitsRightTimeInFuture() {
        let next = TaskScheduler.nextFire(for: .dailyAt(hour: 8, minute: 30), after: t0, calendar: cal)
        guard let next else { return XCTFail("nil") }
        XCTAssertGreaterThan(next, t0)
        let c = cal.dateComponents([.hour, .minute], from: next)
        XCTAssertEqual(c.hour, 8); XCTAssertEqual(c.minute, 30)
        // t0 is 11:00, so the next 08:30 is the following day.
        XCTAssertLessThan(next.timeIntervalSince(t0), 24 * 3600)
    }

    func testWeeklyPicksSoonestMatchingWeekday() {
        // Mondays(2) and Wednesdays(4) at 09:00.
        let next = TaskScheduler.nextFire(for: .weekly(weekdays: [2, 4], hour: 9, minute: 0),
                                          after: t0, calendar: cal)
        guard let next else { return XCTFail("nil") }
        let c = cal.dateComponents([.weekday, .hour, .minute], from: next)
        XCTAssertTrue([2, 4].contains(c.weekday!))
        XCTAssertEqual(c.hour, 9); XCTAssertEqual(c.minute, 0)
        XCTAssertGreaterThan(next, t0)
    }

    // MARK: catchUpDecision

    private func task(nextFire: Date?, enabled: Bool = true, catchUp: Bool = true) -> ScheduledTask {
        ScheduledTask(title: "t", goal: "g", trigger: .interval(seconds: 3600),
                      enabled: enabled, catchUpMissed: catchUp, nextFireAt: nextFire)
    }

    func testCatchUpDisabledTaskNeverRuns() {
        let d = TaskScheduler.catchUpDecision(task: task(nextFire: t0.addingTimeInterval(-100), enabled: false),
                                              now: t0, calendar: cal)
        XCTAssertFalse(d.runNow); XCTAssertNil(d.nextFire)
    }

    func testCatchUpMissedSlotRunsOnceAndAdvances() {
        let missed = t0.addingTimeInterval(-100)
        let d = TaskScheduler.catchUpDecision(task: task(nextFire: missed, catchUp: true), now: t0, calendar: cal)
        XCTAssertTrue(d.runNow)
        XCTAssertNotNil(d.nextFire)
        XCTAssertGreaterThan(d.nextFire!, t0) // advanced past now, not replaying the missed slot
    }

    func testCatchUpDisabledOnTaskSkipsRunButStillAdvances() {
        let missed = t0.addingTimeInterval(-100)
        let d = TaskScheduler.catchUpDecision(task: task(nextFire: missed, catchUp: false), now: t0, calendar: cal)
        XCTAssertFalse(d.runNow)
        XCTAssertGreaterThan(d.nextFire!, t0)
    }

    func testCatchUpFutureSlotUnchanged() {
        let future = t0.addingTimeInterval(1000)
        let d = TaskScheduler.catchUpDecision(task: task(nextFire: future), now: t0, calendar: cal)
        XCTAssertFalse(d.runNow)
        XCTAssertEqual(d.nextFire, future)
    }

    func testCatchUpNilComputesFresh() {
        let d = TaskScheduler.catchUpDecision(task: task(nextFire: nil), now: t0, calendar: cal)
        XCTAssertFalse(d.runNow)
        XCTAssertEqual(d.nextFire, t0.addingTimeInterval(3600))
    }

    // MARK: stripTrailingDenial

    func testStripTrailingDenialPair() {
        var summary = ChatMessage(role: .assistant, content: "**writeFile** → denied by user")
        summary.isAgentSummary = true
        var tool = ChatMessage(role: .system, content: "Error: user denied this tool call.")
        tool.toolCallId = "call_1"
        let msgs = [
            ChatMessage(role: .user, content: "do it"),
            ChatMessage(role: .assistant, content: "I'll write the file."),
            summary, tool,
        ]
        let cleaned = TaskScheduler.stripTrailingDenial(msgs, toolCallId: "call_1")
        XCTAssertEqual(cleaned.count, 2)
        XCTAssertEqual(cleaned.last?.content, "I'll write the file.")
    }

    func testStripLeavesCleanTranscript() {
        let msgs = [ChatMessage(role: .user, content: "hi"),
                    ChatMessage(role: .assistant, content: "done")]
        XCTAssertEqual(TaskScheduler.stripTrailingDenial(msgs, toolCallId: "x").count, 2)
    }
}
