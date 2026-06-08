import XCTest
@testable import MLXCore

/// The plain-language schedule grammar (the headline creation UX). Pinned so the
/// live "✓ every day at 8:00 AM" confirmation stays correct as the grammar grows.
final class ScheduleParserTests: XCTestCase {

    func testIntervals() {
        XCTAssertEqual(ScheduleParser.parse("every 15 minutes"), .interval(seconds: 900))
        XCTAssertEqual(ScheduleParser.parse("every 5 min"), .interval(seconds: 300))
        XCTAssertEqual(ScheduleParser.parse("every 2 hours"), .interval(seconds: 7200))
        XCTAssertEqual(ScheduleParser.parse("hourly"), .interval(seconds: 3600))
        XCTAssertEqual(ScheduleParser.parse("every hour"), .interval(seconds: 3600))
    }

    func testDaily() {
        XCTAssertEqual(ScheduleParser.parse("daily at 8am"), .dailyAt(hour: 8, minute: 0))
        XCTAssertEqual(ScheduleParser.parse("every day at 8:30am"), .dailyAt(hour: 8, minute: 30))
        XCTAssertEqual(ScheduleParser.parse("at 6pm"), .dailyAt(hour: 18, minute: 0))
        XCTAssertEqual(ScheduleParser.parse("daily at 18:00"), .dailyAt(hour: 18, minute: 0))
        XCTAssertEqual(ScheduleParser.parse("at 12am"), .dailyAt(hour: 0, minute: 0))   // midnight
        XCTAssertEqual(ScheduleParser.parse("at 12pm"), .dailyAt(hour: 12, minute: 0))  // noon
    }

    func testWeekly() {
        XCTAssertEqual(ScheduleParser.parse("every weekday at 8am"),
                       .weekly(weekdays: [2, 3, 4, 5, 6], hour: 8, minute: 0))
        XCTAssertEqual(ScheduleParser.parse("every monday and wednesday at 18:00"),
                       .weekly(weekdays: [2, 4], hour: 18, minute: 0))
        XCTAssertEqual(ScheduleParser.parse("weekends at 10am"),
                       .weekly(weekdays: [1, 7], hour: 10, minute: 0))
    }

    func testCron() {
        XCTAssertEqual(ScheduleParser.parse("*/5 * * * *"), .cron(expression: "*/5 * * * *"))
        XCTAssertEqual(ScheduleParser.parse("0 9 * * 1-5"), .cron(expression: "0 9 * * 1-5"))
    }

    func testGarbageReturnsNil() {
        XCTAssertNil(ScheduleParser.parse(""))
        XCTAssertNil(ScheduleParser.parse("sometime soon"))
        XCTAssertNil(ScheduleParser.parse("when I feel like it"))
    }

    func testDescribe() {
        XCTAssertEqual(ScheduleParser.describe(.interval(seconds: 900)), "every 15 minutes")
        XCTAssertEqual(ScheduleParser.describe(.interval(seconds: 3600)), "every hour")
        XCTAssertEqual(ScheduleParser.describe(.interval(seconds: 7200)), "every 2 hours")
        XCTAssertEqual(ScheduleParser.describe(.dailyAt(hour: 8, minute: 0)), "every day at 8:00 AM")
        XCTAssertEqual(ScheduleParser.describe(.dailyAt(hour: 18, minute: 30)), "every day at 6:30 PM")
        XCTAssertEqual(ScheduleParser.describe(.weekly(weekdays: [2, 3, 4, 5, 6], hour: 8, minute: 0)),
                       "every weekday at 8:00 AM")
        XCTAssertEqual(ScheduleParser.describe(.weekly(weekdays: [2, 4], hour: 18, minute: 0)),
                       "every Mon, Wed at 6:00 PM")
    }

    func testParseDescribeRoundTripStable() {
        for phrase in ["every 15 minutes", "daily at 8am", "every weekday at 8am"] {
            guard let t = ScheduleParser.parse(phrase) else { return XCTFail("parse \(phrase)") }
            // describe(parse(x)) should itself re-parse to the same trigger.
            XCTAssertEqual(ScheduleParser.parse(ScheduleParser.describe(t)), t, "unstable for \(phrase)")
        }
    }
}
