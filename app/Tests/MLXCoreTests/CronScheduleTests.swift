import XCTest
@testable import MLXCore

/// 5-field cron parsing + next-fire. Pinned against a fixed UTC calendar.
final class CronScheduleTests: XCTestCase {

    private var cal: Calendar = {
        var c = Calendar(identifier: .gregorian)
        c.timeZone = TimeZone(identifier: "UTC")!
        return c
    }()
    private let t0 = Date(timeIntervalSince1970: 1_767_265_200) // 2026-01-01T11:00:00Z (Thursday)

    func testParseStarStep() {
        let f = CronSchedule.parse("*/15 * * * *")
        XCTAssertEqual(f?.minute, [0, 15, 30, 45])
        XCTAssertEqual(f?.hour.count, 24)
    }

    func testParseRangeAndList() {
        let f = CronSchedule.parse("0 9 * * 1-5")
        XCTAssertEqual(f?.minute, [0])
        XCTAssertEqual(f?.hour, [9])
        XCTAssertEqual(f?.dayOfWeek, [1, 2, 3, 4, 5]) // Mon-Fri (cron 1-5)
    }

    func testParseSundayNormalization() {
        // cron allows 0 and 7 for Sunday; both normalize to 0.
        XCTAssertEqual(CronSchedule.parse("0 0 * * 7")?.dayOfWeek, [0])
        XCTAssertEqual(CronSchedule.parse("0 0 * * 0")?.dayOfWeek, [0])
    }

    func testParseRejectsBadField() {
        XCTAssertNil(CronSchedule.parse("0 9 * *"))        // too few fields
        XCTAssertNil(CronSchedule.parse("99 * * * *"))     // out of range
        XCTAssertNil(CronSchedule.parse("a * * * *"))      // non-numeric
    }

    func testNextFireEveryFifteen() {
        let next = CronSchedule.nextFire("*/15 * * * *", after: t0, calendar: cal)
        guard let next else { return XCTFail("nil") }
        XCTAssertGreaterThan(next, t0)
        let c = cal.dateComponents([.minute], from: next)
        XCTAssertEqual(c.minute! % 15, 0)
        XCTAssertLessThanOrEqual(next.timeIntervalSince(t0), 15 * 60)
    }

    func testNextFireDailyNineAM() {
        let next = CronSchedule.nextFire("0 9 * * *", after: t0, calendar: cal)
        guard let next else { return XCTFail("nil") }
        let c = cal.dateComponents([.hour, .minute], from: next)
        XCTAssertEqual(c.hour, 9); XCTAssertEqual(c.minute, 0)
        XCTAssertGreaterThan(next, t0) // t0 is 11:00, so next 09:00 is tomorrow
    }

    func testNextFireWeekdaysOnly() {
        // 0 9 * * 1-5 after a Thursday 11:00 -> Friday 09:00, never Saturday/Sunday.
        let next = CronSchedule.nextFire("0 9 * * 1-5", after: t0, calendar: cal)
        guard let next else { return XCTFail("nil") }
        let wd = cal.dateComponents([.weekday], from: next).weekday!
        XCTAssertTrue((2...6).contains(wd)) // Calendar Mon-Fri = 2...6
    }

    func testInvalidExpressionNextFireNil() {
        XCTAssertNil(CronSchedule.nextFire("nonsense", after: t0, calendar: cal))
    }
}
