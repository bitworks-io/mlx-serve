import Foundation

/// Minimal 5-field cron support (`minute hour day-of-month month day-of-week`).
/// Each field is `*`, a number, a comma list, a range `a-b`, or a step `*/n`.
/// Used for the "advanced" schedule option. Pure/testable.
enum CronSchedule {

    /// Soonest time the expression matches strictly after `date` (minute resolution),
    /// or nil if the expression is invalid or nothing matches within a year.
    static func nextFire(_ expression: String, after date: Date, calendar: Calendar) -> Date? {
        guard let fields = parse(expression) else { return nil }
        // Start at the next whole minute after `date`.
        var candidate = calendar.date(bySetting: .second, value: 0, of: date) ?? date
        candidate = candidate.addingTimeInterval(60)
        // Search up to ~1 year of minutes (cap to stay bounded).
        let limit = 366 * 24 * 60
        var step = 0
        while step < limit {
            let c = calendar.dateComponents([.minute, .hour, .day, .month, .weekday], from: candidate)
            if let minute = c.minute, let hour = c.hour, let day = c.day,
               let month = c.month, let weekday = c.weekday,
               fields.minute.contains(minute), fields.hour.contains(hour),
               fields.dayOfMonth.contains(day), fields.month.contains(month),
               // cron weekday: 0/7 = Sunday; Calendar weekday: 1 = Sunday.
               fields.dayOfWeek.contains(weekday - 1) {
                return candidate
            }
            candidate = candidate.addingTimeInterval(60)
            step += 1
        }
        return nil
    }

    struct Fields: Equatable {
        var minute: Set<Int>
        var hour: Set<Int>
        var dayOfMonth: Set<Int>
        var month: Set<Int>
        var dayOfWeek: Set<Int>   // 0...6, Sunday = 0
    }

    static func parse(_ expression: String) -> Fields? {
        let parts = expression.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard parts.count == 5 else { return nil }
        guard let minute = field(parts[0], 0, 59),
              let hour = field(parts[1], 0, 23),
              let dom = field(parts[2], 1, 31),
              let month = field(parts[3], 1, 12),
              var dow = field(parts[4], 0, 7) else { return nil }
        // Normalize cron's dual Sunday (7 -> 0) so matching is uniform.
        if dow.contains(7) { dow.remove(7); dow.insert(0) }
        return Fields(minute: minute, hour: hour, dayOfMonth: dom, month: month, dayOfWeek: dow)
    }

    /// Expand one cron field into the concrete set of values it matches.
    private static func field(_ s: String, _ lo: Int, _ hi: Int) -> Set<Int>? {
        var result = Set<Int>()
        for token in s.split(separator: ",") {
            // step: */n or a-b/n or *
            let stepParts = token.split(separator: "/", maxSplits: 1).map(String.init)
            let base = stepParts[0]
            let step = stepParts.count == 2 ? Int(stepParts[1]) : 1
            guard let step, step >= 1 else { return nil }

            let rangeLo: Int, rangeHi: Int
            if base == "*" {
                rangeLo = lo; rangeHi = hi
            } else if base.contains("-") {
                let bounds = base.split(separator: "-").map { Int($0) }
                guard bounds.count == 2, let a = bounds[0], let b = bounds[1] else { return nil }
                rangeLo = a; rangeHi = b
            } else {
                guard let v = Int(base) else { return nil }
                rangeLo = v; rangeHi = v
            }
            guard rangeLo >= lo, rangeHi <= hi, rangeLo <= rangeHi else { return nil }
            var v = rangeLo
            while v <= rangeHi { result.insert(v); v += step }
        }
        return result.isEmpty ? nil : result
    }
}
