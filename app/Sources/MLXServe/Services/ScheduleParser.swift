import Foundation

/// Turns plain-language schedule phrases ("every weekday at 8am") into a concrete
/// `TaskTrigger`, and renders a trigger back to a human confirmation string. Pure
/// and deterministic so the creation UX can echo "✓ every day at 8:00 AM" live and
/// so the whole grammar is unit-testable. Phase 1 covers the common forms; the full
/// grammar + cron next-fire are Phase 4.
enum ScheduleParser {

    // MARK: - Describe (trigger -> confirmation string)

    static func describe(_ trigger: TaskTrigger) -> String {
        switch trigger {
        case .interval(let seconds):
            if seconds % 3600 == 0 {
                let h = seconds / 3600
                return h == 1 ? "every hour" : "every \(h) hours"
            } else if seconds % 60 == 0 {
                let m = seconds / 60
                return m == 1 ? "every minute" : "every \(m) minutes"
            }
            return "every \(seconds) seconds"
        case .dailyAt(let h, let m):
            return "every day at \(timeString(h, m))"
        case .weekly(let days, let h, let m):
            return "every \(weekdayList(days)) at \(timeString(h, m))"
        case .cron(let expr):
            return "cron: \(expr)"
        }
    }

    // MARK: - Parse (text -> trigger)

    static func parse(_ raw: String) -> TaskTrigger? {
        let text = raw.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }

        // Raw cron: 5 whitespace-separated fields of cron-legal chars.
        let cronFields = text.split(separator: " ")
        if cronFields.count == 5,
           cronFields.allSatisfy({ $0.allSatisfy { "0123456789*/,-".contains($0) } }) {
            return .cron(expression: text)
        }

        // Intervals.
        if text == "hourly" { return .interval(seconds: 3600) }
        if text == "every minute" { return .interval(seconds: 60) }
        if text == "every hour" { return .interval(seconds: 3600) }
        if let n = matchInt(text, #"every\s+(\d+)\s*(?:minutes?|mins?|m)\b"#) {
            return .interval(seconds: n * 60)
        }
        if let n = matchInt(text, #"every\s+(\d+)\s*(?:hours?|hrs?|h)\b"#) {
            return .interval(seconds: n * 3600)
        }

        let time = parseTime(in: text) ?? (9, 0)

        // Weekday sets.
        if text.contains("weekday") {
            return .weekly(weekdays: [2, 3, 4, 5, 6], hour: time.0, minute: time.1)
        }
        if text.contains("weekend") {
            return .weekly(weekdays: [1, 7], hour: time.0, minute: time.1)
        }
        let named = namedWeekdays(in: text)
        if !named.isEmpty {
            return .weekly(weekdays: named, hour: time.0, minute: time.1)
        }

        // Daily.
        if text.contains("daily") || text.contains("every day") || text.hasPrefix("at ") || parseTime(in: text) != nil {
            return .dailyAt(hour: time.0, minute: time.1)
        }
        return nil
    }

    // MARK: - Helpers

    static func timeString(_ hour: Int, _ minute: Int) -> String {
        let period = hour < 12 ? "AM" : "PM"
        var h12 = hour % 12
        if h12 == 0 { h12 = 12 }
        return String(format: "%d:%02d %@", h12, minute, period)
    }

    /// Calendar weekday (1=Sun…7=Sat) -> short label list, in week order.
    static func weekdayList(_ days: Set<Int>) -> String {
        if days == [2, 3, 4, 5, 6] { return "weekday" }
        if days == [1, 7] { return "weekend" }
        let names = ["", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        return days.sorted().compactMap { (1...7).contains($0) ? names[$0] : nil }.joined(separator: ", ")
    }

    private static func parseTime(in text: String) -> (Int, Int)? {
        // 8am, 8:30am, 8 am, 18:00, 08:00 — optionally preceded by "at".
        let pattern = #"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"#
        guard let re = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        // Anchor on "at <time>" first; otherwise take the first time-looking token.
        let searchRange = NSRange(location: 0, length: ns.length)
        let matches = re.matches(in: text, range: searchRange)
        for m in matches {
            guard let hr = Int(group(m, 1, ns) ?? "") else { continue }
            let minStr = group(m, 2, ns)
            let ampm = group(m, 3, ns)
            // Skip bare integers that are part of "every 5 minutes" (handled earlier);
            // require either a colon, an am/pm, or an explicit "at" before it.
            let hasColon = minStr != nil
            let hasPeriod = ampm != nil
            let prefixIsAt = m.range.location >= 3 &&
                ns.substring(with: NSRange(location: max(0, m.range.location - 3), length: 3)).contains("at")
            guard hasColon || hasPeriod || prefixIsAt else { continue }
            var h = hr
            let mins = Int(minStr ?? "0") ?? 0
            if let ampm {
                if ampm == "pm" && h < 12 { h += 12 }
                if ampm == "am" && h == 12 { h = 0 }
            }
            guard (0...23).contains(h), (0...59).contains(mins) else { continue }
            return (h, mins)
        }
        return nil
    }

    private static func namedWeekdays(in text: String) -> Set<Int> {
        let map: [(String, Int)] = [
            ("sunday", 1), ("sun", 1), ("monday", 2), ("mon", 2), ("tuesday", 3), ("tue", 3),
            ("wednesday", 4), ("wed", 4), ("thursday", 5), ("thu", 5),
            ("friday", 6), ("fri", 6), ("saturday", 7), ("sat", 7),
        ]
        var result = Set<Int>()
        for (name, n) in map where text.contains(name) { result.insert(n) }
        return result
    }

    private static func matchInt(_ text: String, _ pattern: String) -> Int? {
        guard let re = try? NSRegularExpression(pattern: pattern) else { return nil }
        let ns = text as NSString
        guard let m = re.firstMatch(in: text, range: NSRange(location: 0, length: ns.length)),
              let s = group(m, 1, ns) else { return nil }
        return Int(s)
    }

    private static func group(_ m: NSTextCheckingResult, _ i: Int, _ ns: NSString) -> String? {
        guard i < m.numberOfRanges else { return nil }
        let r = m.range(at: i)
        guard r.location != NSNotFound else { return nil }
        return ns.substring(with: r)
    }
}
