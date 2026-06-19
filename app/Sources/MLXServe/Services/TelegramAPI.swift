import Foundation

/// A parsed Telegram `Update` that carries a text message. Non-message and
/// non-text updates are dropped during parse — the bridge only acts on text.
struct TelegramUpdate: Equatable {
    let updateId: Int64
    let chatId: Int64
    let text: String
    /// Best-effort display name for the chat (sender first name / username),
    /// used to title the hidden session and for log lines.
    let senderName: String
}

/// Stateless Telegram Bot API helpers: URL / request-body construction and
/// response parsing. Everything here is pure and unit-tested without a network;
/// the live I/O lives in `TelegramBridge`. Keeping this layer pure mirrors the
/// project's "factor a testable helper out of the untestable surface" rule.
enum TelegramAPI {
    static let apiBase = "https://api.telegram.org"

    /// Telegram's per-message character cap is 4096 UTF-16 code units. We chunk
    /// to this exact limit (no artificial margin — `splitForTelegram` measures
    /// UTF-16 precisely).
    static let messageLimit = 4096

    // MARK: - URLs

    /// `getUpdates` long-poll URL. `offset` acknowledges every update with a
    /// lower id (so they're not redelivered); `timeout` is the server-side
    /// long-poll hold in seconds. We request only `message` updates to keep the
    /// payload small and ignore edits / callback queries.
    static func getUpdatesURL(token: String, offset: Int64, timeout: Int) -> URL? {
        guard var c = URLComponents(string: "\(apiBase)/bot\(token)/getUpdates") else { return nil }
        c.queryItems = [
            URLQueryItem(name: "offset", value: "\(offset)"),
            URLQueryItem(name: "timeout", value: "\(timeout)"),
            URLQueryItem(name: "allowed_updates", value: "[\"message\"]"),
        ]
        return c.url
    }

    static func getMeURL(token: String) -> URL? {
        URL(string: "\(apiBase)/bot\(token)/getMe")
    }

    static func sendMessageURL(token: String) -> URL? {
        URL(string: "\(apiBase)/bot\(token)/sendMessage")
    }

    // MARK: - Request bodies

    /// JSON body for `sendMessage`. Built with `JSONSerialization` so arbitrary
    /// reply text (quotes, control bytes, emoji) is correctly escaped — the same
    /// discipline as the server-side hand-rolled JSON, but here the std lib owns
    /// the escaping.
    static func sendMessageBody(chatId: Int64, text: String) -> Data {
        let obj: [String: Any] = ["chat_id": chatId, "text": text]
        return (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
    }

    // MARK: - Response parsing

    /// Parse a `getUpdates` response into its text updates (in order) and the
    /// next poll offset (`max(update_id) + 1`). `nextOffset` is nil when the
    /// result is empty so the caller keeps its current offset. Non-text and
    /// non-message updates are skipped but STILL advance the offset, so a photo
    /// or sticker can't wedge the poll loop by being redelivered forever.
    static func parseUpdates(_ data: Data) -> (updates: [TelegramUpdate], nextOffset: Int64?) {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              (json["ok"] as? Bool) == true,
              let result = json["result"] as? [[String: Any]] else {
            return ([], nil)
        }
        var updates: [TelegramUpdate] = []
        var maxId: Int64?
        for item in result {
            guard let updateId = int64(item["update_id"]) else { continue }
            maxId = Swift.max(maxId ?? updateId, updateId)
            guard let message = item["message"] as? [String: Any],
                  let chat = message["chat"] as? [String: Any],
                  let chatId = int64(chat["id"]),
                  let text = message["text"] as? String else { continue }
            let from = message["from"] as? [String: Any]
            let name = (from?["first_name"] as? String)
                ?? (from?["username"] as? String)
                ?? (chat["first_name"] as? String)
                ?? "user"
            updates.append(TelegramUpdate(updateId: updateId, chatId: chatId, text: text, senderName: name))
        }
        return (updates, maxId.map { $0 + 1 })
    }

    /// Parse a `getMe` response → the bot's @username (without the leading `@`),
    /// or nil on failure. Used only to show "Connected as @bot" in Settings.
    static func parseUsername(_ data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              (json["ok"] as? Bool) == true,
              let result = json["result"] as? [String: Any] else { return nil }
        return result["username"] as? String
    }

    /// Telegram chat / update ids exceed Int32, and `JSONSerialization` hands
    /// numbers back as `NSNumber`, so always read them as Int64.
    private static func int64(_ any: Any?) -> Int64? {
        if let n = any as? NSNumber { return n.int64Value }
        if let i = any as? Int64 { return i }
        if let i = any as? Int { return Int64(i) }
        return nil
    }

    // MARK: - Message chunking

    /// Split a reply into chunks that each fit Telegram's per-message limit
    /// (measured in UTF-16 code units, so a wall of emoji can't overflow). The
    /// split prefers the last newline, then the last space, before the cut so we
    /// don't slice mid-word; a single oversized run is hard-split. The chunks
    /// concatenate back to the exact input (no content dropped). Empty input
    /// yields one empty chunk so the caller always has something to send.
    static func splitForTelegram(_ text: String, limit: Int = messageLimit) -> [String] {
        guard limit > 0 else { return [text] }
        if text.isEmpty { return [""] }
        var chunks: [String] = []
        var remaining = Substring(text)
        while !remaining.isEmpty {
            if remaining.utf16.count <= limit {
                chunks.append(String(remaining))
                break
            }
            // Walk character boundaries until the next char would exceed `limit`.
            var cut = remaining.startIndex
            var width = 0
            var idx = remaining.startIndex
            while idx < remaining.endIndex {
                let w = remaining[idx].utf16.count   // 1 or 2
                if width + w > limit { break }
                width += w
                idx = remaining.index(after: idx)
                cut = idx
            }
            // Guarantee progress when even the first character is wider than
            // `limit` (only reachable with a pathologically small test limit).
            if cut == remaining.startIndex {
                cut = remaining.index(after: remaining.startIndex)
            }
            // Prefer to break on whitespace inside the [start, cut) window.
            let window = remaining[remaining.startIndex..<cut]
            var breakAt = cut
            if let nl = window.lastIndex(of: "\n") {
                breakAt = remaining.index(after: nl)
            } else if let sp = window.lastIndex(of: " ") {
                breakAt = remaining.index(after: sp)
            }
            if breakAt <= remaining.startIndex { breakAt = cut }
            chunks.append(String(remaining[remaining.startIndex..<breakAt]))
            remaining = remaining[breakAt...]
        }
        return chunks
    }
}
