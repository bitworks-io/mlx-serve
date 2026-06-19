import Foundation

/// A parsed Telegram `Update`. Carries the message text (or an attachment's
/// caption) plus an optional `attachment`. Non-message updates and messages
/// with neither text nor a usable attachment are dropped during parse.
struct TelegramUpdate: Equatable {
    let updateId: Int64
    let chatId: Int64
    /// Message text, or the caption when the message is a photo/voice/document.
    /// May be empty for an attachment sent with no caption.
    let text: String
    /// Best-effort display name for the chat (sender first name / username),
    /// used to title the hidden session and for log lines.
    let senderName: String
    /// A media attachment on the message, if any. `nil` for a plain text turn.
    let attachment: Attachment?

    /// The media kinds the bridge knows how to act on. Telegram delivers each
    /// as a distinct field on `message`; we normalise them here so routing and
    /// download work off one value. `fileId` is fed to `getFile`.
    enum Attachment: Equatable {
        /// A compressed photo (`message.photo`) — always JPEG. `fileId` is the
        /// largest available size.
        case photo(fileId: String)
        /// A voice note (`message.voice`) — always Ogg/Opus.
        case voice(fileId: String)
        /// A music/audio file (`message.audio`) — usually mp3/m4a, occasionally ogg.
        case audio(fileId: String)
        /// Any other uploaded file (`message.document`): could be an image
        /// (`image/*`), audio (`audio/*`), or something we can't use.
        case document(fileId: String, mimeType: String, fileName: String)
    }

    init(updateId: Int64, chatId: Int64, text: String, senderName: String,
         attachment: Attachment? = nil) {
        self.updateId = updateId
        self.chatId = chatId
        self.text = text
        self.senderName = senderName
        self.attachment = attachment
    }
}

/// What the bridge should do with an incoming update, decided purely from the
/// attachment kind and the loaded model's capabilities. Keeping this a pure
/// function (no network, no `AppState`) lets every branch be unit-tested — the
/// live I/O in `TelegramBridge` is then a thin switch over the result.
enum TelegramAttachmentAction: Equatable {
    /// No (usable) attachment — handle `update.text` as a normal turn.
    case textOnly
    /// Forward the image to a vision-capable model. `fileId` to download.
    case image(fileId: String)
    /// An image arrived but the model can't see — reply that it's not possible.
    case imageUnsupported
    /// Audio (voice / music): `transcribe == false` feeds the decoded PCM to an
    /// audio-capable model; `true` transcribes on-device and sends the text so
    /// voice still works on a text/vision-only model.
    case audio(fileId: String, transcribe: Bool)
    /// A document we can't use (e.g. a PDF/zip) — reply with `reason`.
    case unsupported(reason: String)
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

    /// `sendChatAction` powers the "typing…" indicator while the model works.
    /// The action shown to the user expires after ~5 s, so the bridge re-posts
    /// it on a short timer for the whole download/decode/generate span.
    static func sendChatActionURL(token: String) -> URL? {
        URL(string: "\(apiBase)/bot\(token)/sendChatAction")
    }

    /// `getFile` resolves a `file_id` to a temporary `file_path` for download.
    static func getFileURL(token: String, fileId: String) -> URL? {
        guard var c = URLComponents(string: "\(apiBase)/bot\(token)/getFile") else { return nil }
        c.queryItems = [URLQueryItem(name: "file_id", value: fileId)]
        return c.url
    }

    /// The actual bytes live at a separate `/file/bot<token>/<file_path>` host
    /// path (not under `/bot<token>/`). `file_path` is server-provided and may
    /// contain `/`, so it's appended raw.
    static func fileDownloadURL(token: String, filePath: String) -> URL? {
        URL(string: "\(apiBase)/file/bot\(token)/\(filePath)")
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

    /// JSON body for `sendChatAction`. `action` is a Telegram action string —
    /// we only ever send `"typing"`.
    static func sendChatActionBody(chatId: Int64, action: String = "typing") -> Data {
        let obj: [String: Any] = ["chat_id": chatId, "action": action]
        return (try? JSONSerialization.data(withJSONObject: obj)) ?? Data("{}".utf8)
    }

    // MARK: - Attachment routing (pure)

    /// Decide what to do with an update given the loaded model's capabilities.
    /// Pure so every capability × attachment combination is unit-tested; the
    /// bridge just switches over the result and does the I/O.
    static func decideAttachmentAction(_ update: TelegramUpdate,
                                       supportsVision: Bool,
                                       supportsAudio: Bool) -> TelegramAttachmentAction {
        guard let attachment = update.attachment else { return .textOnly }
        switch attachment {
        case .photo(let fileId):
            return supportsVision ? .image(fileId: fileId) : .imageUnsupported
        case .voice(let fileId), .audio(let fileId):
            return .audio(fileId: fileId, transcribe: !supportsAudio)
        case .document(let fileId, let mimeType, let fileName):
            let mime = mimeType.lowercased()
            if mime.hasPrefix("image/") {
                return supportsVision ? .image(fileId: fileId) : .imageUnsupported
            }
            if mime.hasPrefix("audio/") {
                return .audio(fileId: fileId, transcribe: !supportsAudio)
            }
            let label = fileName.isEmpty ? "that file type" : fileName
            return .unsupported(reason: "📎 I can only handle images and audio right now (got \(label)).")
        }
    }

    // MARK: - Response parsing

    /// Parse a `getUpdates` response into its updates (in order) and the next
    /// poll offset (`max(update_id) + 1`). `nextOffset` is nil when the result
    /// is empty so the caller keeps its current offset. A turn is produced for
    /// a message with text OR a usable attachment (photo / voice / audio /
    /// document); the caption rides along as `text`. Messages with neither
    /// (stickers, service messages) are skipped but STILL advance the offset,
    /// so they can't wedge the poll loop by being redelivered forever.
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
                  let chatId = int64(chat["id"]) else { continue }
            let attachment = parseAttachment(message)
            // `text` for a plain message; `caption` when an attachment carries one.
            let text = (message["text"] as? String) ?? (message["caption"] as? String) ?? ""
            // Nothing actionable (e.g. a sticker) → skip, but the offset above
            // already advanced past it.
            guard !text.isEmpty || attachment != nil else { continue }
            let from = message["from"] as? [String: Any]
            let name = (from?["first_name"] as? String)
                ?? (from?["username"] as? String)
                ?? (chat["first_name"] as? String)
                ?? "user"
            updates.append(TelegramUpdate(updateId: updateId, chatId: chatId, text: text,
                                          senderName: name, attachment: attachment))
        }
        return (updates, maxId.map { $0 + 1 })
    }

    /// Pull the first usable media attachment off a `message` object. Telegram
    /// puts each kind in its own field; photos arrive as an array of sizes
    /// (ascending), so we pick the largest.
    private static func parseAttachment(_ message: [String: Any]) -> TelegramUpdate.Attachment? {
        if let photo = message["photo"] as? [[String: Any]], !photo.isEmpty {
            let largest = photo.max { photoArea($0) < photoArea($1) } ?? photo.last
            if let fileId = largest?["file_id"] as? String { return .photo(fileId: fileId) }
        }
        if let voice = message["voice"] as? [String: Any],
           let fileId = voice["file_id"] as? String { return .voice(fileId: fileId) }
        if let audio = message["audio"] as? [String: Any],
           let fileId = audio["file_id"] as? String { return .audio(fileId: fileId) }
        if let doc = message["document"] as? [String: Any],
           let fileId = doc["file_id"] as? String {
            let mime = (doc["mime_type"] as? String) ?? ""
            let name = (doc["file_name"] as? String) ?? ""
            return .document(fileId: fileId, mimeType: mime, fileName: name)
        }
        return nil
    }

    /// A photo size's pixel area (Telegram orders sizes ascending; we don't
    /// rely on the order). Falls back to `file_size` when dimensions are absent.
    private static func photoArea(_ size: [String: Any]) -> Int {
        let w = (size["width"] as? NSNumber)?.intValue ?? 0
        let h = (size["height"] as? NSNumber)?.intValue ?? 0
        let area = w * h
        return area > 0 ? area : ((size["file_size"] as? NSNumber)?.intValue ?? 0)
    }

    /// Parse a `getFile` response → the temporary `file_path` for download, or
    /// nil on failure.
    static func parseFilePath(_ data: Data) -> String? {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              (json["ok"] as? Bool) == true,
              let result = json["result"] as? [String: Any] else { return nil }
        return result["file_path"] as? String
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
