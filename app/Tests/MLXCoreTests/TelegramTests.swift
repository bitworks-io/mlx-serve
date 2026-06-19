import XCTest
@testable import MLXCore

/// Unit tests for the Telegram bot bridge's pure layer: config / access logic
/// (`ServerOptions.TelegramConfig`) and the stateless API helpers
/// (`TelegramAPI`). The live long-poll loop in `TelegramBridge` is thin glue
/// over these — all the logic that can be wrong without a network lives here.
final class TelegramTests: XCTestCase {

    // MARK: - TelegramConfig

    func testTelegramDefaultsAreOff() {
        let c = ServerOptions.TelegramConfig()
        XCTAssertFalse(c.enabled)
        XCTAssertFalse(c.agentMode, "agent mode (shell/file tools from the phone) must be opt-in")
        XCTAssertFalse(c.useMCP, "MCP exposure must be opt-in")
        XCTAssertTrue(c.botToken.isEmpty)
        XCTAssertTrue(c.allowedChatIds.isEmpty)
        XCTAssertFalse(c.isRunnable, "no token + disabled is not runnable")
    }

    func testIsRunnableNeedsEnabledAndToken() {
        var c = ServerOptions.TelegramConfig()
        c.enabled = true
        XCTAssertFalse(c.isRunnable, "enabled but blank token is not runnable")
        c.botToken = "   "
        XCTAssertFalse(c.isRunnable, "whitespace-only token is not runnable")
        c.botToken = "123:abc"
        XCTAssertTrue(c.isRunnable)
        c.enabled = false
        XCTAssertFalse(c.isRunnable, "token but disabled is not runnable")
    }

    func testTrimmedTokenStripsWhitespace() {
        var c = ServerOptions.TelegramConfig()
        c.botToken = "  123:abc\n"
        XCTAssertEqual(c.trimmedToken, "123:abc")
    }

    /// Access gate is the bot's only protection against a stranger driving the
    /// local model/agent, so its three branches are pinned exactly.
    func testAccessAdoptsFirstChatThenLocks() {
        var c = ServerOptions.TelegramConfig()
        // Empty allow-list → first chat is adopted (trust-on-first-use).
        XCTAssertEqual(c.access(forChatId: 555), .adopt)

        // Once an owner is on the list, that owner is allowed…
        c.allowedChatIds = [555]
        XCTAssertEqual(c.access(forChatId: 555), .allowed)
        // …and everyone else is rejected.
        XCTAssertEqual(c.access(forChatId: 999), .rejected)
    }

    func testTelegramConfigRoundTripsThroughServerOptions() throws {
        var opts = ServerOptions()
        opts.telegram.enabled = true
        opts.telegram.botToken = "8864:TESTONLY"
        opts.telegram.agentMode = true
        opts.telegram.useMCP = true
        opts.telegram.enableThinking = true
        opts.telegram.allowedChatIds = [7123456789, 42]  // > Int32 to pin Int64 storage

        let data = try JSONEncoder().encode(opts)
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: data)
        XCTAssertEqual(opts, decoded)
        XCTAssertEqual(decoded.telegram.allowedChatIds, [7123456789, 42])
        XCTAssertTrue(decoded.telegram.useMCP)
    }

    /// MIGRATION GUARD (nested): a telegram blob written before `useMCP` existed
    /// must decode with useMCP=false rather than throwing — a throw inside the
    /// nested TelegramConfig propagates up and resets the WHOLE ServerOptions
    /// (token included). This is the exact shape already on disk for early users.
    func testTelegramConfigDecodesWhenNestedFieldMissing() throws {
        var opts = ServerOptions()
        opts.telegram.enabled = true
        opts.telegram.botToken = "8864:TESTONLY"
        opts.telegram.agentMode = true
        var full = try JSONSerialization.jsonObject(with: try JSONEncoder().encode(opts)) as! [String: Any]
        var tg = full["telegram"] as! [String: Any]
        tg.removeValue(forKey: "useMCP")   // pre-useMCP telegram blob
        full["telegram"] = tg
        let data = try JSONSerialization.data(withJSONObject: full)

        let decoded = try JSONDecoder().decode(ServerOptions.self, from: data)
        XCTAssertTrue(decoded.telegram.enabled, "token + flags must survive")
        XCTAssertEqual(decoded.telegram.botToken, "8864:TESTONLY")
        XCTAssertTrue(decoded.telegram.agentMode)
        XCTAssertFalse(decoded.telegram.useMCP, "missing nested key → safe default")
    }

    /// MIGRATION GUARD: an existing user's saved `serverOptions` blob predates
    /// the `telegram` field. Decoding it under the new struct must SUCCEED with
    /// `telegram` defaulting to off — otherwise `ServerOptions.load()` silently
    /// resets every upgrading user's tuning to defaults (it `try?`s the decode
    /// and falls back to `ServerOptions()`). Simulate the old blob by encoding a
    /// current options object and stripping the `telegram` key.
    func testDecodesOldBlobWithoutTelegramKey() throws {
        var opts = ServerOptions()
        opts.port = 9876
        opts.defaultTemperature = 0.55
        let full = try JSONSerialization.jsonObject(with: try JSONEncoder().encode(opts)) as! [String: Any]
        XCTAssertNotNil(full["telegram"], "sanity: new encoder emits the telegram key")
        var old = full
        // Strip a representative spread of keys added across releases — proving
        // the fix is a class guard, not a one-off telegram patch.
        for key in ["telegram", "tokenizeCacheEntries", "skipMemPreflight", "llamaKvQuant"] {
            old.removeValue(forKey: key)
        }
        let oldData = try JSONSerialization.data(withJSONObject: old)

        let decoded = try JSONDecoder().decode(ServerOptions.self, from: oldData)
        // Preserved keys keep their stored values…
        XCTAssertEqual(decoded.port, 9876, "existing tuning must survive the upgrade")
        XCTAssertEqual(decoded.defaultTemperature, 0.55, accuracy: 0.0001)
        // …and every stripped key falls back to its default instead of throwing.
        XCTAssertFalse(decoded.telegram.enabled, "missing telegram key → safe default (off)")
        XCTAssertTrue(decoded.telegram.botToken.isEmpty)
        XCTAssertEqual(decoded.tokenizeCacheEntries, ServerOptions().tokenizeCacheEntries)
        XCTAssertEqual(decoded.skipMemPreflight, ServerOptions().skipMemPreflight)
        XCTAssertEqual(decoded.llamaKvQuant, ServerOptions().llamaKvQuant)
    }

    /// A blob that is just `{}` (or wholly unknown) must still decode to a
    /// full-defaults options object — the ultimate degenerate upgrade case.
    func testDecodesEmptyObjectToDefaults() throws {
        let decoded = try JSONDecoder().decode(ServerOptions.self, from: Data("{}".utf8))
        XCTAssertEqual(decoded, ServerOptions())
    }

    /// The bridge runs inside the app — flipping it must NOT prompt a server
    /// restart, so telegram fields are excluded from `serverLaunchEquals`.
    func testTelegramChangesDoNotTriggerServerRestart() {
        let a = ServerOptions()
        var b = ServerOptions()
        b.telegram.enabled = true
        b.telegram.botToken = "123:abc"
        b.telegram.agentMode = true
        b.telegram.allowedChatIds = [1]
        XCTAssertTrue(a.serverLaunchEquals(b),
                      "Telegram bridge config is app-level — it must not require a server restart")
    }

    // MARK: - Sidebar visibility

    /// Telegram bridge sessions are now shown in the chat sidebar as read-only
    /// mirrors; transient task-run vehicles stay hidden.
    func testSidebarShowsTelegramSessionsButHidesTaskRuns() {
        let normal = ChatSession(title: "normal chat")
        var telegram = ChatSession(title: "Dave (Telegram)")
        telegram.isExternalBridge = true
        var taskRun = ChatSession(title: "task vehicle")
        taskRun.taskRunId = UUID()

        let titles = AppState.sidebarSessions(from: [normal, telegram, taskRun]).map(\.title)
        XCTAssertTrue(titles.contains("normal chat"))
        XCTAssertTrue(titles.contains("Dave (Telegram)"), "Telegram mirrors must appear in the sidebar")
        XCTAssertFalse(titles.contains("task vehicle"), "task-run vehicles stay hidden")
    }

    // MARK: - TelegramAPI URLs

    func testGetUpdatesURLCarriesOffsetTimeoutAndMessageFilter() throws {
        let url = try XCTUnwrap(TelegramAPI.getUpdatesURL(token: "TOK", offset: 17, timeout: 25))
        let s = url.absoluteString
        XCTAssertTrue(s.hasPrefix("https://api.telegram.org/botTOK/getUpdates"), s)
        XCTAssertTrue(s.contains("offset=17"), s)
        XCTAssertTrue(s.contains("timeout=25"), s)
        // allowed_updates=["message"] — percent-encoded in the query.
        XCTAssertTrue(s.contains("allowed_updates="), s)
        let comps = try XCTUnwrap(URLComponents(url: url, resolvingAgainstBaseURL: false))
        let allowed = comps.queryItems?.first { $0.name == "allowed_updates" }?.value
        XCTAssertEqual(allowed, "[\"message\"]", "filter must decode back to the message-only array")
    }

    func testSendMessageAndGetMeURLs() throws {
        XCTAssertEqual(TelegramAPI.getMeURL(token: "TOK")?.absoluteString,
                       "https://api.telegram.org/botTOK/getMe")
        XCTAssertEqual(TelegramAPI.sendMessageURL(token: "TOK")?.absoluteString,
                       "https://api.telegram.org/botTOK/sendMessage")
    }

    // MARK: - sendMessage body

    func testSendMessageBodyEscapesControlBytesAndCarriesFields() throws {
        // A reply containing a newline, a quote and a control byte must produce
        // VALID JSON the Telegram API accepts (the std lib owns the escaping).
        let text = "line1\nsaid \"hi\"\u{0007}"
        let body = TelegramAPI.sendMessageBody(chatId: 7123456789, text: text)
        let obj = try XCTUnwrap(try JSONSerialization.jsonObject(with: body) as? [String: Any])
        XCTAssertEqual((obj["chat_id"] as? NSNumber)?.int64Value, 7123456789)
        XCTAssertEqual(obj["text"] as? String, text, "text must round-trip byte-for-byte")
    }

    // MARK: - parseUpdates

    func testParseUpdatesExtractsTextAndNextOffset() {
        let json = """
        {"ok":true,"result":[
          {"update_id":100,"message":{"message_id":1,
            "from":{"id":7123456789,"first_name":"Dave","username":"dave"},
            "chat":{"id":7123456789,"type":"private","first_name":"Dave"},
            "date":1,"text":"hello model"}}
        ]}
        """
        let (updates, nextOffset) = TelegramAPI.parseUpdates(Data(json.utf8))
        XCTAssertEqual(updates.count, 1)
        XCTAssertEqual(updates.first?.chatId, 7123456789, "ids over Int32 must survive")
        XCTAssertEqual(updates.first?.text, "hello model")
        XCTAssertEqual(updates.first?.senderName, "Dave")
        XCTAssertEqual(nextOffset, 101, "next offset is max(update_id)+1")
    }

    func testParseUpdatesSkipsNonTextButStillAdvancesOffset() {
        // A photo (no `text`) must be skipped as a turn, but the offset MUST
        // advance past it — otherwise getUpdates redelivers it forever.
        let json = """
        {"ok":true,"result":[
          {"update_id":200,"message":{"message_id":5,
            "chat":{"id":42,"type":"private"},"date":1,
            "photo":[{"file_id":"x"}]}}
        ]}
        """
        let (updates, nextOffset) = TelegramAPI.parseUpdates(Data(json.utf8))
        XCTAssertTrue(updates.isEmpty, "non-text update yields no turn")
        XCTAssertEqual(nextOffset, 201, "offset still advances so the photo isn't redelivered")
    }

    func testParseUpdatesEmptyResultKeepsOffset() {
        let (updates, nextOffset) = TelegramAPI.parseUpdates(Data(#"{"ok":true,"result":[]}"#.utf8))
        XCTAssertTrue(updates.isEmpty)
        XCTAssertNil(nextOffset, "empty long-poll must not bump the offset")
    }

    func testParseUpdatesRejectsGarbage() {
        let (updates, nextOffset) = TelegramAPI.parseUpdates(Data("not json".utf8))
        XCTAssertTrue(updates.isEmpty)
        XCTAssertNil(nextOffset)
    }

    func testParseUsername() {
        let json = #"{"ok":true,"result":{"id":1,"is_bot":true,"username":"mlxcoremini_bot"}}"#
        XCTAssertEqual(TelegramAPI.parseUsername(Data(json.utf8)), "mlxcoremini_bot")
        XCTAssertNil(TelegramAPI.parseUsername(Data(#"{"ok":false}"#.utf8)))
    }

    // MARK: - splitForTelegram

    func testSplitShortTextIsSingleChunk() {
        XCTAssertEqual(TelegramAPI.splitForTelegram("hello"), ["hello"])
    }

    func testSplitEmptyTextYieldsOneEmptyChunk() {
        XCTAssertEqual(TelegramAPI.splitForTelegram(""), [""],
                       "caller always needs at least one chunk to send")
    }

    func testSplitNeverExceedsLimitAndLosesNothing() {
        // 10k chars of mixed words/newlines, split at a small limit.
        let line = "the quick brown fox jumps over the lazy dog "
        let big = String(repeating: line, count: 300) + "\n" + String(repeating: "x", count: 500)
        let limit = 100
        let chunks = TelegramAPI.splitForTelegram(big, limit: limit)
        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            XCTAssertLessThanOrEqual(chunk.utf16.count, limit, "every chunk must fit the limit")
        }
        XCTAssertEqual(chunks.joined(), big, "chunks must reconstruct the exact input")
    }

    func testSplitPrefersNewlineBoundary() {
        let text = "alpha\nbeta\ngamma"
        // Limit chosen so "alpha\nbeta\n" (11) fits but "...gamma" (16) doesn't.
        let chunks = TelegramAPI.splitForTelegram(text, limit: 12)
        XCTAssertEqual(chunks.joined(), text)
        XCTAssertTrue(chunks[0].hasSuffix("\n"), "first chunk should break on a newline: \(chunks)")
    }

    func testSplitHandlesWideCharactersWithoutOverflowOrHang() {
        // Emoji are 2 UTF-16 units each; ensure no infinite loop and no overflow.
        let emoji = String(repeating: "😀", count: 50)   // 100 UTF-16 units
        let chunks = TelegramAPI.splitForTelegram(emoji, limit: 10)
        XCTAssertEqual(chunks.joined(), emoji)
        for chunk in chunks { XCTAssertLessThanOrEqual(chunk.utf16.count, 10) }
    }
}
