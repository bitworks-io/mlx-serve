import Foundation
import Combine

/// Bridges a Telegram bot to the local model. Long-polls Telegram for incoming
/// messages, runs each through the app's *canonical* chat/agent engine against a
/// hidden per-chat session, then sends the reply back. Communication is
/// outbound HTTPS only (Telegram long-poll), so it works behind home NAT with no
/// public URL, tunnel, or port-forward.
///
/// Lifecycle is driven by `ServerOptions.telegram` through `AppState`:
/// `reconcile()` starts / stops / restarts the poll loop when the token or the
/// enabled flag changes. `agentMode`, `enableThinking`, and the allow-list are
/// read live per message, so changing them needs no restart.
///
/// Reuse, not reimplementation: a *dedicated* `ChatTurnEngine` instance runs the
/// same loop the chat window and voice assistant use (tools, MCP-off, memory,
/// thinking, truncation/retry recovery). Using a separate instance means a
/// phone-driven turn never cancels — or is cancelled by — the user's in-app
/// turn; both still serialize on the one model server-side, which is correct.
@MainActor
final class TelegramBridge: ObservableObject {
    enum Status: Equatable {
        case off
        case connecting
        case listening(username: String?)
        case error(String)

        /// Short human label for the Settings status pill.
        var label: String {
            switch self {
            case .off: return "Off"
            case .connecting: return "Connecting…"
            case .listening(let u): return u.map { "Listening as @\($0)" } ?? "Listening"
            case .error(let m): return m
            }
        }

        var isHealthy: Bool { if case .listening = self { return true }; return false }
    }

    @Published private(set) var status: Status = .off
    /// True while a Telegram-driven turn is generating. `TaskScheduler.drain()`
    /// reads this so a createTask spawned from a Telegram turn waits for that turn
    /// to finish instead of running a second engine against the model concurrently.
    @Published private(set) var isProcessing = false

    /// Owning app state. `unowned` because the bridge is a `lazy var` on
    /// `AppState` — it never outlives its owner.
    unowned let appState: AppState

    /// Dedicated engine — isolates Telegram turns from the user's in-app chat
    /// engine (see the type doc).
    private lazy var engine = ChatTurnEngine(appState: appState)

    private var pollTask: Task<Void, Never>?
    /// Debounces `reconcile()`: every Settings keystroke mutates `serverOptions`
    /// (→ didSet → reconcile), so applying immediately would tear down + restart
    /// the poll loop on each character of the bot token — hammering getMe with a
    /// partial token (401 thrash). We coalesce to the last change after a quiet gap.
    private var reconcileDebounce: Task<Void, Never>?
    /// Config the running loop was started with. Lets `reconcile()` no-op on the
    /// unrelated `ServerOptions` mutations that fire on every Settings keystroke.
    private var appliedConfig: ServerOptions.TelegramConfig?
    /// Telegram chat id → hidden session id. In-memory only: conversations reset
    /// when the app restarts (they live on the phone, not the chat sidebar).
    private var sessions: [Int64: UUID] = [:]

    /// Telegram long-poll hold, seconds. The URLSession request timeout must
    /// comfortably exceed this.
    private let pollTimeout = 25
    private let session: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = 60
        c.timeoutIntervalForResource = 120
        c.waitsForConnectivity = true
        return URLSession(configuration: c)
    }()

    /// Sandbox the Telegram agent's file tools land in (agent mode only).
    static let agentWorkspace: String = {
        let p = NSString(string: "~/.mlx-serve/telegram-workspace").expandingTildeInPath
        try? FileManager.default.createDirectory(atPath: p, withIntermediateDirectories: true)
        return p
    }()

    init(appState: AppState) { self.appState = appState }

    // MARK: - Lifecycle

    /// Start / stop / restart the poll loop to match the current config. Cheap
    /// no-op when the fields that affect the connection (token, enabled) are
    /// unchanged — everything else is read live per message.
    func reconcile() {
        // Debounce: rapid serverOptions mutations (typing/pasting the token) each
        // fire didSet → reconcile; coalesce to the last change after a quiet gap so
        // the poll loop isn't torn down + restarted per keystroke against a partial
        // token (which would 401-thrash and write the prefs file every character).
        reconcileDebounce?.cancel()
        reconcileDebounce = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 800_000_000)
            guard !Task.isCancelled, let self else { return }
            self.reconcileDebounce = nil
            self.applyReconcile()
        }
    }

    private func applyReconcile() {
        let cfg = appState.serverOptions.telegram

        // Already running with the same connection params → nothing to restart.
        if cfg.isRunnable, pollTask != nil, let applied = appliedConfig,
           applied.enabled == cfg.enabled, applied.trimmedToken == cfg.trimmedToken {
            appliedConfig = cfg
            return
        }

        stop()
        appliedConfig = cfg
        guard cfg.isRunnable else { status = .off; return }
        status = .connecting
        let token = cfg.trimmedToken
        pollTask = Task { [weak self] in await self?.runLoop(token: token) }
    }

    func stop() {
        reconcileDebounce?.cancel()
        reconcileDebounce = nil
        pollTask?.cancel()
        pollTask = nil
        status = .off
    }

    // MARK: - Poll loop

    private func runLoop(token: String) async {
        // Best-effort getMe so the Settings pill can show "Listening as @bot".
        let username = await fetchUsername(token: token)
        if Task.isCancelled { return }
        status = .listening(username: username)

        // Resume from the last persisted offset so a restart doesn't re-fetch
        // (and re-answer) updates Telegram hadn't yet had confirmed.
        var offset: Int64 = Self.loadOffset(token: token)
        var backoffSeconds = 1
        while !Task.isCancelled {
            guard let url = TelegramAPI.getUpdatesURL(token: token, offset: offset, timeout: pollTimeout) else {
                status = .error("Malformed bot token.")
                return
            }
            do {
                let (data, response) = try await session.data(from: url)
                let httpStatus = (response as? HTTPURLResponse)?.statusCode ?? 200
                if httpStatus == 401 {
                    status = .error("Invalid bot token (401) — re-check it in @BotFather.")
                    return
                }
                if httpStatus != 200 {
                    // 429 / 409 / 5xx return instantly (no long-poll hold). Falling
                    // through would parse {"ok":false} → no offset advance → tight
                    // busy-loop hammering the API. Back off and retry instead.
                    status = .error("Telegram API \(httpStatus) — retrying…")
                    try? await Task.sleep(nanoseconds: UInt64(backoffSeconds) * 1_000_000_000)
                    backoffSeconds = min(backoffSeconds * 2, 30)
                    continue
                }
                // Recovered from a prior transient error.
                if !status.isHealthy { status = .listening(username: username) }
                backoffSeconds = 1

                let (updates, nextOffset) = TelegramAPI.parseUpdates(data)
                if let nextOffset { offset = nextOffset }
                for update in updates {
                    if Task.isCancelled { return }
                    await handle(update, token: token)
                }
                // Persist AFTER handling: a crash mid-batch leaves the offset on the
                // prior value so nothing is lost (at worst a handled item replays),
                // while a clean restart resumes past the batch — no duplicate replies.
                if let nextOffset { Self.saveOffset(nextOffset, token: token) }
            } catch {
                if Task.isCancelled { return }
                // Transient network issue — surface briefly, back off, retry.
                status = .error("Reconnecting…")
                try? await Task.sleep(nanoseconds: UInt64(backoffSeconds) * 1_000_000_000)
                backoffSeconds = min(backoffSeconds * 2, 30)
            }
        }
    }

    // Persisted poll offset, keyed by the bot id (the non-secret part of the token
    // before ":") so changing tokens doesn't resume at a stale offset.
    private static func offsetKey(token: String) -> String {
        let botId = token.split(separator: ":").first.map(String.init) ?? "default"
        return "telegramPollOffset.\(botId)"
    }
    private static func loadOffset(token: String) -> Int64 {
        Int64(UserDefaults.standard.integer(forKey: offsetKey(token: token)))
    }
    private static func saveOffset(_ offset: Int64, token: String) {
        UserDefaults.standard.set(offset, forKey: offsetKey(token: token))
    }

    // MARK: - Per-message handling

    private func handle(_ update: TelegramUpdate, token: String) async {
        let text = update.text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        let chatId = update.chatId

        // Access gate — read live so allow-list edits in Settings take effect at
        // once. This is the bot's only protection against a stranger driving the
        // local model, so it runs before anything else.
        switch appState.serverOptions.telegram.access(forChatId: chatId) {
        case .rejected:
            await send(token: token, chatId: chatId, text: "⛔️ This bot is locked to another chat.")
            return
        case .adopt:
            // Trust-on-first-use: the first chat to message becomes the owner.
            // Mutating serverOptions persists it (AppState.didSet) and harmlessly
            // re-enters reconcile() as a no-op (token/enabled unchanged).
            appState.serverOptions.telegram.allowedChatIds.append(chatId)
            await send(token: token, chatId: chatId,
                       text: "✅ Locked to this chat. I'll relay your messages to the local model on \(hostName()). Send /new anytime to start a fresh conversation.")
            // Fall through and answer this first message (unless it's a command).
        case .allowed:
            break
        }

        // Commands.
        switch text {
        case "/start":
            await send(token: token, chatId: chatId, text: startHelp())
            return
        case "/new", "/reset":
            sessions[chatId] = nil
            await send(token: token, chatId: chatId, text: "🧹 Started a new conversation.")
            return
        default:
            break
        }

        guard appState.server.status == .running else {
            await send(token: token, chatId: chatId,
                       text: "⚠️ No model is loaded right now. Open MLX Core, start a model, then message me again.")
            return
        }

        let reply = await generateReply(chatId: chatId, senderName: update.senderName, text: text)
        for chunk in TelegramAPI.splitForTelegram(reply) {
            await send(token: token, chatId: chatId, text: chunk)
        }
    }

    /// Run one turn through the canonical engine against the chat's hidden
    /// session and return the final assistant text.
    private func generateReply(chatId: Int64, senderName: String, text: String) async -> String {
        let cfg = appState.serverOptions.telegram
        let sessionId = sessionId(for: chatId, senderName: senderName, agentMode: cfg.agentMode)
        let workspace = Self.agentWorkspace
        let turnConfig = ChatTurnEngine.TurnConfig(
            agentMode: cfg.agentMode,
            mcpMode: cfg.useMCP,
            enableThinking: cfg.enableThinking,
            voiceStyle: false,
            workingDirectory: (cfg.agentMode || cfg.useMCP) ? workspace : nil,
            telegramChatId: chatId   // so the agent's createTask reports back here
        )

        engine.runTurn(
            sessionId: sessionId,
            userText: text,
            images: nil,
            audio: nil,
            config: turnConfig,
            approval: { tc in
                // Agent-over-Telegram has no interactive approval surface, so
                // reuse the pure, tested ApprovalPolicy at fullAuto: read-only +
                // shell + workspace-confined writes auto-allow; out-of-workspace
                // writes and unknown tools are denied (the loop adapts). The
                // allow-list is the real gate keeping strangers out.
                ApprovalPolicy.decide(
                    tool: tc.name, autonomy: .fullAuto,
                    arguments: tc.arguments, rawArguments: tc.rawArguments,
                    workingDirectory: workspace
                ) == .allow
            }
        )
        isProcessing = true
        defer { isProcessing = false }
        await awaitEngineIdle()
        return lastAssistantText(sessionId: sessionId)
    }

    // MARK: - Session bookkeeping

    private func sessionId(for chatId: Int64, senderName: String, agentMode: Bool) -> UUID {
        if let existing = sessions[chatId],
           appState.chatSessions.contains(where: { $0.id == existing }) {
            return existing
        }
        var s = ChatSession(title: "\(senderName) (Telegram)")
        s.isExternalBridge = true
        if agentMode {
            s.mode = .agent
            s.workingDirectory = Self.agentWorkspace
        }
        appState.chatSessions.append(s)   // hidden from the sidebar; never persisted
        sessions[chatId] = s.id
        return s.id
    }

    /// Await the dedicated engine returning to idle. `runTurn` sets
    /// `isGenerating = true` synchronously when it starts, so by the time we
    /// subscribe the value is either still `true` (wait for the `false`
    /// transition) or already `false` (turn finished — return at once).
    private func awaitEngineIdle() async {
        for await generating in engine.$isGenerating.values {
            if !generating { return }
        }
    }

    private func lastAssistantText(sessionId: UUID) -> String {
        let msgs = appState.chatSessions.first { $0.id == sessionId }?.messages ?? []
        let content = msgs.last {
            $0.role == .assistant && !$0.isAgentSummary && !$0.failedRetry
        }?.content ?? ""
        let cleaned = content
            .replacingOccurrences(of: "<pad>", with: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return cleaned.isEmpty
            ? "⚠️ I couldn't generate a response. Try /new and rephrase."
            : cleaned
    }

    // MARK: - Task result delivery (called by TaskScheduler.finalize)

    /// Push a finished task's result to the Telegram chat that created it (via the
    /// agent's `createTask` tool). Runs ALONGSIDE the desktop notification, not
    /// instead of it. Best-effort and independent of the poll loop — it sends as
    /// long as a token is configured, so a scheduled task still reports even if
    /// the bridge happens to be toggled off.
    func deliverTaskResult(chatId: Int64, task: ScheduledTask, run: TaskRun) {
        let token = appState.serverOptions.telegram.trimmedToken
        guard !token.isEmpty else { return }
        let header = run.status == .completed
            ? "✅ Task “\(task.title)” finished"
            : "⚠️ Task “\(task.title)” failed"
        let text = header + (run.summary.map { "\n\n\($0)" } ?? "")
        Task { [weak self] in
            guard let self else { return }
            for chunk in TelegramAPI.splitForTelegram(text) {
                await self.send(token: token, chatId: chatId, text: chunk)
            }
        }
    }

    // MARK: - Telegram I/O

    @discardableResult
    private func send(token: String, chatId: Int64, text: String) async -> Bool {
        guard let url = TelegramAPI.sendMessageURL(token: token) else { return false }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = TelegramAPI.sendMessageBody(chatId: chatId, text: text)
        do {
            let (_, response) = try await session.data(for: req)
            return (response as? HTTPURLResponse)?.statusCode == 200
        } catch {
            return false
        }
    }

    private func fetchUsername(token: String) async -> String? {
        guard let url = TelegramAPI.getMeURL(token: token),
              let (data, _) = try? await session.data(from: url) else { return nil }
        return TelegramAPI.parseUsername(data)
    }

    // MARK: - Copy

    private func startHelp() -> String {
        let mode = appState.serverOptions.telegram.agentMode
            ? "agent mode (it can run shell commands and edit files on the Mac)"
            : "chat mode"
        return """
        👋 Connected to MLX Core in \(mode).
        Send a message and I'll relay it to your local model.
        Commands:
        /new — start a fresh conversation
        """
    }

    private func hostName() -> String { Host.current().localizedName ?? "this Mac" }
}
