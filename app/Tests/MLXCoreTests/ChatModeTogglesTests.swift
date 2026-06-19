import XCTest
@testable import MLXCore

/// Pins which source the chat toolbar's Think/Agent/MCP toggles reflect: a
/// Telegram bridge session must mirror `serverOptions.telegram` (so the toolbar
/// stays in sync with Settings — one source of truth), while a normal session
/// uses the in-app per-session / app-level state.
final class ChatModeTogglesTests: XCTestCase {

    func testTelegramSessionReflectsTelegramConfigNotInApp() {
        let t = ChatModeToggles.resolve(
            isExternalBridge: true,
            telegramThinking: true, telegramAgent: true, telegramMCP: false,
            inAppThinking: false, inAppAgent: false, inAppMCP: true)   // in-app differs → must be ignored
        XCTAssertEqual(t, ChatModeToggles(thinking: true, agent: true, mcp: false))
    }

    func testNormalSessionReflectsInAppStateNotTelegram() {
        let t = ChatModeToggles.resolve(
            isExternalBridge: false,
            telegramThinking: true, telegramAgent: true, telegramMCP: true,   // telegram differs → ignored
            inAppThinking: false, inAppAgent: true, inAppMCP: false)
        XCTAssertEqual(t, ChatModeToggles(thinking: false, agent: true, mcp: false))
    }
}
