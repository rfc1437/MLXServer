import Foundation

/// Persisted app preferences via UserDefaults.
enum Preferences {
    nonisolated(unsafe) private static let defaults = UserDefaults.standard

    // MARK: - Last used model

    private static let lastModelKey = "lastModelId"

    static var lastModelId: String? {
        get { defaults.string(forKey: lastModelKey) }
        set { defaults.set(newValue, forKey: lastModelKey) }
    }

    // MARK: - Default startup model

    private static let defaultModelKey = "defaultModelId"

    static var defaultModelId: String? {
        get { defaults.string(forKey: defaultModelKey) }
        set { defaults.set(newValue, forKey: defaultModelKey) }
    }

    // MARK: - System prompt

    private static let systemPromptKey = "systemPrompt"

    static var systemPrompt: String {
        get { defaults.string(forKey: systemPromptKey) ?? "" }
        set { defaults.set(newValue, forKey: systemPromptKey) }
    }

    // MARK: - API server

    private static let apiPortKey = "apiPort"
    private static let apiAutoStartKey = "apiAutoStart"

    static var apiPort: Int {
        get {
            let val = defaults.integer(forKey: apiPortKey)
            return val > 0 ? val : 1234
        }
        set { defaults.set(newValue, forKey: apiPortKey) }
    }

    static var apiAutoStart: Bool {
        get { defaults.bool(forKey: apiAutoStartKey) }
        set { defaults.set(newValue, forKey: apiAutoStartKey) }
    }

    // MARK: - Thinking mode

    private static let enableThinkingKey = "enableThinking"

    /// Whether to enable thinking/reasoning mode for models that support it (e.g. Qwen3.5).
    /// When disabled, the model skips internal reasoning and responds directly.
    static var enableThinking: Bool {
        get { defaults.object(forKey: enableThinkingKey) == nil ? true : defaults.bool(forKey: enableThinkingKey) }
        set { defaults.set(newValue, forKey: enableThinkingKey) }
    }

    // MARK: - Idle unload

    private static let idleUnloadMinutesKey = "idleUnloadMinutes"

    static var idleUnloadMinutes: Int {
        get {
            let val = defaults.integer(forKey: idleUnloadMinutesKey)
            return val > 0 ? val : 3
        }
        set { defaults.set(newValue, forKey: idleUnloadMinutesKey) }
    }
}
