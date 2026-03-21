import Foundation

/// Persisted app preferences via UserDefaults.
enum Preferences {
    nonisolated(unsafe) private static let defaults = UserDefaults.standard

    private static let jsonEncoder = JSONEncoder()
    private static let jsonDecoder = JSONDecoder()

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

    // MARK: - Scenes

    private static let scenesKey = "chatScenes"
    private static let lastSceneIdKey = "lastSceneId"

    static var scenes: [ChatScene] {
        get {
            guard let data = defaults.data(forKey: scenesKey) else { return [] }
            return (try? jsonDecoder.decode([ChatScene].self, from: data)) ?? []
        }
        set {
            guard let data = try? jsonEncoder.encode(newValue) else { return }
            defaults.set(data, forKey: scenesKey)
        }
    }

    static var lastSceneId: UUID? {
        get {
            guard let rawValue = defaults.string(forKey: lastSceneIdKey) else { return nil }
            return UUID(uuidString: rawValue)
        }
        set { defaults.set(newValue?.uuidString, forKey: lastSceneIdKey) }
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

    // MARK: - KV Cache Quantization

    private static let kvQuantizationEnabledKey = "kvQuantizationEnabled"
    private static let kvQuantizationBitsKey = "kvQuantizationBits"

    /// Whether to quantize KV caches for compact storage (50% memory savings at 8-bit).
    /// Default: false (disabled for maximum quality). Requires TokenPrefixCache Phase 6.
    static var kvQuantizationEnabled: Bool {
        get { defaults.object(forKey: kvQuantizationEnabledKey) == nil ? false : defaults.bool(forKey: kvQuantizationEnabledKey) }
        set { defaults.set(newValue, forKey: kvQuantizationEnabledKey) }
    }

    /// Bit width for KV cache quantization. Standard: 8 (recommended). Range: 4-16.
    /// Lower bits = more compression but potential quality loss. 8-bit is proven in production.
    static var kvQuantizationBits: Int {
        get {
            let val = defaults.integer(forKey: kvQuantizationBitsKey)
            return val > 0 ? val : 8
        }
        set {
            // Clamp to valid range
            let clamped = max(4, min(newValue, 16))
            defaults.set(clamped, forKey: kvQuantizationBitsKey)
        }
    }
}
