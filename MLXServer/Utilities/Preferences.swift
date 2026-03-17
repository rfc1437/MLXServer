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
