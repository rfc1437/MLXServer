import SwiftUI
import MLX

@main
struct MLXServerApp: App {
    @State private var modelManager = ModelManager()

    init() {
        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(modelManager)
                .task {
                    // Auto-load: configured default → last used → built-in default
                    let modelId = Preferences.defaultModelId ?? Preferences.lastModelId ?? ModelConfig.default.id
                    if let config = ModelConfig.availableModels.first(where: { $0.id == modelId }) {
                        await modelManager.loadModel(config)
                    }
                }
        }
        .windowStyle(.titleBar)
        .defaultSize(width: 800, height: 700)
        .commands {
            SaveChatCommands()
        }

        #if os(macOS)
        Settings {
            SettingsView()
        }
        #endif
    }
}
