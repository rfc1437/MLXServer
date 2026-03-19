import SwiftUI
import MLX

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    var chatViewModel: ChatViewModel?

    func application(_ application: NSApplication, open urls: [URL]) {
        ChatDocumentController.shared.enqueueOpenRequests(urls)
    }

    func applicationWillTerminate(_ notification: Notification) {
        chatViewModel?.autosaveToSandbox()
    }
}

@main
struct MLXServerApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @State private var documentController = ChatDocumentController.shared
    @State private var modelManager = ModelManager()
    @State private var sceneStore = SceneStore()

    init() {
        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(documentController)
                .environment(modelManager)
                .environment(sceneStore)
                .task {
                    guard !documentController.hasPendingOpenRequests else { return }
                    guard !ChatViewModel.hasAutosavedSession else { return }
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
            SceneCommands()
        }

        Window("Scenes", id: SceneManagementWindow.windowID) {
            SceneManagementView()
                .environment(sceneStore)
        }
        .defaultSize(width: 900, height: 560)

        #if os(macOS)
        Settings {
            SettingsView()
                .environment(sceneStore)
        }
        #endif
    }
}
