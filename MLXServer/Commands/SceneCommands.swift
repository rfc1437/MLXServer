import SwiftUI

struct SceneCommands: Commands {
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandMenu("Scenes") {
            Button("Manage Scenes…") {
                openWindow(id: SceneManagementWindow.windowID)
            }
            .keyboardShortcut(",", modifiers: [.command, .shift])
        }
    }
}

struct ModelCommands: Commands {
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandMenu("Models") {
            Button("Manage Models…") {
                openWindow(id: ModelManagementWindow.windowID)
            }
        }
    }
}