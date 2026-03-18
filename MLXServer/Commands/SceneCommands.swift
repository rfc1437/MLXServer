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