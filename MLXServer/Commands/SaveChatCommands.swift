import SwiftUI

/// Adds "Export Chat…" to the File menu.
struct SaveChatCommands: Commands {
    @FocusedValue(\.exportChatAction) private var exportChatAction

    var body: some Commands {
        CommandGroup(after: .saveItem) {
            Button("Export Chat…") {
                exportChatAction?()
            }
            .keyboardShortcut("s", modifiers: [.command, .shift])
            .disabled(exportChatAction == nil)
        }
    }
}
