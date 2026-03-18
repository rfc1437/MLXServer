import SwiftUI

/// Adds "Export Chat…" to the File menu.
struct SaveChatCommands: Commands {
    @FocusedBinding(\.exportTrigger) var isExporting

    var body: some Commands {
        CommandGroup(after: .saveItem) {
            Button("Export Chat…") {
                isExporting = true
            }
            .keyboardShortcut("e", modifiers: [.command, .shift])
            .disabled(isExporting == nil)
        }
    }
}
