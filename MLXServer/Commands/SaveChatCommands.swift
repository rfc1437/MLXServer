import SwiftUI

struct SaveChatCommands: Commands {
    @FocusedValue(\.newChatAction) private var newChatAction
    @FocusedValue(\.openChatAction) private var openChatAction
    @FocusedValue(\.saveChatAction) private var saveChatAction
    @FocusedValue(\.saveChatAsAction) private var saveChatAsAction
    @FocusedValue(\.revertChatAction) private var revertChatAction
    @FocusedValue(\.exportChatAction) private var exportChatAction

    var body: some Commands {
        CommandGroup(replacing: .newItem) {
            Button("New Chat") {
                newChatAction?()
            }
            .keyboardShortcut("n", modifiers: .command)

            Button("Open Chat…") {
                openChatAction?()
            }
            .keyboardShortcut("o", modifiers: .command)
            .disabled(openChatAction == nil)
        }

        CommandGroup(replacing: .saveItem) {
            Button("Save Chat") {
                saveChatAction?()
            }
            .keyboardShortcut("s", modifiers: .command)
            .disabled(saveChatAction == nil)

            Button("Save Chat As…") {
                saveChatAsAction?()
            }
            .keyboardShortcut("s", modifiers: [.command, .shift])
            .disabled(saveChatAsAction == nil)

            Divider()

            Button("Revert To Saved") {
                revertChatAction?()
            }
            .disabled(revertChatAction == nil)

            Divider()

            Button("Export Chat…") {
                exportChatAction?()
            }
            .keyboardShortcut("e", modifiers: [.command, .shift])
            .disabled(exportChatAction == nil)
        }
    }
}
