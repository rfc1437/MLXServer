import SwiftUI

struct ExportChatAction {
    let perform: () -> Void

    func callAsFunction() {
        perform()
    }
}

/// Focused value key for triggering chat export from the menu bar.
struct FocusedExportActionKey: FocusedValueKey {
    typealias Value = ExportChatAction
}

extension FocusedValues {
    var exportChatAction: ExportChatAction? {
        get { self[FocusedExportActionKey.self] }
        set { self[FocusedExportActionKey.self] = newValue }
    }
}
