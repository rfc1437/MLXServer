import SwiftUI

struct ChatCommandAction {
    let perform: () -> Void

    func callAsFunction() {
        perform()
    }
}

typealias ExportChatAction = ChatCommandAction
typealias NewChatAction = ChatCommandAction
typealias OpenChatAction = ChatCommandAction
typealias SaveChatAction = ChatCommandAction
typealias SaveChatAsAction = ChatCommandAction
typealias RevertChatAction = ChatCommandAction

struct FocusedExportActionKey: FocusedValueKey {
    typealias Value = ExportChatAction
}

struct FocusedNewChatActionKey: FocusedValueKey {
    typealias Value = NewChatAction
}

struct FocusedOpenChatActionKey: FocusedValueKey {
    typealias Value = OpenChatAction
}

struct FocusedSaveChatActionKey: FocusedValueKey {
    typealias Value = SaveChatAction
}

struct FocusedSaveChatAsActionKey: FocusedValueKey {
    typealias Value = SaveChatAsAction
}

struct FocusedRevertChatActionKey: FocusedValueKey {
    typealias Value = RevertChatAction
}

extension FocusedValues {
    var exportChatAction: ExportChatAction? {
        get { self[FocusedExportActionKey.self] }
        set { self[FocusedExportActionKey.self] = newValue }
    }

    var newChatAction: NewChatAction? {
        get { self[FocusedNewChatActionKey.self] }
        set { self[FocusedNewChatActionKey.self] = newValue }
    }

    var openChatAction: OpenChatAction? {
        get { self[FocusedOpenChatActionKey.self] }
        set { self[FocusedOpenChatActionKey.self] = newValue }
    }

    var saveChatAction: SaveChatAction? {
        get { self[FocusedSaveChatActionKey.self] }
        set { self[FocusedSaveChatActionKey.self] = newValue }
    }

    var saveChatAsAction: SaveChatAsAction? {
        get { self[FocusedSaveChatAsActionKey.self] }
        set { self[FocusedSaveChatAsActionKey.self] = newValue }
    }

    var revertChatAction: RevertChatAction? {
        get { self[FocusedRevertChatActionKey.self] }
        set { self[FocusedRevertChatActionKey.self] = newValue }
    }
}
