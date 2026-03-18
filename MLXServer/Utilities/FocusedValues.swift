import SwiftUI

/// Focused value key for triggering chat export from the menu bar.
struct FocusedExportTriggerKey: FocusedValueKey {
    typealias Value = Binding<Bool>
}

extension FocusedValues {
    var exportTrigger: Binding<Bool>? {
        get { self[FocusedExportTriggerKey.self] }
        set { self[FocusedExportTriggerKey.self] = newValue }
    }
}
