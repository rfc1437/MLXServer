import SwiftUI

struct GenerationDefaultsEditor: View {
    @Binding var settings: GenerationSettings

    var body: some View {
        Toggle("Enable thinking mode", isOn: $settings.thinkingEnabled)
        DecimalSettingRow(title: "Temperature", value: $settings.temperature)
        DecimalSettingRow(title: "Top P", value: $settings.topP)
        IntegerSettingRow(title: "Top K", value: $settings.topK)
        DecimalSettingRow(title: "Min P", value: $settings.minP)
        IntegerSettingRow(title: "Max tokens", value: $settings.maxTokens)
        OptionalDecimalSettingRow(title: "Repetition penalty", value: $settings.repetitionPenalty, fallbackValue: 1.0)
        OptionalDecimalSettingRow(title: "Presence penalty", value: $settings.presencePenalty, fallbackValue: 0.0)
        OptionalDecimalSettingRow(title: "Frequency penalty", value: $settings.frequencyPenalty, fallbackValue: 0.0)
    }
}

struct GenerationOverridesEditor: View {
    @Binding var overrides: GenerationSettingsOverride
    let inheritedSettings: GenerationSettings
    let inheritedSource: String

    var body: some View {
        Picker("Thinking mode", selection: $overrides.thinkingEnabled) {
            Text("Inherited (\(inheritedSettings.thinkingEnabled ? "Enabled" : "Disabled"))").tag(Optional<Bool>.none)
            Text("Enabled").tag(Optional(true))
            Text("Disabled").tag(Optional(false))
        }

        OptionalDecimalSettingRow(title: "Temperature", value: $overrides.temperature, fallbackValue: inheritedSettings.temperature, inherited: true)
        OptionalDecimalSettingRow(title: "Top P", value: $overrides.topP, fallbackValue: inheritedSettings.topP, inherited: true)
        OptionalIntegerSettingRow(title: "Top K", value: $overrides.topK, fallbackValue: inheritedSettings.topK, inherited: true)
        OptionalDecimalSettingRow(title: "Min P", value: $overrides.minP, fallbackValue: inheritedSettings.minP, inherited: true)
        OptionalIntegerSettingRow(title: "Max tokens", value: $overrides.maxTokens, fallbackValue: inheritedSettings.maxTokens, inherited: true)
        OptionalDecimalSettingRow(title: "Repetition penalty", value: $overrides.repetitionPenalty, fallbackValue: inheritedSettings.repetitionPenalty ?? 0, inherited: true)
        OptionalDecimalSettingRow(title: "Presence penalty", value: $overrides.presencePenalty, fallbackValue: inheritedSettings.presencePenalty ?? 0, inherited: true)
        OptionalDecimalSettingRow(title: "Frequency penalty", value: $overrides.frequencyPenalty, fallbackValue: inheritedSettings.frequencyPenalty ?? 0, inherited: true)

        Text("Unset fields inherit from \(inheritedSource). The values shown are the effective starting values for this scene.")
            .font(.caption)
            .foregroundStyle(.secondary)
    }
}

private struct DecimalSettingRow: View {
    let title: String
    @Binding var value: Double
    @State private var text: String

    init(title: String, value: Binding<Double>) {
        self.title = title
        self._value = value
        self._text = State(initialValue: NumericFieldFormatting.doubleString(value.wrappedValue))
    }

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            TextField("", text: $text)
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
                .onChange(of: text) {
                    if let parsed = NumericFieldFormatting.parseDouble(text) {
                        value = parsed
                    }
                }
                .onChange(of: value) {
                    let formatted = NumericFieldFormatting.doubleString(value)
                    if text != formatted {
                        text = formatted
                    }
                }
        }
    }
}

private struct IntegerSettingRow: View {
    let title: String
    @Binding var value: Int
    @State private var text: String

    init(title: String, value: Binding<Int>) {
        self.title = title
        self._value = value
        self._text = State(initialValue: NumericFieldFormatting.intString(value.wrappedValue))
    }

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            TextField("", text: $text)
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
                .onChange(of: text) {
                    if let parsed = NumericFieldFormatting.parseInt(text) {
                        value = parsed
                    }
                }
                .onChange(of: value) {
                    let formatted = NumericFieldFormatting.intString(value)
                    if text != formatted {
                        text = formatted
                    }
                }
        }
    }
}

private struct OptionalDecimalSettingRow: View {
    let title: String
    @Binding var value: Double?
    let fallbackValue: Double
    var inherited = false
    @State private var text: String

    init(title: String, value: Binding<Double?>, fallbackValue: Double, inherited: Bool = false) {
        self.title = title
        self._value = value
        self.fallbackValue = fallbackValue
        self.inherited = inherited
        self._text = State(initialValue: NumericFieldFormatting.doubleString(value.wrappedValue ?? fallbackValue))
    }

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            TextField("", text: $text)
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
                .onChange(of: text) {
                    if let parsed = NumericFieldFormatting.parseDouble(text) {
                        value = parsed
                    }
                }
                .onChange(of: value) {
                    syncText()
                }
                .onChange(of: fallbackValue) {
                    if value == nil {
                        syncText()
                    }
                }
            if inherited && value == nil {
                Text("Inherited")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Button(value == nil ? "Override" : "Clear") {
                if value == nil {
                    value = fallbackValue
                } else {
                    value = nil
                }
                syncText()
            }
            .buttonStyle(.link)
        }
    }

    private func syncText() {
        let formatted = NumericFieldFormatting.doubleString(value ?? fallbackValue)
        if text != formatted {
            text = formatted
        }
    }
}

private struct OptionalIntegerSettingRow: View {
    let title: String
    @Binding var value: Int?
    let fallbackValue: Int
    var inherited = false
    @State private var text: String

    init(title: String, value: Binding<Int?>, fallbackValue: Int, inherited: Bool = false) {
        self.title = title
        self._value = value
        self.fallbackValue = fallbackValue
        self.inherited = inherited
        self._text = State(initialValue: NumericFieldFormatting.intString(value.wrappedValue ?? fallbackValue))
    }

    var body: some View {
        HStack {
            Text(title)
            Spacer()
            TextField("", text: $text)
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
                .onChange(of: text) {
                    if let parsed = NumericFieldFormatting.parseInt(text) {
                        value = parsed
                    }
                }
                .onChange(of: value) {
                    syncText()
                }
                .onChange(of: fallbackValue) {
                    if value == nil {
                        syncText()
                    }
                }
            if inherited && value == nil {
                Text("Inherited")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Button(value == nil ? "Override" : "Clear") {
                if value == nil {
                    value = fallbackValue
                } else {
                    value = nil
                }
                syncText()
            }
            .buttonStyle(.link)
        }
    }

    private func syncText() {
        let formatted = NumericFieldFormatting.intString(value ?? fallbackValue)
        if text != formatted {
            text = formatted
        }
    }
}

private enum NumericFieldFormatting {
    static func parseDouble(_ text: String) -> Double? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return Double(trimmed.replacingOccurrences(of: ",", with: "."))
    }

    static func parseInt(_ text: String) -> Int? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return Int(trimmed)
    }

    static func doubleString(_ value: Double) -> String {
        if value.rounded() == value {
            return String(Int(value))
        }
        return String(value)
    }

    static func intString(_ value: Int) -> String {
        String(value)
    }
}