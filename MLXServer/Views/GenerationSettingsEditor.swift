import SwiftUI

private let generationDoubleFormat = FloatingPointFormatStyle<Double>.number.precision(.fractionLength(0...2))
private let generationIntegerFormat = IntegerFormatStyle<Int>.number.grouping(.never)

struct GenerationDefaultsEditor: View {
    @Binding var settings: GenerationSettings

    var body: some View {
        Toggle("Enable thinking mode", isOn: $settings.thinkingEnabled)
        doubleRow("Temperature", value: $settings.temperature)
        doubleRow("Top P", value: $settings.topP)
        intRow("Top K", value: $settings.topK)
        doubleRow("Min P", value: $settings.minP)
        intRow("Max tokens", value: $settings.maxTokens)
        optionalDoubleRow("Repetition penalty", value: $settings.repetitionPenalty)
        optionalDoubleRow("Presence penalty", value: $settings.presencePenalty)
        optionalDoubleRow("Frequency penalty", value: $settings.frequencyPenalty)
    }

    private func doubleRow(_ title: String, value: Binding<Double>) -> some View {
        HStack {
            Text(title)
            Spacer()
            TextField(title, value: value, format: generationDoubleFormat)
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
        }
    }

    private func intRow(_ title: String, value: Binding<Int>) -> some View {
        HStack {
            Text(title)
            Spacer()
            TextField(title, value: value, format: generationIntegerFormat)
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
        }
    }

    private func optionalDoubleRow(_ title: String, value: Binding<Double?>) -> some View {
        HStack {
            Text(title)
            Spacer()
            TextField(title, value: binding(for: value), format: generationDoubleFormat)
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
            Button(value.wrappedValue == nil ? "Set" : "Clear") {
                if value.wrappedValue == nil {
                    value.wrappedValue = 1.0
                } else {
                    value.wrappedValue = nil
                }
            }
            .buttonStyle(.link)
        }
    }

    private func binding(for value: Binding<Double?>) -> Binding<Double> {
        Binding(
            get: { value.wrappedValue ?? 1.0 },
            set: { value.wrappedValue = $0 }
        )
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

        optionalDoubleRow("Temperature", value: $overrides.temperature, inheritedValue: inheritedSettings.temperature)
        optionalDoubleRow("Top P", value: $overrides.topP, inheritedValue: inheritedSettings.topP)
        optionalIntRow("Top K", value: $overrides.topK, inheritedValue: inheritedSettings.topK)
        optionalDoubleRow("Min P", value: $overrides.minP, inheritedValue: inheritedSettings.minP)
        optionalIntRow("Max tokens", value: $overrides.maxTokens, inheritedValue: inheritedSettings.maxTokens)
        optionalDoubleRow("Repetition penalty", value: $overrides.repetitionPenalty, inheritedValue: inheritedSettings.repetitionPenalty ?? 0)
        optionalDoubleRow("Presence penalty", value: $overrides.presencePenalty, inheritedValue: inheritedSettings.presencePenalty ?? 0)
        optionalDoubleRow("Frequency penalty", value: $overrides.frequencyPenalty, inheritedValue: inheritedSettings.frequencyPenalty ?? 0)

        Text("Unset fields inherit from \(inheritedSource). The values shown are the effective starting values for this scene.")
            .font(.caption)
            .foregroundStyle(.secondary)
    }

    private func optionalDoubleRow(_ title: String, value: Binding<Double?>, inheritedValue: Double) -> some View {
        HStack {
            Text(title)
            Spacer()
            TextField(title, value: Binding(
                get: { value.wrappedValue ?? inheritedValue },
                set: { value.wrappedValue = $0 }
            ), format: generationDoubleFormat)
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
            if value.wrappedValue == nil {
                Text("Inherited")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Button(value.wrappedValue == nil ? "Override" : "Clear") {
                if value.wrappedValue == nil {
                    value.wrappedValue = inheritedValue
                } else {
                    value.wrappedValue = nil
                }
            }
            .buttonStyle(.link)
        }
    }

    private func optionalIntRow(_ title: String, value: Binding<Int?>, inheritedValue: Int) -> some View {
        HStack {
            Text(title)
            Spacer()
            TextField(title, value: Binding(
                get: { value.wrappedValue ?? inheritedValue },
                set: { value.wrappedValue = $0 }
            ), format: generationIntegerFormat)
                .multilineTextAlignment(.trailing)
                .frame(width: 90)
            if value.wrappedValue == nil {
                Text("Inherited")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Button(value.wrappedValue == nil ? "Override" : "Clear") {
                if value.wrappedValue == nil {
                    value.wrappedValue = inheritedValue
                } else {
                    value.wrappedValue = nil
                }
            }
            .buttonStyle(.link)
        }
    }
}