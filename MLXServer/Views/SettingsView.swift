import SwiftUI

struct SettingsView: View {
    @Environment(\.openWindow) private var openWindow
    @Environment(SceneStore.self) private var sceneStore
    @State private var systemPrompt: String = Preferences.systemPrompt
    @State private var apiPort: String = String(Preferences.apiPort)
    @State private var apiAutoStart: Bool = Preferences.apiAutoStart
    @State private var idleUnloadMinutes: String = String(Preferences.idleUnloadMinutes)
    @State private var defaultModelId: String = Preferences.defaultModelId ?? ModelConfig.default.id
    @State private var generationDefaultsModelId: String = Preferences.defaultModelId ?? ModelConfig.default.id
    @State private var kvQuantizationEnabled: Bool = Preferences.kvQuantizationEnabled
    @State private var kvQuantizationBits: Int = Preferences.kvQuantizationBits

    private var kvQuantizationConfig: TokenPrefixCache.QuantizationConfig {
        guard kvQuantizationEnabled else {
            return .default
        }

        return .init(
            enabled: true,
            bits: kvQuantizationBits,
            groupSize: 64,
            minTokens: 256
        )
    }

    var body: some View {
        Form {
            Section("Startup") {
                Picker("Default model", selection: $defaultModelId) {
                    ForEach(ModelConfig.availableModels) { model in
                        Text(model.displayName).tag(model.id)
                    }
                }
                .onChange(of: defaultModelId) {
                    Preferences.defaultModelId = defaultModelId
                }

                Text("The model to load automatically when the app starts.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Generation Defaults") {
                Picker("Defaults for model", selection: $generationDefaultsModelId) {
                    ForEach(ModelConfig.availableModels) { model in
                        Text(model.displayName).tag(model.id)
                    }
                }

                GenerationDefaultsEditor(settings: generationDefaultsBinding)

                Text("These are the per-model defaults used by chat sessions and by the API server whenever a request omits a generation parameter. Lower temperature and stronger repetition penalties are usually better for technical work; higher temperature is usually better for improvisation and roleplay.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("System Prompt") {
                TextEditor(text: $systemPrompt)
                    .font(.body.monospaced())
                    .frame(minHeight: 80)
                    .onChange(of: systemPrompt) {
                        Preferences.systemPrompt = systemPrompt
                    }

                Text("Applied to new conversations. Leave empty for no system prompt.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Scenes") {
                Button("Manage Scenes…") {
                    openWindow(id: SceneManagementWindow.windowID)
                }

                Text("Create reusable roleplay or task presets with a dedicated model, extra system prompt, and an auto-sent opening message.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                if sceneStore.scenes.isEmpty {
                    Text("No saved scenes yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Saved scenes: \(sceneStore.scenes.count)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section("API Server") {
                HStack {
                    Text("Port")
                    TextField("1234", text: $apiPort)
                        .frame(width: 80)
                        .onChange(of: apiPort) {
                            if let port = Int(apiPort), port > 0, port < 65536 {
                                Preferences.apiPort = port
                            }
                        }
                }

                Toggle("Start API server automatically", isOn: $apiAutoStart)
                    .onChange(of: apiAutoStart) {
                        Preferences.apiAutoStart = apiAutoStart
                    }
            }

            Section("Memory") {
                HStack {
                    Text("Unload model after idle")
                    TextField("3", text: $idleUnloadMinutes)
                        .frame(width: 50)
                        .onChange(of: idleUnloadMinutes) {
                            if let mins = Int(idleUnloadMinutes), mins > 0 {
                                Preferences.idleUnloadMinutes = mins
                            }
                        }
                    Text("minutes")
                        .foregroundStyle(.secondary)
                }

                Text("The model is automatically unloaded to free memory after being idle, and reloaded on the next request.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Cache Quantization") {
                Toggle("Enable KV cache quantization", isOn: $kvQuantizationEnabled)
                    .onChange(of: kvQuantizationEnabled) {
                        Preferences.kvQuantizationEnabled = kvQuantizationEnabled
                        TokenPrefixCache.shared.setQuantizationConfig(kvQuantizationConfig)
                    }

                if kvQuantizationEnabled {
                    HStack {
                        Text("Bit width")
                        Spacer()
                        Stepper(
                            value: $kvQuantizationBits,
                            in: 4...16,
                            step: 1
                        ) {
                            Text("\(kvQuantizationBits)-bit")
                        }
                        .onChange(of: kvQuantizationBits) {
                            Preferences.kvQuantizationBits = kvQuantizationBits
                            TokenPrefixCache.shared.setQuantizationConfig(kvQuantizationConfig)
                        }
                    }
                }

                if kvQuantizationEnabled {
                    Text("Quantizes KV caches to \(kvQuantizationBits)-bit for \(kvQuantizationBits == 8 ? "~50%" : "~\((16 - kvQuantizationBits) * 6)%") memory savings. Lower bits = more compression but may impact response quality. 8-bit is recommended.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("When enabled, KV caches are quantized for compact storage, reducing memory usage on long conversations. Disabled by default for maximum quality.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 450, height: 650)
    }

    private var generationDefaultsBinding: Binding<GenerationSettings> {
        Binding(
            get: { Preferences.generationSettings(forModelId: generationDefaultsModelId) },
            set: { Preferences.setGenerationSettings($0, forModelId: generationDefaultsModelId) }
        )
    }
}
