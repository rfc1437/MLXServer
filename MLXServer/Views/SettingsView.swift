import SwiftUI

struct SettingsView: View {
    @State private var systemPrompt: String = Preferences.systemPrompt
    @State private var apiPort: String = String(Preferences.apiPort)
    @State private var apiAutoStart: Bool = Preferences.apiAutoStart
    @State private var idleUnloadMinutes: String = String(Preferences.idleUnloadMinutes)

    var body: some View {
        Form {
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
        }
        .formStyle(.grouped)
        .frame(width: 450, height: 380)
    }
}
