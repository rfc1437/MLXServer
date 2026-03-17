import SwiftUI

struct ContentView: View {
    @Environment(ModelManager.self) private var modelManager
    @State private var chatVM: ChatViewModel?
    @State private var showLoadError = false

    var body: some View {
        Group {
            if let chatVM {
                ChatView(viewModel: chatVM)
            } else {
                ProgressView("Initializing…")
            }
        }
        .navigationTitle(modelManager.currentModel?.displayName ?? "MLX Server")
        .onAppear {
            if chatVM == nil {
                chatVM = ChatViewModel(modelManager: modelManager)
                // Auto-start API server if configured
                if Preferences.apiAutoStart {
                    chatVM?.startAPIServer()
                }
            }
        }
        .onChange(of: modelManager.currentModel) {
            chatVM?.resetSession()
            // Persist last used model
            if let id = modelManager.currentModel?.id {
                Preferences.lastModelId = id
            }
        }
        .onChange(of: modelManager.errorMessage) {
            showLoadError = modelManager.errorMessage != nil
        }
        .alert("Model Error", isPresented: $showLoadError) {
            Button("Retry") {
                if let config = modelManager.currentModel ?? ModelConfig.availableModels.first {
                    Task { await modelManager.loadModel(config) }
                }
            }
            Button("Cancel", role: .cancel) {
                modelManager.errorMessage = nil
            }
        } message: {
            Text(modelManager.errorMessage ?? "Unknown error loading model.")
        }
        .toolbar {
            ToolbarItem(placement: .principal) {
                ModelPickerView()
            }

            ToolbarItemGroup(placement: .primaryAction) {
                // API server toggle
                Button {
                    if let chatVM {
                        if chatVM.apiServer.isRunning {
                            chatVM.stopAPIServer()
                        } else {
                            chatVM.startAPIServer()
                        }
                    }
                } label: {
                    // Running → solid globe (green tint), click to stop
                    // Stopped → slashed globe, click to start
                    Label(
                        chatVM?.apiServer.isRunning == true ? "Stop API" : "Start API",
                        systemImage: chatVM?.apiServer.isRunning == true ? "network" : "network.slash"
                    )
                    .foregroundStyle(chatVM?.apiServer.isRunning == true ? .green : .secondary)
                }
                .help(chatVM?.apiServer.isRunning == true ? "API server running on port \(Preferences.apiPort) — click to stop" : "Click to start API server")

                // New conversation
                Button {
                    chatVM?.newConversation()
                } label: {
                    Label("New Chat", systemImage: "plus.message")
                }
                .keyboardShortcut("n", modifiers: .command)
            }
        }
        // Cmd+1/2/3 model switching
        .background {
            modelSwitchShortcuts
        }
    }

    @ViewBuilder
    private var modelSwitchShortcuts: some View {
        ForEach(Array(ModelConfig.availableModels.enumerated()), id: \.element.id) { index, config in
            if index < 9 {
                Button("") {
                    Task { await modelManager.loadModel(config) }
                }
                .keyboardShortcut(KeyEquivalent(Character(String(index + 1))), modifiers: .command)
                .hidden()
            }
        }
    }
}

/// The main chat layout: messages + input area + status bar.
struct ChatView: View {
    @Bindable var viewModel: ChatViewModel

    var body: some View {
        VStack(spacing: 0) {
            ChatMessagesView(viewModel: viewModel)
            Divider()
            ChatInputView(viewModel: viewModel)
            StatusBarView(viewModel: viewModel)
        }
    }
}
