import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(ModelManager.self) private var modelManager
    @State private var chatVM: ChatViewModel?
    @State private var showLoadError = false
    @State private var showMonitor = false
    @State private var isExporting = false

    var body: some View {
        mainContent
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
                    toolbarButtons
                }
            }
            // Cmd+1/2/3 model switching
            .background {
                modelSwitchShortcuts
            }
            // Expose export trigger to menu bar command
            .focusedSceneValue(\.exportTrigger, $isExporting)
            .fileExporter(
                isPresented: $isExporting,
                document: ChatExportDocument(
                    messages: chatVM?.conversation.messages ?? [],
                    modelName: modelManager.currentModel?.displayName
                ),
                contentTypes: ChatExportDocument.writableContentTypes,
                defaultFilename: "chat"
            ) { result in
                if case .failure(let error) = result {
                    print("[Export] Failed: \(error.localizedDescription)")
                }
            }
    }

    @ViewBuilder
    private var mainContent: some View {
        ZStack {
            if let chatVM {
                if showMonitor {
                    MonitorView(stats: chatVM.apiServer.inferenceStats)
                } else {
                    ChatView(viewModel: chatVM)
                }
            } else {
                ProgressView("Initializing…")
            }

            // Download modal overlay
            if modelManager.isDownloading {
                Color.black.opacity(0.3)
                    .ignoresSafeArea()
                DownloadModalView()
            }
        }
    }

    @ViewBuilder
    private var toolbarButtons: some View {
        // API server toggle
        let isRunning = chatVM?.apiServer.isRunning == true
        Button {
            if let chatVM {
                if chatVM.apiServer.isRunning {
                    chatVM.stopAPIServer()
                } else {
                    chatVM.startAPIServer()
                }
            }
        } label: {
            Label(
                isRunning ? "Stop API" : "Start API",
                systemImage: isRunning ? "network" : "network.slash"
            )
            .foregroundStyle(isRunning ? .green : .secondary)
        }
        .help(isRunning ? "API server running on port \(Preferences.apiPort) — click to stop" : "Click to start API server")

        // Monitor toggle
        Button {
            showMonitor.toggle()
        } label: {
            Label(
                showMonitor ? "Chat" : "Monitor",
                systemImage: showMonitor ? "bubble.left.and.text.bubble.right" : "chart.xyaxis.line"
            )
            .foregroundStyle(showMonitor ? Color.accentColor : Color.secondary)
        }
        .help(showMonitor ? "Switch to chat" : "Show inference monitor")
        .keyboardShortcut("m", modifiers: [.command, .shift])

        // New conversation
        Button {
            chatVM?.newConversation()
        } label: {
            Label("New Chat", systemImage: "plus.message")
        }
        .keyboardShortcut("n", modifiers: .command)
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
