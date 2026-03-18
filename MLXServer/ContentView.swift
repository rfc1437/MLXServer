import AppKit
import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @Environment(ChatDocumentController.self) private var documentController
    @Environment(ModelManager.self) private var modelManager
    @Environment(\.openWindow) private var openWindow
    @Environment(SceneStore.self) private var sceneStore
    @State private var chatVM: ChatViewModel?
    @State private var showLoadError = false
    @State private var showMonitor = false
    @State private var showScenePicker = false
    @State private var exportDocument: ChatExportDocument?
    @State private var documentErrorMessage: String?
    @State private var exportErrorMessage: String?

    var body: some View {
        exportedContent
    }

    private var lifecycleContent: some View {
        AnyView(mainContent)
            .navigationTitle(navigationTitleText)
            .onAppear {
                if chatVM == nil {
                    chatVM = ChatViewModel(modelManager: modelManager)
                    // Auto-start API server if configured
                    if Preferences.apiAutoStart {
                        chatVM?.startAPIServer()
                    }
                }

                processPendingOpenRequests()
            }
            .onChange(of: modelManager.currentModel) {
                chatVM?.handleModelChange()
                chatVM?.markDirtyIfNeeded()
                // Persist last used model
                if let id = modelManager.currentModel?.id {
                    Preferences.lastModelId = id
                }
            }
            .onChange(of: chatVM?.inputText ?? "") {
                chatVM?.markDirtyIfNeeded()
            }
            .onChange(of: modelManager.errorMessage) {
                showLoadError = modelManager.errorMessage != nil
            }
            .onChange(of: documentController.openRequestNonce) {
                processPendingOpenRequests()
            }
    }

    private var alertContent: some View {
        AnyView(lifecycleContent)
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
            .alert("Document Error", isPresented: documentErrorBinding) {
                Button("OK", role: .cancel) {
                    documentErrorMessage = nil
                }
            } message: {
                Text(documentErrorMessage ?? "Unknown document error.")
            }
            .alert("Export Failed", isPresented: exportErrorBinding) {
                Button("OK", role: .cancel) {
                    exportErrorMessage = nil
                }
            } message: {
                Text(exportErrorMessage ?? "Unknown export error.")
            }
    }

    private var exportedContent: some View {
        AnyView(alertContent)
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
            .focusedSceneValue(\.newChatAction, NewChatAction(perform: beginNewChat))
            .focusedSceneValue(\.openChatAction, OpenChatAction(perform: beginOpenDocument))
            .focusedSceneValue(\.saveChatAction, SaveChatAction(perform: saveCurrentDocument))
            .focusedSceneValue(\.saveChatAsAction, SaveChatAsAction(perform: saveCurrentDocumentAs))
            .focusedSceneValue(\.revertChatAction, RevertChatAction(perform: beginRevertToSaved))
            .focusedSceneValue(\.exportChatAction, ExportChatAction(perform: beginExport))
            .fileExporter(
                isPresented: Binding(
                    get: { exportDocument != nil },
                    set: {
                        if !$0 {
                            exportDocument = nil
                        }
                    }
                ),
                document: exportDocument,
                contentTypes: ChatExportDocument.writableContentTypes,
                defaultFilename: exportDefaultFilename
            ) { result in
                exportDocument = nil
                if case .failure(let error) = result {
                    print("[Export] Failed: \(error.localizedDescription)")
                    exportErrorMessage = error.localizedDescription
                }
            }
    }

    private var navigationTitleText: String {
        if let title = chatVM?.windowTitle {
            return title
        }
        return modelManager.currentModel?.displayName ?? "MLX Server"
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
            beginNewChat()
        } label: {
            Label("New Chat", systemImage: "plus.message")
        }
        .keyboardShortcut("n", modifiers: .command)
        .popover(isPresented: $showScenePicker, arrowEdge: .top) {
            SceneSelectionView(
                scenes: sceneStore.scenes,
                activeSceneId: chatVM?.activeScene?.id,
                currentModelName: modelManager.currentModel?.displayName,
                onSelectNeutral: {
                    showScenePicker = false
                    startConversation(scene: nil)
                },
                onSelectScene: { scene in
                    showScenePicker = false
                    startConversation(scene: scene)
                },
                onManageScenes: {
                    showScenePicker = false
                    openWindow(id: SceneManagementWindow.windowID)
                }
            )
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

    private var exportErrorBinding: Binding<Bool> {
        Binding(
            get: { exportErrorMessage != nil },
            set: {
                if !$0 {
                    exportErrorMessage = nil
                }
            }
        )
    }

    private var exportDefaultFilename: String {
        if let currentDocumentURL = chatVM?.currentDocumentURL {
            return currentDocumentURL.deletingPathExtension().lastPathComponent
        }

        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd-HHmm"
        return "chat-\(formatter.string(from: .now))"
    }

    private func beginExport() {
        guard exportDocument == nil else { return }
        exportDocument = ChatExportDocument(
            messages: chatVM?.conversation.messages ?? [],
            modelName: modelManager.currentModel?.displayName
        )
    }

    private var documentDefaultFilename: String {
        if let currentDocumentURL = chatVM?.currentDocumentURL {
            return currentDocumentURL.deletingPathExtension().lastPathComponent
        }
        return exportDefaultFilename
    }

    private var documentErrorBinding: Binding<Bool> {
        Binding(
            get: { documentErrorMessage != nil },
            set: {
                if !$0 {
                    documentErrorMessage = nil
                }
            }
        )
    }

    private func beginNewChat() {
        showScenePicker = true
    }

    private func startConversation(scene: ChatScene?) {
        guard confirmDiscardUnsavedChanges(
            title: "Discard Unsaved Changes?",
            message: "Starting a new chat will replace the current conversation."
        ) else {
            return
        }

        Task {
            await chatVM?.startNewConversation(scene: scene)
        }
    }

    private func beginOpenDocument() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.mlxChatDocument]
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.treatsFilePackagesAsDirectories = false

        guard panel.runModal() == .OK, let url = panel.url else { return }
        Task {
            await openDocument(at: url)
        }
    }

    private func saveCurrentDocument() {
        guard let chatVM else { return }

        if let currentDocumentURL = chatVM.currentDocumentURL {
            do {
                try chatVM.saveDocument(to: currentDocumentURL)
            } catch {
                documentErrorMessage = error.localizedDescription
            }
        } else {
            saveCurrentDocumentAs()
        }
    }

    private func saveCurrentDocumentAs() {
        guard let chatVM else { return }

        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mlxChatDocument]
        panel.canCreateDirectories = true
        panel.isExtensionHidden = false
        panel.nameFieldStringValue = documentDefaultFilename

        guard panel.runModal() == .OK, let panelURL = panel.url else { return }

        let saveURL: URL
        if panelURL.pathExtension.lowercased() == "mlxchat" {
            saveURL = panelURL
        } else {
            saveURL = panelURL.appendingPathExtension("mlxchat")
        }

        do {
            try chatVM.saveDocument(to: saveURL)
        } catch {
            documentErrorMessage = error.localizedDescription
        }
    }

    private func beginRevertToSaved() {
        guard let currentDocumentURL = chatVM?.currentDocumentURL else { return }
        guard confirmDiscardUnsavedChanges(
            title: "Revert To Saved Version?",
            message: "All unsaved changes in the current chat will be lost."
        ) else {
            return
        }

        Task {
            await openDocument(at: currentDocumentURL, skipUnsavedCheck: true)
        }
    }

    private func processPendingOpenRequests() {
        guard chatVM != nil else { return }

        Task {
            while let url = documentController.consumeNextOpenRequest() {
                await openDocument(at: url)
            }
        }
    }

    private func openDocument(at url: URL, skipUnsavedCheck: Bool = false) async {
        if !skipUnsavedCheck {
            let shouldContinue = confirmDiscardUnsavedChanges(
                title: "Discard Unsaved Changes?",
                message: "Opening another chat will replace the current conversation."
            )
            guard shouldContinue else { return }
        }

        do {
            try await chatVM?.loadDocument(from: url)
        } catch {
            documentErrorMessage = error.localizedDescription
        }
    }

    private func confirmDiscardUnsavedChanges(title: String, message: String) -> Bool {
        guard chatVM?.hasUnsavedChanges == true else { return true }

        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "Discard Changes")
        alert.addButton(withTitle: "Cancel")
        return alert.runModal() == .alertFirstButtonReturn
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
