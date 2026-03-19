import AppKit
import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import MLXVLM

/// Drives the chat UI: sending messages, streaming responses, managing images.
@Observable
@MainActor
final class ChatViewModel {
    var conversation = Conversation()
    var inputText = ""
    var attachedImages: [NSImage] = []
    var activeScene: ChatScene?
    var currentDocumentURL: URL?
    var hasUnsavedChanges = false
    var isGenerating = false
    var tokensPerSecond: Double = 0
    var promptTokens: Int = 0
    var generationTokens: Int = 0

    private(set) var lastSavedSnapshotHash: String?

    private var generationTask: Task<Void, Never>?
    private var autosaveTask: Task<Void, Never>?
    private var chatSession: ChatSession?
    private var documentId = UUID()
    private var documentCreatedAt = Date()
    private var documentSystemPromptOverride: String?
    private var documentThinkingOverride: Bool?
    private var documentTemperature = 0.7

    let modelManager: ModelManager
    let apiServer = APIServer()

    init(modelManager: ModelManager) {
        self.modelManager = modelManager
    }

    var activeSceneName: String {
        activeScene?.displayName ?? "Neutral"
    }

    var documentDisplayName: String {
        currentDocumentURL?.deletingPathExtension().lastPathComponent ?? "Untitled Chat"
    }

    var windowTitle: String {
        hasUnsavedChanges ? "\(documentDisplayName) *" : documentDisplayName
    }

    /// Ensure a ChatSession exists for the current model.
    private func ensureSession() {
        guard let container = modelManager.modelContainer else { return }
        if chatSession == nil {
            let systemPrompt = effectiveSystemPrompt
            // Pass enable_thinking to the Jinja chat template context.
            // Qwen3.5 and similar models use this to control reasoning mode.
            let thinkingContext: [String: any Sendable]? = effectiveThinkingEnabled
                ? nil
                : ["enable_thinking": false]
            let generateParameters = GenerateParameters(temperature: Float(documentTemperature))
            let history = conversation.messages.compactMap(historyMessage(from:))
            if history.isEmpty {
                chatSession = ChatSession(
                    container,
                    instructions: systemPrompt.isEmpty ? nil : systemPrompt,
                    generateParameters: generateParameters,
                    additionalContext: thinkingContext
                )
            } else {
                chatSession = ChatSession(
                    container,
                    instructions: systemPrompt.isEmpty ? nil : systemPrompt,
                    history: history,
                    generateParameters: generateParameters,
                    additionalContext: thinkingContext
                )
            }
        }
    }

    private var effectiveSystemPrompt: String {
        if let documentSystemPromptOverride {
            return documentSystemPromptOverride
        }

        let parts = [
            Preferences.systemPrompt,
            activeScene?.systemPrompt ?? ""
        ]
        .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }

        return parts.joined(separator: "\n\n")
    }

    private var effectiveThinkingEnabled: Bool {
        documentThinkingOverride ?? Preferences.enableThinking
    }

    func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, modelManager.isReady else { return }

        modelManager.touchActivity()
        ensureSession()
        guard let session = chatSession else { return }

        let images = modelManager.currentModel?.supportsImages == true ? attachedImages : []
        inputText = ""
        attachedImages = []

        conversation.addUserMessage(text, images: images)
        markDirtyIfNeeded()
        let assistantIndex = conversation.addAssistantMessage()

        isGenerating = true
        tokensPerSecond = 0
        promptTokens = 0
        generationTokens = 0

        // Convert NSImages to UserInput.Image
        let inputImages: [UserInput.Image] = images.compactMap { nsImage in
            guard let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
                return nil
            }
            return .ciImage(CIImage(cgImage: cgImage))
        }

        generationTask = Task {
            do {
                let stream = session.streamDetails(
                    to: text,
                    images: inputImages,
                    videos: []
                )

                var tokenCount = 0
                let startTime = Date()

                for try await generation in stream {
                    if Task.isCancelled { break }

                    switch generation {
                    case .chunk(let text):
                        conversation.appendToMessage(at: assistantIndex, chunk: text)
                        tokenCount += 1
                        let elapsed = Date().timeIntervalSince(startTime)
                        if elapsed > 0 {
                            tokensPerSecond = Double(tokenCount) / elapsed
                        }
                        generationTokens = tokenCount

                    case .info(let info):
                        promptTokens = info.promptTokenCount
                        if info.tokensPerSecond > 0 {
                            tokensPerSecond = info.tokensPerSecond
                        }

                    case .toolCall:
                        break
                    }
                }
            } catch {
                if !Task.isCancelled {
                    conversation.appendToMessage(
                        at: assistantIndex,
                        chunk: "\n\n[Error: \(error.localizedDescription)]"
                    )
                }
            }

            conversation.finalizeMessage(at: assistantIndex)
            markDirtyIfNeeded()
            isGenerating = false
            generationTask = nil
            modelManager.touchActivity()
        }
    }

    func stop() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false

        if let last = conversation.messages.indices.last,
           conversation.messages[last].isStreaming {
            conversation.finalizeMessage(at: last)
            markDirtyIfNeeded()
        }
    }

    func attachImage(_ image: NSImage) {
        guard modelManager.currentModel?.supportsImages == true else { return }
        attachedImages.append(image)
    }

    func removeImage(at index: Int) {
        guard attachedImages.indices.contains(index) else { return }
        attachedImages.remove(at: index)
    }

    func newConversation() {
        stop()
        conversation.clear()
        inputText = ""
        attachedImages = []
        activeScene = nil
        resetSession()
        resetDocumentState()
        Preferences.lastSceneId = nil
        scheduleAutosaveIfNeeded()
    }

    func startNewConversation(scene: ChatScene?) async {
        stop()

        if let config = scene?.resolvedModel,
           modelManager.currentModel?.id != config.id {
            await modelManager.loadModel(config)
        }

        conversation.clear()
        activeScene = scene
        inputText = scene?.starterPrompt.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        attachedImages = []
        resetSession()
        resetDocumentState()
        Preferences.lastSceneId = scene?.id

        markDirtyIfNeeded()

        if !inputText.isEmpty {
            send()
        }
    }

    /// Reset the chat session (e.g. on model switch or new conversation).
    func resetSession() {
        chatSession = nil
    }

    func handleModelChange() {
        resetSession()
        if modelManager.currentModel?.supportsImages != true {
            attachedImages = []
        }
    }

    func loadDocument(from url: URL) async throws {
        autosaveTask?.cancel()

        let package = try ChatDocumentPackage(contentsOf: url)
        let restoredMessages = try package.manifest.messages.map { storedMessage in
            try restoreMessage(storedMessage, attachmentContents: package.attachmentContents)
        }

        stop()
        conversation.replaceMessages(restoredMessages)
        inputText = package.manifest.uiState.draftInput
        attachedImages = []
        activeScene = nil
        currentDocumentURL = url
        documentId = package.manifest.documentId
        documentCreatedAt = package.manifest.createdAt
        documentSystemPromptOverride = package.manifest.settings.systemPrompt
        documentThinkingOverride = package.manifest.settings.thinkingEnabled
        documentTemperature = package.manifest.settings.temperature
        resetSession()
        lastSavedSnapshotHash = try snapshotHash()
        hasUnsavedChanges = false
        Preferences.lastSceneId = nil

        if let storedModel = package.manifest.model,
           let config = ModelConfig.resolve(storedModel.id) ?? ModelConfig.resolve(storedModel.repoId),
           modelManager.currentModel?.id != config.id {
            await modelManager.loadModel(config)
        }
    }

    func saveDocument(to url: URL) throws {
        guard !isGenerating else {
            throw ChatDocumentError.saveWhileGenerating
        }

        autosaveTask?.cancel()
        let package = try makeDocumentPackage(updatedAt: Date())
        try package.write(to: url)
        currentDocumentURL = url
        lastSavedSnapshotHash = try snapshotHash()
        hasUnsavedChanges = false
        Self.removeAutosave()
    }

    func markDirtyIfNeeded() {
        if let lastSavedSnapshotHash {
            hasUnsavedChanges = (try? snapshotHash()) != lastSavedSnapshotHash
        } else {
            hasUnsavedChanges = !conversation.messages.isEmpty
                || !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || activeScene != nil
        }

            scheduleAutosaveIfNeeded()
    }

    private func resetDocumentState() {
        currentDocumentURL = nil
        hasUnsavedChanges = false
        lastSavedSnapshotHash = nil
        documentId = UUID()
        documentCreatedAt = Date()
        documentSystemPromptOverride = nil
        documentThinkingOverride = nil
        documentTemperature = 0.7
    }

    private func restoreMessage(
        _ storedMessage: ChatDocumentManifest.StoredChatMessage,
        attachmentContents: [String: Data]
    ) throws -> ChatMessage {
        let attachments = try storedMessage.attachments.map { attachment in
            guard let data = attachmentContents[attachment.relativePath] else {
                throw ChatDocumentError.missingAttachment(attachment.relativePath)
            }
            guard let restoredAttachment = ChatAttachment(
                id: attachment.id,
                data: data,
                mimeType: attachment.mimeType,
                pixelWidth: attachment.pixelWidth,
                pixelHeight: attachment.pixelHeight,
                sha256: attachment.sha256
            ) else {
                throw ChatDocumentError.invalidAttachmentData(attachment.relativePath)
            }
            return restoredAttachment
        }

        return ChatMessage(
            id: storedMessage.id,
            role: ChatMessage.Role(rawValue: storedMessage.role.rawValue) ?? .assistant,
            content: storedMessage.content,
            attachments: attachments,
            isStreaming: storedMessage.streamingState == .streaming,
            timestamp: storedMessage.createdAt,
            rawContent: storedMessage.rawContent,
            thinkingContent: storedMessage.thinkingContent,
            isThinking: storedMessage.streamingState == .streaming
        )
    }

    private func makeDocumentPackage(updatedAt: Date) throws -> ChatDocumentPackage {
        let manifest = makeManifest(updatedAt: updatedAt)
        var attachmentContents: [String: Data] = [:]

        for message in conversation.messages {
            for attachment in message.attachments {
                attachmentContents[attachmentRelativePath(for: attachment)] = attachment.data
            }
        }

        return ChatDocumentPackage(manifest: manifest, attachmentContents: attachmentContents)
    }

    private func makeManifest(updatedAt: Date) -> ChatDocumentManifest {
        let messages = conversation.messages.map { message in
            ChatDocumentManifest.StoredChatMessage(
                id: message.id,
                role: ChatDocumentManifest.StoredChatMessage.Role(rawValue: message.role.rawValue) ?? .assistant,
                createdAt: message.timestamp,
                content: message.content,
                rawContent: message.rawContent,
                thinkingContent: message.thinkingContent,
                streamingState: message.isStreaming ? .streaming : .completed,
                attachments: message.attachments.map { attachment in
                    ChatDocumentManifest.StoredAttachment(
                        id: attachment.id,
                        type: "image",
                        relativePath: attachmentRelativePath(for: attachment),
                        mimeType: attachment.mimeType,
                        pixelWidth: attachment.pixelWidth,
                        pixelHeight: attachment.pixelHeight,
                        sha256: attachment.sha256
                    )
                }
            )
        }

        return ChatDocumentManifest(
            schemaVersion: ChatDocumentManifest.currentSchemaVersion,
            documentId: documentId,
            createdAt: documentCreatedAt,
            updatedAt: updatedAt,
            appVersion: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0.0",
            model: currentStoredModelInfo,
            settings: .init(
                systemPrompt: effectiveSystemPrompt,
                thinkingEnabled: effectiveThinkingEnabled,
                temperature: documentTemperature
            ),
            messages: messages,
            uiState: .init(
                draftInput: inputText,
                scrollAnchorMessageId: conversation.messages.last?.id
            )
        )
    }

    private var currentStoredModelInfo: ChatDocumentManifest.StoredModelInfo? {
        guard let model = modelManager.currentModel else { return nil }
        return .init(id: model.id, displayName: model.displayName, repoId: model.repoId)
    }

    private func attachmentRelativePath(for attachment: ChatAttachment) -> String {
        "attachments/\(attachment.id.uuidString).\(attachment.fileExtension)"
    }

    private func historyMessage(from message: ChatMessage) -> Chat.Message? {
        let role: Chat.Message.Role
        switch message.role {
        case .assistant:
            role = .assistant
        case .system:
            return nil
        case .user:
            role = .user
        }

        return Chat.Message(
            role: role,
            content: message.sessionContent,
            images: message.attachments.compactMap(\.userInputImage)
        )
    }

    private func snapshotHash() throws -> String {
        let snapshot = ChatDocumentSnapshot(
            documentId: documentId,
            createdAt: documentCreatedAt,
            model: currentStoredModelInfo,
            settings: .init(
                systemPrompt: effectiveSystemPrompt,
                thinkingEnabled: effectiveThinkingEnabled,
                temperature: documentTemperature
            ),
            messages: makeManifest(updatedAt: documentCreatedAt).messages,
            uiState: .init(draftInput: inputText, scrollAnchorMessageId: conversation.messages.last?.id)
        )
        let data = try Self.snapshotEncoder.encode(snapshot)
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    private static var snapshotEncoder: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    // MARK: - Autosave / Restore

    /// Location for the automatic session save inside the sandbox container.
    static var autosaveURL: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = appSupport.appendingPathComponent("MLXServer", isDirectory: true)
        return dir.appendingPathComponent("autosave.mlxchat")
    }

    static var hasAutosavedSession: Bool {
        FileManager.default.fileExists(atPath: autosaveURL.path)
    }

    /// Persist the current session so it survives a quit.
    func autosaveToSandbox() {
        autosaveTask?.cancel()

        guard currentDocumentURL == nil else {
            Self.removeAutosave()
            return
        }

        // Nothing to save if conversation is empty and no draft text
        let hasDraft = !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        guard !conversation.messages.isEmpty || hasDraft else {
            // Remove stale autosave if conversation was cleared
            Self.removeAutosave()
            return
        }

        do {
            let url = Self.autosaveURL
            try FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true,
                attributes: nil
            )
            let package = try makeDocumentPackage(updatedAt: Date())
            try package.write(to: url)
        } catch {
            print("[Autosave] Failed: \(error.localizedDescription)")
        }
    }

    /// Restore a previously autosaved session. Returns true if restored.
    func restoreFromAutosave() async -> Bool {
        let url = Self.autosaveURL
        guard FileManager.default.fileExists(atPath: url.path) else { return false }

        do {
            try await loadDocument(from: url)
            // Clear document URL so this doesn't look like a user-saved file
            currentDocumentURL = nil
            hasUnsavedChanges = false
            lastSavedSnapshotHash = nil

            if modelManager.currentModel == nil {
                let modelId = Preferences.defaultModelId ?? Preferences.lastModelId ?? ModelConfig.default.id
                if let config = ModelConfig.availableModels.first(where: { $0.id == modelId }) {
                    await modelManager.loadModel(config)
                }
            }

            return true
        } catch {
            print("[Autosave] Restore failed: \(error.localizedDescription)")
            return false
        }
    }

    static func removeAutosave() {
        let url = autosaveURL
        try? FileManager.default.removeItem(at: url)
    }

    private func scheduleAutosaveIfNeeded() {
        autosaveTask?.cancel()

        guard currentDocumentURL == nil else {
            Self.removeAutosave()
            return
        }

        let hasDraft = !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        guard !conversation.messages.isEmpty || hasDraft || activeScene != nil else {
            Self.removeAutosave()
            return
        }

        autosaveTask = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(800))
            guard !Task.isCancelled else { return }
            await self?.autosaveToSandbox()
        }
    }

    // MARK: - API Server

    func startAPIServer() {
        apiServer.start(modelManager: modelManager, port: Preferences.apiPort)
    }

    func stopAPIServer() {
        apiServer.stop()
    }
}
