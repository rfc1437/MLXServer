import AppKit
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
    var isGenerating = false
    var tokensPerSecond: Double = 0
    var promptTokens: Int = 0
    var generationTokens: Int = 0

    private var generationTask: Task<Void, Never>?
    private var chatSession: ChatSession?

    let modelManager: ModelManager
    let apiServer = APIServer()

    init(modelManager: ModelManager) {
        self.modelManager = modelManager
    }

    /// Ensure a ChatSession exists for the current model.
    private func ensureSession() {
        guard let container = modelManager.modelContainer else { return }
        if chatSession == nil {
            let systemPrompt = Preferences.systemPrompt
            // Pass enable_thinking to the Jinja chat template context.
            // Qwen3.5 and similar models use this to control reasoning mode.
            let thinkingContext: [String: any Sendable]? = Preferences.enableThinking
                ? nil
                : ["enable_thinking": false]
            chatSession = ChatSession(
                container,
                instructions: systemPrompt.isEmpty ? nil : systemPrompt,
                generateParameters: GenerateParameters(temperature: 0.7),
                additionalContext: thinkingContext
            )
        }
    }

    func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, modelManager.isReady else { return }

        modelManager.touchActivity()
        ensureSession()
        guard let session = chatSession else { return }

        let images = attachedImages
        inputText = ""
        attachedImages = []

        conversation.addUserMessage(text, images: images)
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
        }
    }

    func attachImage(_ image: NSImage) {
        attachedImages.append(image)
    }

    func removeImage(at index: Int) {
        guard attachedImages.indices.contains(index) else { return }
        attachedImages.remove(at: index)
    }

    func newConversation() {
        stop()
        conversation.clear()
        resetSession()
    }

    /// Reset the chat session (e.g. on model switch or new conversation).
    func resetSession() {
        chatSession = nil
    }

    // MARK: - API Server

    func startAPIServer() {
        apiServer.start(modelManager: modelManager, port: Preferences.apiPort)
    }

    func stopAPIServer() {
        apiServer.stop()
    }
}
