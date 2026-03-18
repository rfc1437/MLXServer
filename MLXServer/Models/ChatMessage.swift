import AppKit
import Foundation

/// A single message in the chat conversation.
struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    var content: String
    var images: [NSImage]
    var isStreaming: Bool
    let timestamp: Date

    /// Raw streamed text including <think> tags (only for assistant messages).
    /// `content` and `thinkingContent` are derived from this.
    var rawContent: String = ""

    /// The thinking/reasoning content extracted from <think>...</think> tags.
    var thinkingContent: String = ""

    /// Whether the model is currently in a thinking block.
    var isThinking: Bool = false

    enum Role: String {
        case system
        case user
        case assistant
    }

    init(role: Role, content: String, images: [NSImage] = [], isStreaming: Bool = false) {
        self.role = role
        self.content = content
        self.rawContent = content
        self.images = images
        self.isStreaming = isStreaming
        self.timestamp = Date()
    }
}

/// Observable conversation state holding all messages.
@Observable
@MainActor
final class Conversation {
    var messages: [ChatMessage] = []

    func addUserMessage(_ text: String, images: [NSImage] = []) {
        messages.append(ChatMessage(role: .user, content: text, images: images))
    }

    /// Adds an empty assistant message (to be filled via streaming) and returns its index.
    func addAssistantMessage() -> Int {
        let msg = ChatMessage(role: .assistant, content: "", isStreaming: true)
        messages.append(msg)
        return messages.count - 1
    }

    /// Appends a text chunk to the assistant message at the given index.
    /// Handles `<think>...</think>` tags by routing content to `thinkingContent` vs `content`.
    func appendToMessage(at index: Int, chunk: String) {
        guard index < messages.count else { return }
        messages[index].rawContent += chunk

        // Parse the full raw content to separate thinking from response.
        // This is simpler and more robust than incremental parsing since
        // tag boundaries can split across chunks.
        let raw = messages[index].rawContent
        var thinking = ""
        var visible = ""
        var isInThink = false

        var scanner = raw[raw.startIndex...]
        while !scanner.isEmpty {
            if isInThink {
                if let endRange = scanner.range(of: "</think>") {
                    thinking += String(scanner[scanner.startIndex..<endRange.lowerBound])
                    scanner = scanner[endRange.upperBound...]
                    isInThink = false
                } else {
                    // Still inside thinking — all remaining text is thinking
                    thinking += String(scanner)
                    break
                }
            } else {
                if let startRange = scanner.range(of: "<think>") {
                    visible += String(scanner[scanner.startIndex..<startRange.lowerBound])
                    scanner = scanner[startRange.upperBound...]
                    isInThink = true
                } else {
                    visible += String(scanner)
                    break
                }
            }
        }

        messages[index].thinkingContent = thinking.trimmingCharacters(in: .whitespacesAndNewlines)
        messages[index].content = visible.trimmingCharacters(in: .whitespacesAndNewlines)
        messages[index].isThinking = isInThink
    }

    /// Marks the assistant message at the given index as done streaming.
    func finalizeMessage(at index: Int) {
        guard index < messages.count else { return }
        messages[index].isStreaming = false
        messages[index].isThinking = false
    }

    func clear() {
        messages.removeAll()
    }
}
