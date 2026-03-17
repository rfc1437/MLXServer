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

    enum Role: String {
        case system
        case user
        case assistant
    }

    init(role: Role, content: String, images: [NSImage] = [], isStreaming: Bool = false) {
        self.role = role
        self.content = content
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
    func appendToMessage(at index: Int, chunk: String) {
        guard index < messages.count else { return }
        messages[index].content += chunk
    }

    /// Marks the assistant message at the given index as done streaming.
    func finalizeMessage(at index: Int) {
        guard index < messages.count else { return }
        messages[index].isStreaming = false
    }

    func clear() {
        messages.removeAll()
    }
}
