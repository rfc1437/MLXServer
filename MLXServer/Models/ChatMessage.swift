import AppKit
import CryptoKit
import Foundation
import MLXLMCommon
import UniformTypeIdentifiers

struct ChatAttachment: Identifiable, Hashable {
    let id: UUID
    let data: Data
    let mimeType: String
    let pixelWidth: Int?
    let pixelHeight: Int?
    let sha256: String

    init?(
        id: UUID = UUID(),
        data: Data,
        mimeType: String,
        pixelWidth: Int? = nil,
        pixelHeight: Int? = nil,
        sha256: String? = nil
    ) {
        guard NSImage(data: data) != nil else { return nil }

        self.id = id
        self.data = data
        self.mimeType = mimeType

        let dimensions = Self.resolveDimensions(from: data)
        self.pixelWidth = pixelWidth ?? dimensions.width
        self.pixelHeight = pixelHeight ?? dimensions.height
        self.sha256 = sha256 ?? Self.sha256Hex(for: data)
    }

    init?(id: UUID = UUID(), image: NSImage) {
        guard let tiffData = image.tiffRepresentation,
              let bitmap = NSBitmapImageRep(data: tiffData),
              let pngData = bitmap.representation(using: .png, properties: [:]) else {
            return nil
        }

        self.init(
            id: id,
            data: pngData,
            mimeType: "image/png",
            pixelWidth: bitmap.pixelsWide,
            pixelHeight: bitmap.pixelsHigh
        )
    }

    var fileExtension: String {
        UTType(mimeType: mimeType)?.preferredFilenameExtension ?? "bin"
    }

    var image: NSImage? {
        NSImage(data: data)
    }

    var userInputImage: UserInput.Image? {
        guard let image,
              let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }
        return .ciImage(CIImage(cgImage: cgImage))
    }

    private static func resolveDimensions(from data: Data) -> (width: Int?, height: Int?) {
        guard let image = NSImage(data: data),
              let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return (nil, nil)
        }

        return (cgImage.width, cgImage.height)
    }

    private static func sha256Hex(for data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}

/// A single message in the chat conversation.
struct ChatMessage: Identifiable {
    let id: UUID
    let role: Role
    var content: String
    var attachments: [ChatAttachment]
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

    init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        attachments: [ChatAttachment] = [],
        isStreaming: Bool = false,
        timestamp: Date = Date(),
        rawContent: String? = nil,
        thinkingContent: String? = nil,
        isThinking: Bool = false
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.rawContent = rawContent ?? content
        self.attachments = attachments
        self.isStreaming = isStreaming
        self.timestamp = timestamp
        self.thinkingContent = thinkingContent ?? ""
        self.isThinking = isThinking

        if role == .assistant, rawContent != nil {
            applyParsedContent(Self.parseAssistantContent(self.rawContent))
        }
    }

    var sessionContent: String {
        role == .assistant ? rawContent : content
    }

    mutating func refreshAssistantContentFromRaw() {
        applyParsedContent(Self.parseAssistantContent(rawContent))
    }

    private mutating func applyParsedContent(_ parsed: ParsedAssistantContent) {
        thinkingContent = parsed.thinking
        content = parsed.visible
        isThinking = parsed.isInThinkingBlock
    }

    private struct ParsedAssistantContent {
        let visible: String
        let thinking: String
        let isInThinkingBlock: Bool
    }

    private static func parseAssistantContent(_ raw: String) -> ParsedAssistantContent {
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

        return ParsedAssistantContent(
            visible: visible.trimmingCharacters(in: .whitespacesAndNewlines),
            thinking: thinking.trimmingCharacters(in: .whitespacesAndNewlines),
            isInThinkingBlock: isInThink
        )
    }
}

/// Observable conversation state holding all messages.
@Observable
@MainActor
final class Conversation {
    var messages: [ChatMessage] = []

    func addUserMessage(_ text: String, images: [NSImage] = []) {
        let attachments = images.compactMap { ChatAttachment(image: $0) }
        messages.append(ChatMessage(role: .user, content: text, attachments: attachments))
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
        messages[index].refreshAssistantContentFromRaw()
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

    func replaceMessages(_ restoredMessages: [ChatMessage]) {
        messages = restoredMessages
    }
}
