import Foundation

/// Pre-computes static JSON parts for SSE streaming.
/// Only the dynamic delta payload is serialized per token.
struct StreamingSSEEncoder: Sendable {
    private let requestId: String
    private let created: Int
    private let modelName: String

    init(requestId: String, created: Int, modelName: String) {
        self.requestId = requestId
        self.created = created
        self.modelName = modelName
    }

    func encodeContentDelta(_ text: String) -> Data {
        Self.encodeChunk(
            APIChatCompletionChunk(
                id: requestId,
                object: "chat.completion.chunk",
                created: created,
                model: modelName,
                choices: [
                    APIStreamChoice(
                        index: 0,
                        delta: APIDeltaMessage(role: nil, content: text, tool_calls: nil),
                        finish_reason: nil
                    )
                ],
                usage: nil
            )
        )
    }

    func encodeRoleDelta(_ role: String) -> Data {
        Self.encodeChunk(
            APIChatCompletionChunk(
                id: requestId,
                object: "chat.completion.chunk",
                created: created,
                model: modelName,
                choices: [
                    APIStreamChoice(
                        index: 0,
                        delta: APIDeltaMessage(role: role, content: nil, tool_calls: nil),
                        finish_reason: nil
                    )
                ],
                usage: nil
            )
        )
    }

    static func encodeFinalChunk(_ chunk: APIChatCompletionChunk) -> Data {
        encodeChunk(chunk)
    }

    private static func encodeChunk(_ chunk: APIChatCompletionChunk) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]

        guard let json = try? encoder.encode(chunk) else {
            return Data("data: {}\n\n".utf8)
        }

        var data = Data(capacity: json.count + 8)
        data.append(Data("data: ".utf8))
        data.append(json)
        data.append(Data("\n\n".utf8))
        return data
    }
}