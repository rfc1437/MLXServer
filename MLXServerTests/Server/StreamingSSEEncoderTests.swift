import XCTest
@testable import MLX_Server

final class StreamingSSEEncoderTests: XCTestCase {
    func testEncodeContentDeltaMatchesJSONEncoderOutput() throws {
        let encoder = StreamingSSEEncoder(requestId: "chatcmpl-test", created: 1_234_567, modelName: "qwen\"model")
        let text = "line 1\nline 2\t\"quoted\"\\slash"

        let actual = encoder.encodeContentDelta(text)
        let expected = try baselineData(
            for: APIChatCompletionChunk(
                id: "chatcmpl-test",
                object: "chat.completion.chunk",
                created: 1_234_567,
                model: "qwen\"model",
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

        XCTAssertEqual(actual, expected)
    }

    func testEncodeRoleDeltaMatchesJSONEncoderOutput() throws {
        let encoder = StreamingSSEEncoder(requestId: "chatcmpl-role", created: 99, modelName: "gemma")

        let actual = encoder.encodeRoleDelta("assistant")
        let expected = try baselineData(
            for: APIChatCompletionChunk(
                id: "chatcmpl-role",
                object: "chat.completion.chunk",
                created: 99,
                model: "gemma",
                choices: [
                    APIStreamChoice(
                        index: 0,
                        delta: APIDeltaMessage(role: "assistant", content: nil, tool_calls: nil),
                        finish_reason: nil
                    )
                ],
                usage: nil
            )
        )

        XCTAssertEqual(actual, expected)
    }

    func testEncodeFinalChunkMatchesBaseline() throws {
        let chunk = APIChatCompletionChunk(
            id: "chatcmpl-final",
            object: "chat.completion.chunk",
            created: 7,
            model: "gemma",
            choices: [
                APIStreamChoice(
                    index: 0,
                    delta: APIDeltaMessage(role: nil, content: nil, tool_calls: nil),
                    finish_reason: "stop"
                )
            ],
            usage: APIUsageInfo(prompt_tokens: 10, completion_tokens: 3, total_tokens: 13)
        )

        XCTAssertEqual(StreamingSSEEncoder.encodeFinalChunk(chunk), try baselineData(for: chunk))
    }

    private func baselineData(for chunk: APIChatCompletionChunk) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let json = try encoder.encode(chunk)
        var data = Data("data: ".utf8)
        data.append(json)
        data.append(Data("\n\n".utf8))
        return data
    }
}