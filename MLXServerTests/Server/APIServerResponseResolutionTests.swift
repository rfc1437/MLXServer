import MLXLMCommon
import XCTest
@testable import MLX_Server

final class APIServerResponseResolutionTests: XCTestCase {
    @MainActor
    func testResolveAssistantResponseUsesFrameworkToolCalls() throws {
        let frameworkToolCalls = [
            ToolCall(function: ToolCall.Function(name: "weather", arguments: ["city": "Berlin"]))
        ]

        let resolved = APIServer.resolveAssistantResponse(
            fullText: "I will call the tool.",
            frameworkToolCalls: frameworkToolCalls,
            tools: [mockWeatherTool]
        )

        XCTAssertEqual(resolved.finishReason, "tool_calls")
        XCTAssertEqual(resolved.content, "I will call the tool.")
        let toolCall = try XCTUnwrap(resolved.toolCalls?.first)
        XCTAssertEqual(toolCall.function.name, "weather")
        XCTAssertEqual(toolCall.function.arguments, #"{"city":"Berlin"}"#)
    }

    private var mockWeatherTool: APIToolDefinition {
        APIToolDefinition(
            type: "function",
            function: APIFunctionDefinition(
                name: "weather",
                description: "Look up weather for a city.",
                parameters: [
                    "type": AnyCodable("object"),
                    "properties": AnyCodable([
                        "city": [
                            "type": "string",
                            "description": "City name"
                        ]
                    ]),
                    "required": AnyCodable(["city"])
                ]
            )
        )
    }
}
