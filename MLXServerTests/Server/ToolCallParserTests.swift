import XCTest
@testable import MLX_Server

final class ToolCallParserTests: XCTestCase {
    func testParseGemmaToolCodeBlockExtractsToolCallAndStripsFence() throws {
        let tools = [mockWeatherTool]
        let text = "Before\n```tool_code\nweather(city=\"Berlin\")\n```\nAfter"

        let parsed = ToolCallParser.parse(text: text, tools: tools)

        XCTAssertEqual(parsed.0, "Before\n\nAfter")
        let toolCall = try XCTUnwrap(parsed.1.first)
        XCTAssertEqual(toolCall.name, "weather")
        XCTAssertEqual(toolCall.arguments, #"{"city":"Berlin"}"#)
    }

    func testParseQwenToolCallTagExtractsJSONPayloadAndStripsTag() throws {
        let text = "<tool_call>{\"name\":\"weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call>"

        let parsed = ToolCallParser.parse(text: text, tools: [mockWeatherTool])

        XCTAssertEqual(parsed.0, "")
        let toolCall = try XCTUnwrap(parsed.1.first)
        XCTAssertEqual(toolCall.name, "weather")
        XCTAssertEqual(toolCall.arguments, #"{"city":"Paris"}"#)
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
