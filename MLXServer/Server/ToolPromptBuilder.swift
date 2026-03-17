import Foundation

/// Builds model-specific system prompts that inform the model about available tools.
/// Mirrors the Python server's `_build_tool_system_prompt()` and `_build_qwen_tool_system_prompt()`.
enum ToolPromptBuilder {

    /// Build a tool system prompt appropriate for the current model.
    /// - Parameters:
    ///   - tools: OpenAI-format tool definitions
    ///   - modelId: The model's repo ID (to determine format)
    /// - Returns: A system prompt string describing the available tools
    static func buildSystemPrompt(tools: [APIToolDefinition], modelId: String) -> String {
        if modelId.lowercased().contains("qwen") {
            return buildQwenToolPrompt(tools: tools)
        } else {
            return buildGemmaToolPrompt(tools: tools)
        }
    }

    // MARK: - Gemma (tool_code format)

    /// Build the tool system prompt using Google's Gemma 3 tool_code convention.
    private static func buildGemmaToolPrompt(tools: [APIToolDefinition]) -> String {
        let funcDefs = tools.map { toolToPythonSignature($0.function) }
        let functionsBlock = funcDefs.joined(separator: "\n\n")

        return """
            At each turn, if you decide to invoke any of the function(s), \
            it should be wrapped with ```tool_code```. \
            The python methods described below are imported and available, \
            you can only use defined methods. \
            The generated code should be readable and efficient. \
            The response to a method will be wrapped in ```tool_output``` \
            use it to call more tools or generate a helpful, friendly response.

            \(functionsBlock)
            """
    }

    /// Convert an OpenAI function definition to a Python function signature with docstring.
    private static func toolToPythonSignature(_ func: APIFunctionDefinition) -> String {
        let name = `func`.name
        let desc = `func`.description ?? ""
        let properties = `func`.parameters?["properties"]?.value as? [String: Any] ?? [:]
        let requiredArr = `func`.parameters?["required"]?.value as? [String] ?? []
        let required = Set(requiredArr)

        var paramParts: [String] = []
        var docArgs: [String] = []

        // Sort keys for deterministic output
        for pname in properties.keys.sorted() {
            guard let pinfo = properties[pname] as? [String: Any] else { continue }
            let ptype = jsonTypeToPython(pinfo["type"] as? String ?? "str")
            let pdesc = pinfo["description"] as? String ?? ""

            if required.contains(pname) {
                paramParts.append("\(pname): \(ptype)")
            } else {
                let defaultVal = jsonTypeDefault(pinfo["type"] as? String ?? "str")
                paramParts.append("\(pname): \(ptype) = \(defaultVal)")
            }
            docArgs.append(pdesc.isEmpty ? "      \(pname)" : "      \(pname): \(pdesc)")
        }

        let sig = "def \(name)(\(paramParts.joined(separator: ", "))):"
        var docLines = ["    \"\"\"\(desc)"]
        if !docArgs.isEmpty {
            docLines.append("")
            docLines.append("    Args:")
            docLines.append(contentsOf: docArgs)
        }
        docLines.append("    \"\"\"")

        return sig + "\n" + docLines.joined(separator: "\n")
    }

    private static func jsonTypeToPython(_ type: String) -> String {
        switch type {
        case "string": return "str"
        case "integer": return "int"
        case "number": return "float"
        case "boolean": return "bool"
        case "array": return "list"
        case "object": return "dict"
        default: return "str"
        }
    }

    private static func jsonTypeDefault(_ type: String) -> String {
        switch type {
        case "string": return "None"
        case "integer": return "0"
        case "number": return "0.0"
        case "boolean": return "False"
        case "array": return "[]"
        case "object": return "{}"
        default: return "None"
        }
    }

    // MARK: - Qwen (<tool_call> format)

    /// Build the tool system prompt for Qwen3 using its native JSON format.
    private static func buildQwenToolPrompt(tools: [APIToolDefinition]) -> String {
        var toolDescs: [[String: Any]] = []
        for tool in tools {
            var funcDict: [String: Any] = [
                "name": tool.function.name,
                "description": tool.function.description ?? "",
            ]
            if let params = tool.function.parameters {
                funcDict["parameters"] = params.mapValues(\.value)
            }
            toolDescs.append([
                "type": "function",
                "function": funcDict,
            ])
        }

        let toolsJSON: String
        if let data = try? JSONSerialization.data(withJSONObject: toolDescs, options: [.prettyPrinted, .sortedKeys]),
           let str = String(data: data, encoding: .utf8) {
            toolsJSON = str
        } else {
            toolsJSON = "[]"
        }

        return """
            # Tools

            You are a helpful assistant with access to the following tools. \
            Use them when appropriate by responding with a JSON tool call.

            ## Available Tools

            \(toolsJSON)

            ## Tool Call Format

            When you need to call a tool, respond with:
            <tool_call>
            {"name": "<function_name>", "arguments": {<args>}}
            </tool_call>
            """
    }

    // MARK: - Format tool calls back into model-specific format for prompt history

    /// Format OpenAI-style tool calls back into Gemma's tool_code blocks for prompt history.
    static func formatGemmaToolCalls(_ toolCalls: [APIToolCall]) -> String {
        var parts: [String] = []
        for tc in toolCalls {
            let name = tc.function.name
            let argsStr = tc.function.arguments
            if let data = argsStr.data(using: .utf8),
               let args = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                let argParts = args.keys.sorted().map { key -> String in
                    let val = args[key]!
                    return "\(key)=\(pythonRepr(val))"
                }
                let callStr = "\(name)(\(argParts.joined(separator: ", ")))"
                parts.append("```tool_code\n\(callStr)\n```")
            } else {
                parts.append("```tool_code\n\(name)()\n```")
            }
        }
        return parts.joined(separator: "\n")
    }

    /// Format OpenAI-style tool calls back into Qwen's <tool_call> tags for prompt history.
    static func formatQwenToolCalls(_ toolCalls: [APIToolCall]) -> String {
        var parts: [String] = []
        for tc in toolCalls {
            let name = tc.function.name
            let argsStr = tc.function.arguments
            var callObj: [String: Any] = ["name": name]
            if let data = argsStr.data(using: .utf8),
               let args = try? JSONSerialization.jsonObject(with: data) {
                callObj["arguments"] = args
            }
            if let data = try? JSONSerialization.data(withJSONObject: callObj),
               let str = String(data: data, encoding: .utf8) {
                parts.append("<tool_call>\n\(str)\n</tool_call>")
            }
        }
        return parts.joined(separator: "\n")
    }

    private static func pythonRepr(_ value: Any) -> String {
        switch value {
        case let s as String: return "\"\(s)\""
        case let i as Int: return "\(i)"
        case let d as Double: return "\(d)"
        case let b as Bool: return b ? "True" : "False"
        default: return "\"\(value)\""
        }
    }
}
