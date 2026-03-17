import Foundation

/// Parses tool calls from model output text.
/// Supports both Gemma's ```tool_code``` blocks and Qwen's <tool_call> XML tags.
enum ToolCallParser {

    struct ParsedToolCall {
        let id: String
        let name: String
        let arguments: String // JSON string
    }

    /// Parse tool calls from model output. Returns (cleanText, toolCalls).
    static func parse(text: String, tools: [APIToolDefinition]?) -> (String, [ParsedToolCall]) {
        // Try Qwen-style first (<tool_call> tags)
        let (qwenClean, qwenCalls) = parseQwen(text: text)
        if !qwenCalls.isEmpty {
            return (qwenClean, qwenCalls)
        }

        // Try Gemma-style (```tool_code``` blocks)
        let (gemmaClean, gemmaCalls) = parseGemma(text: text, tools: tools)
        if !gemmaCalls.isEmpty {
            return (gemmaClean, gemmaCalls)
        }

        // Try bare function calls matching known tool names: tool_name(args...)
        let (bareClean, bareCalls) = parseBareToolCalls(text: text, tools: tools)
        if !bareCalls.isEmpty {
            return (bareClean, bareCalls)
        }

        return (text, [])
    }

    // MARK: - Gemma: ```tool_code``` blocks

    /// Parse Gemma's tool_code blocks: ```tool_code\nfunc_name(arg="value")\n```
    private static func parseGemma(text: String, tools: [APIToolDefinition]?) -> (String, [ParsedToolCall]) {
        let pattern = #"```tool_code\s*(.*?)\s*```"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .dotMatchesLineSeparators) else {
            return (text, [])
        }

        let nsText = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsText.length))
        guard !matches.isEmpty else { return (text, []) }

        var toolCalls: [ParsedToolCall] = []

        // Build tool definitions map for parameter inference
        var toolDefs: [String: [String]] = [:]
        if let tools {
            for tool in tools {
                let paramNames = tool.function.parameters?["properties"]?.value as? [String: Any]
                toolDefs[tool.function.name] = paramNames.map { Array($0.keys).sorted() } ?? []
            }
        }

        for (i, match) in matches.enumerated() {
            let callStr = nsText.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)

            if let (name, args) = parsePythonCall(callStr, toolDefs: toolDefs) {
                let argsJSON = (try? JSONSerialization.data(withJSONObject: args))
                    .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
                let callId = String(format: "call_%d_%08d", i, abs(callStr.hashValue) % 100_000_000)
                toolCalls.append(ParsedToolCall(id: callId, name: name, arguments: argsJSON))
            }
        }

        // Remove tool_code blocks from text
        let cleanText = regex.stringByReplacingMatches(
            in: text, range: NSRange(location: 0, length: nsText.length),
            withTemplate: ""
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        return (cleanText, toolCalls)
    }

    /// Parse a Python-style function call: func_name(arg1="value", arg2=42)
    private static func parsePythonCall(_ callStr: String, toolDefs: [String: [String]]) -> (String, [String: Any])? {
        // Match: func_name(args...)
        let pattern = #"^(\w+)\s*\((.*)\)\s*$"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .dotMatchesLineSeparators) else {
            return nil
        }

        let nsCall = callStr as NSString
        let match = regex.firstMatch(in: callStr, range: NSRange(location: 0, length: nsCall.length))
        guard let match else {
            // Shell-style: "func_name arg1"
            let parts = callStr.split(separator: " ", maxSplits: 1)
            guard let name = parts.first.map(String.init), !name.isEmpty else { return nil }
            if parts.count > 1 {
                let argValue = String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines)
                let paramNames = toolDefs[name] ?? []
                let key = paramNames.first ?? "arg0"
                return (name, [key: argValue])
            }
            return (name, [:])
        }

        let name = nsCall.substring(with: match.range(at: 1))
        let argsStr = nsCall.substring(with: match.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)

        if argsStr.isEmpty {
            return (name, [:])
        }

        // Parse keyword arguments: key="value", key2=42
        var args: [String: Any] = [:]
        let kwPattern = #"(\w+)\s*=\s*("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|[^,]+)"#
        if let kwRegex = try? NSRegularExpression(pattern: kwPattern, options: []) {
            let kwMatches = kwRegex.matches(in: argsStr, range: NSRange(location: 0, length: (argsStr as NSString).length))
            for kwMatch in kwMatches {
                let key = (argsStr as NSString).substring(with: kwMatch.range(at: 1))
                var val = (argsStr as NSString).substring(with: kwMatch.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)
                // Strip quotes
                if (val.hasPrefix("\"") && val.hasSuffix("\"")) || (val.hasPrefix("'") && val.hasSuffix("'")) {
                    val = String(val.dropFirst().dropLast())
                }
                // Try to parse as number/bool
                if let intVal = Int(val) { args[key] = intVal }
                else if let doubleVal = Double(val) { args[key] = doubleVal }
                else if val == "True" || val == "true" { args[key] = true }
                else if val == "False" || val == "false" { args[key] = false }
                else { args[key] = val }
            }
        }

        // If no keyword args found, try positional
        if args.isEmpty {
            let paramNames = toolDefs[name] ?? []
            // Try splitting by comma for positional args
            let positionals = argsStr.split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            for (i, pos) in positionals.enumerated() {
                var val = pos
                if (val.hasPrefix("\"") && val.hasSuffix("\"")) || (val.hasPrefix("'") && val.hasSuffix("'")) {
                    val = String(val.dropFirst().dropLast())
                }
                let key = i < paramNames.count ? paramNames[i] : "arg\(i)"
                args[key] = val
            }
        }

        return (name, args)
    }

    // MARK: - Qwen: <tool_call> XML tags

    /// Parse Qwen's tool_call tags: <tool_call>{"name":"func","arguments":{...}}</tool_call>
    private static func parseQwen(text: String) -> (String, [ParsedToolCall]) {
        let pattern = #"<tool_call>\s*(.*?)\s*</tool_call>"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .dotMatchesLineSeparators) else {
            return (text, [])
        }

        let nsText = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsText.length))
        guard !matches.isEmpty else { return (text, []) }

        var toolCalls: [ParsedToolCall] = []

        for (i, match) in matches.enumerated() {
            let jsonStr = nsText.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)

            guard let data = jsonStr.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let name = obj["name"] as? String else {
                continue
            }

            var argsJSON = "{}"
            if let args = obj["arguments"] {
                if let argsDict = args as? [String: Any],
                   let argsData = try? JSONSerialization.data(withJSONObject: argsDict) {
                    argsJSON = String(data: argsData, encoding: .utf8) ?? "{}"
                } else if let argsStr = args as? String {
                    argsJSON = argsStr
                }
            }

            let callId = String(format: "call_%d_%08d", i, abs(jsonStr.hashValue) % 100_000_000)
            toolCalls.append(ParsedToolCall(id: callId, name: name, arguments: argsJSON))
        }

        let cleanText = regex.stringByReplacingMatches(
            in: text, range: NSRange(location: 0, length: nsText.length),
            withTemplate: ""
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        return (cleanText, toolCalls)
    }

    // MARK: - Bare function calls: tool_name(args...)

    /// Parse bare function calls that match known tool names.
    /// Handles models that output tool calls without fences or XML tags.
    private static func parseBareToolCalls(text: String, tools: [APIToolDefinition]?) -> (String, [ParsedToolCall]) {
        guard let tools, !tools.isEmpty else { return (text, []) }

        let toolNames = tools.map { $0.function.name }
        guard !toolNames.isEmpty else { return (text, []) }

        // Build regex: (tool_name1|tool_name2)\s*\(.*\)
        let escapedNames = toolNames.map { NSRegularExpression.escapedPattern(for: $0) }
        let pattern = "(" + escapedNames.joined(separator: "|") + #")\s*\((.*)\)"#
        guard let regex = try? NSRegularExpression(pattern: pattern, options: .dotMatchesLineSeparators) else {
            return (text, [])
        }

        let nsText = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: nsText.length))
        guard !matches.isEmpty else { return (text, []) }

        var toolDefs: [String: [String]] = [:]
        for tool in tools {
            let paramNames = tool.function.parameters?["properties"]?.value as? [String: Any]
            toolDefs[tool.function.name] = paramNames.map { Array($0.keys).sorted() } ?? []
        }

        var toolCalls: [ParsedToolCall] = []
        for (i, match) in matches.enumerated() {
            let name = nsText.substring(with: match.range(at: 1))
            let argsStr = nsText.substring(with: match.range(at: 2)).trimmingCharacters(in: .whitespacesAndNewlines)

            var args: [String: Any] = [:]
            if !argsStr.isEmpty {
                if let (_, parsed) = parsePythonCall("\(name)(\(argsStr))", toolDefs: toolDefs) as (String, [String: Any])? {
                    args = parsed
                }
            }

            let argsJSON = (try? JSONSerialization.data(withJSONObject: args))
                .flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
            let callId = String(format: "call_%d_%08d", i, abs(name.hashValue) % 100_000_000)
            toolCalls.append(ParsedToolCall(id: callId, name: name, arguments: argsJSON))
        }

        let cleanText = regex.stringByReplacingMatches(
            in: text, range: NSRange(location: 0, length: nsText.length),
            withTemplate: ""
        ).trimmingCharacters(in: .whitespacesAndNewlines)

        return (cleanText, toolCalls)
    }
}
