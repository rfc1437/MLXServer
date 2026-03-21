import Foundation

// MARK: - Request models

struct APIFunctionDefinition: Codable {
    let name: String
    let description: String?
    let parameters: [String: AnyCodable]?
}

struct APIToolDefinition: Codable {
    let type: String // "function"
    let function: APIFunctionDefinition
}

struct APIFunctionCall: Codable {
    let name: String
    let arguments: String // JSON string

    init(name: String, arguments: String) {
        self.name = name
        self.arguments = arguments
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decode(String.self, forKey: .name)

        if let argumentString = try? container.decode(String.self, forKey: .arguments) {
            arguments = argumentString
            return
        }

        if let argumentObject = try? container.decode([String: AnyCodable].self, forKey: .arguments) {
            let jsonObject = argumentObject.mapValues(\.value)
            if let data = try? JSONSerialization.data(withJSONObject: jsonObject, options: [.sortedKeys]),
               let string = String(data: data, encoding: .utf8) {
                arguments = string
            } else {
                arguments = "{}"
            }
            return
        }

        if let argumentArray = try? container.decode([AnyCodable].self, forKey: .arguments) {
            let jsonObject = argumentArray.map(\.value)
            if let data = try? JSONSerialization.data(withJSONObject: jsonObject, options: [.sortedKeys]),
               let string = String(data: data, encoding: .utf8) {
                arguments = string
            } else {
                arguments = "[]"
            }
            return
        }

        if (try? container.decodeNil(forKey: .arguments)) == true {
            arguments = "{}"
            return
        }

        arguments = "{}"
    }
}

struct APIToolCall: Codable {
    let index: Int
    let id: String
    let type: String // "function"
    let function: APIFunctionCall

    init(index: Int = 0, id: String, type: String = "function", function: APIFunctionCall) {
        self.index = index
        self.id = id
        self.type = type
        self.function = function
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        index = try container.decodeIfPresent(Int.self, forKey: .index) ?? 0
        id = try container.decodeIfPresent(String.self, forKey: .id) ?? "call_\(UUID().uuidString.lowercased())"
        type = try container.decodeIfPresent(String.self, forKey: .type) ?? "function"
        function = try container.decode(APIFunctionCall.self, forKey: .function)
    }
}

struct APIImageURL: Codable {
    let url: String
    let detail: String?
}

struct APIContentPart: Codable {
    let type: String // "text" or "image_url"
    let text: String?
    let image_url: APIImageURL?
}

struct APIChatMessage: Codable {
    let role: String
    let content: MessageContent?
    let name: String?
    let tool_calls: [APIToolCall]?
    let tool_call_id: String?

    enum MessageContent: Codable {
        case text(String)
        case parts([APIContentPart])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let text = try? container.decode(String.self) {
                self = .text(text)
            } else if let parts = try? container.decode([APIContentPart].self) {
                self = .parts(parts)
            } else {
                self = .text("")
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .text(let text):
                try container.encode(text)
            case .parts(let parts):
                try container.encode(parts)
            }
        }

        /// Extract plain text content.
        var textContent: String {
            switch self {
            case .text(let t): return t
            case .parts(let parts):
                return parts.compactMap { $0.text }.joined()
            }
        }

        /// Extract image URLs/base64 data URIs.
        var imageURLs: [String] {
            switch self {
            case .text: return []
            case .parts(let parts):
                return parts.compactMap { $0.image_url?.url }
            }
        }
    }
}

struct APIChatCompletionRequest: Codable {
    let model: String?
    let messages: [APIChatMessage]
    let temperature: Double?
    let top_p: Double?
    let top_k: Int?
    let min_p: Double?
    let max_tokens: Int?
    let stream: Bool?
    let stop: StopSequence?
    let tools: [APIToolDefinition]?
    let tool_choice: AnyCodable?
    let repetition_penalty: Double?
    let frequency_penalty: Double?
    let presence_penalty: Double?
    let n: Int?

    init(
        model: String?,
        messages: [APIChatMessage],
        temperature: Double? = nil,
        top_p: Double? = nil,
        max_tokens: Int? = nil,
        stream: Bool? = nil,
        stop: StopSequence? = nil,
        tools: [APIToolDefinition]? = nil,
        tool_choice: AnyCodable? = nil,
        frequency_penalty: Double? = nil,
        presence_penalty: Double? = nil,
        n: Int? = nil,
        top_k: Int? = nil,
        min_p: Double? = nil,
        repetition_penalty: Double? = nil
    ) {
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.max_tokens = max_tokens
        self.stream = stream
        self.stop = stop
        self.tools = tools
        self.tool_choice = tool_choice
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.n = n
    }

    enum StopSequence: Codable {
        case single(String)
        case multiple([String])

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let s = try? container.decode(String.self) {
                self = .single(s)
            } else if let arr = try? container.decode([String].self) {
                self = .multiple(arr)
            } else {
                self = .multiple([])
            }
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            switch self {
            case .single(let s): try container.encode(s)
            case .multiple(let arr): try container.encode(arr)
            }
        }
    }
}

// MARK: - Response models

struct APIUsageInfo: Codable {
    let prompt_tokens: Int
    let completion_tokens: Int
    let total_tokens: Int
}

struct APIChoiceMessage: Codable {
    let role: String
    let content: String?
    let tool_calls: [APIToolCall]?
}

struct APIChoice: Codable {
    let index: Int
    let message: APIChoiceMessage
    let finish_reason: String?
}

struct APIChatCompletionResponse: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [APIChoice]
    let usage: APIUsageInfo
}

// MARK: - Streaming response models

struct APIDeltaMessage: Codable {
    let role: String?
    let content: String?
    let tool_calls: [APIToolCall]?
}

struct APIStreamChoice: Codable {
    let index: Int
    let delta: APIDeltaMessage
    let finish_reason: String?
}

struct APIChatCompletionChunk: Codable {
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [APIStreamChoice]
    let usage: APIUsageInfo?
}

// MARK: - Model listing

struct APIModelInfo: Codable {
    let id: String
    let object: String
    let created: Int
    let owned_by: String
    let context_window: Int?
}

struct APIModelListResponse: Codable {
    let object: String
    let data: [APIModelInfo]
}

// MARK: - Utility: type-erased Codable

struct AnyCodable: Codable {
    let value: Any

    init(_ value: Any) {
        self.value = value
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let intVal = try? container.decode(Int.self) { value = intVal }
        else if let doubleVal = try? container.decode(Double.self) { value = doubleVal }
        else if let boolVal = try? container.decode(Bool.self) { value = boolVal }
        else if let stringVal = try? container.decode(String.self) { value = stringVal }
        else if let arrayVal = try? container.decode([AnyCodable].self) { value = arrayVal.map(\.value) }
        else if let dictVal = try? container.decode([String: AnyCodable].self) {
            value = dictVal.mapValues(\.value)
        } else { value = NSNull() }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch value {
        case let v as Int: try container.encode(v)
        case let v as Double: try container.encode(v)
        case let v as Bool: try container.encode(v)
        case let v as String: try container.encode(v)
        case let v as [Any]: try container.encode(v.map { AnyCodable($0) })
        case let v as [String: Any]: try container.encode(v.mapValues { AnyCodable($0) })
        default: try container.encodeNil()
        }
    }
}
