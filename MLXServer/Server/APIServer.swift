import AppKit
import Foundation
import MLXLMCommon
import Network

/// Lightweight HTTP server that exposes OpenAI-compatible endpoints.
/// Runs entirely in-process using NWListener (Network.framework, no third-party deps).
@Observable
@MainActor
final class APIServer {
    var isRunning = false
    var port: Int = 1234
    var requestCount: Int = 0
    let inferenceStats = InferenceStats()

    private var listener: NWListener?
    private var modelManager: ModelManager?

    // Persistent ChatSession for KV cache reuse across requests
    private var cachedSession: ChatSession?
    private var cachedMessages: [Chat.Message]?
    private var cachedModelId: String?

    func start(modelManager: ModelManager, port: Int = 1234) {
        guard !isRunning else { return }
        self.modelManager = modelManager
        self.port = port

        do {
            let params = NWParameters.tcp
            params.allowLocalEndpointReuse = true
            listener = try NWListener(using: params, on: NWEndpoint.Port(integerLiteral: UInt16(port)))

            listener?.stateUpdateHandler = { [weak self] state in
                Task { @MainActor in
                    switch state {
                    case .ready:
                        self?.isRunning = true
                        print("[APIServer] Listening on port \(port)")
                    case .failed(let error):
                        self?.isRunning = false
                        print("[APIServer] Failed: \(error)")
                    case .cancelled:
                        self?.isRunning = false
                    default:
                        break
                    }
                }
            }

            listener?.newConnectionHandler = { [weak self] connection in
                Task { @MainActor in
                    self?.handleConnection(connection)
                }
            }

            listener?.start(queue: .global(qos: .userInitiated))
            inferenceStats.startSampling()
        } catch {
            print("[APIServer] Failed to start: \(error)")
        }
    }

    func stop() {
        listener?.cancel()
        listener = nil
        isRunning = false
        cachedSession = nil
        cachedMessages = nil
        cachedModelId = nil
        inferenceStats.stopSampling()
    }

    // MARK: - Connection handling

    private func handleConnection(_ connection: NWConnection) {
        connection.start(queue: .global(qos: .userInitiated))
        receiveFullHTTPRequest(connection: connection, accumulated: Data())
    }

    /// Receive the full HTTP request, accumulating data until we have the complete body.
    /// This handles large POST bodies (e.g. base64 images) that arrive in multiple chunks.
    private func receiveFullHTTPRequest(connection: NWConnection, accumulated: Data) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 4_194_304) {
            [weak self] data, _, isComplete, error in
            guard let self else { connection.cancel(); return }

            var buffer = accumulated
            if let data { buffer.append(data) }

            // Try to determine if we have the full request
            if let request = HTTPRequest.parse(buffer) {
                // Check if we have enough body data based on Content-Length
                if let clHeader = request.headers["content-length"],
                   let contentLength = Int(clHeader),
                   (request.body?.count ?? 0) < contentLength {
                    // Need more data
                    if isComplete {
                        // Connection closed before we got all data
                        Task { @MainActor in
                            self.sendResponse(connection: connection, status: 400, body: #"{"error":"Incomplete request body"}"#)
                        }
                    } else {
                        self.receiveFullHTTPRequest(connection: connection, accumulated: buffer)
                    }
                    return
                }

                Task { @MainActor in
                    self.requestCount += 1
                    await self.processHTTPRequest(request: request, connection: connection)
                }
            } else if isComplete {
                Task { @MainActor in
                    self.sendResponse(connection: connection, status: 400, body: #"{"error":"Bad Request"}"#)
                }
            } else {
                // Keep accumulating
                self.receiveFullHTTPRequest(connection: connection, accumulated: buffer)
            }
        }
    }

    private func processHTTPRequest(request: HTTPRequest, connection: NWConnection) async {
        // CORS preflight
        if request.method == "OPTIONS" {
            sendResponse(connection: connection, status: 200, body: "", extraHeaders: corsHeaders())
            return
        }

        switch (request.method, request.path) {
        case ("GET", "/health"):
            sendResponse(connection: connection, status: 200, body: #"{"status":"ok"}"#)

        case ("GET", "/v1/models"):
            await handleListModels(connection: connection)

        case ("POST", "/v1/chat/completions"):
            await handleChatCompletions(connection: connection, body: request.body)

        default:
            sendResponse(connection: connection, status: 404, body: #"{"error":"Not Found"}"#)
        }
    }

    // MARK: - GET /v1/models

    private func handleListModels(connection: NWConnection) async {
        let models = ModelConfig.availableModels.map { config in
            APIModelInfo(
                id: config.repoId,
                object: "model",
                created: Int(Date().timeIntervalSince1970),
                owned_by: "local",
                context_window: config.contextLength
            )
        }
        let response = APIModelListResponse(object: "list", data: models)

        if let json = try? JSONEncoder().encode(response) {
            sendResponse(connection: connection, status: 200, body: String(data: json, encoding: .utf8) ?? "{}")
        }
    }

    // MARK: - POST /v1/chat/completions

    private func handleChatCompletions(connection: NWConnection, body: Data?) async {
        guard let body, let request = try? JSONDecoder().decode(APIChatCompletionRequest.self, from: body) else {
            sendResponse(connection: connection, status: 400, body: #"{"error":"Invalid request body"}"#)
            return
        }

        guard let modelManager else {
            sendResponse(connection: connection, status: 503, body: #"{"error":"No model loaded"}"#)
            return
        }

        // Model swapping: if the request specifies a different model, swap to it
        if let requestedModel = request.model, !requestedModel.isEmpty {
            if let targetConfig = ModelConfig.resolve(requestedModel) {
                if modelManager.currentModel?.id != targetConfig.id {
                    print("[APIServer] Swapping model: \(modelManager.currentModel?.repoId ?? "none") -> \(targetConfig.repoId)")
                    cachedSession = nil
                    cachedMessages = nil
                    cachedModelId = nil
                    await modelManager.loadModel(targetConfig)
                }
            }
            // If we can't resolve the model, continue with whatever is loaded
        }

        guard modelManager.isReady, let container = modelManager.modelContainer else {
            sendResponse(connection: connection, status: 503, body: #"{"error":"No model loaded"}"#)
            return
        }

        let isStream = request.stream ?? false
        let temperature = request.temperature ?? 0.7
        let topP = request.top_p ?? 1.0
        let maxTokens = request.max_tokens ?? 4096
        let requestId = "chatcmpl-\(UUID().uuidString.prefix(12).lowercased())"
        let created = Int(Date().timeIntervalSince1970)
        let modelName = request.model ?? modelManager.currentModel?.repoId ?? "unknown"
        let contextLength = modelManager.currentModel?.contextLength ?? 0

        // Convert API messages to Chat.Message, extracting images from content parts
        var chatMessages: [Chat.Message] = []
        var images: [UserInput.Image] = []
        let currentModelRepoId = modelManager.currentModel?.repoId ?? modelName

        // Inject tool definitions into the system prompt if tools are provided
        if let tools = request.tools, !tools.isEmpty {
            let toolSystemPrompt = ToolPromptBuilder.buildSystemPrompt(tools: tools, modelId: currentModelRepoId)

            // Check if there's already a system message
            let hasSystem = request.messages.contains { $0.role == "system" }
            if hasSystem {
                // Append tool prompt to existing system message (handled below during conversion)
            } else {
                // For Gemma: inject as user message (Gemma doesn't support system role natively)
                // For Qwen: inject as system message
                if currentModelRepoId.lowercased().contains("qwen") {
                    chatMessages.append(Chat.Message(role: .system, content: toolSystemPrompt))
                } else {
                    chatMessages.append(Chat.Message(role: .user, content: toolSystemPrompt))
                    chatMessages.append(Chat.Message(role: .assistant, content: "Understood. I will use the provided tools when appropriate."))
                }
            }
        }

        let toolsForInjection = request.tools
        let isQwen = currentModelRepoId.lowercased().contains("qwen")

        for msg in request.messages {
            let role: Chat.Message.Role = switch msg.role {
            case "system": .system
            case "assistant": .assistant
            case "tool": .user
            default: .user
            }

            var text = msg.content?.textContent ?? ""

            // If this is a system message and tools are provided, append tool definitions
            if msg.role == "system", let tools = toolsForInjection, !tools.isEmpty {
                let toolSystemPrompt = ToolPromptBuilder.buildSystemPrompt(tools: tools, modelId: currentModelRepoId)
                text = text + "\n\n" + toolSystemPrompt
            }

            // Format tool_call_id responses as tool_output for the model
            if msg.role == "tool" {
                if isQwen {
                    // Qwen expects tool results as-is in a user message
                    // (the role is already mapped to .user above)
                } else {
                    // Gemma expects tool results wrapped in ```tool_output``` blocks
                    text = "```tool_output\n\(text)\n```"
                }
            }

            // Format assistant tool_calls back into model-native format
            if msg.role == "assistant", let toolCalls = msg.tool_calls, !toolCalls.isEmpty {
                let formattedCalls: String
                if isQwen {
                    formattedCalls = ToolPromptBuilder.formatQwenToolCalls(toolCalls)
                } else {
                    formattedCalls = ToolPromptBuilder.formatGemmaToolCalls(toolCalls)
                }
                text = (text.isEmpty ? "" : text + "\n") + formattedCalls
            }

            // Extract base64 images from content parts
            let imageURLs = msg.content?.imageURLs ?? []
            var messageImages: [UserInput.Image] = []
            for urlString in imageURLs {
                if let image = decodeBase64Image(urlString) {
                    messageImages.append(image)
                }
            }

            // Attach images to this specific message
            chatMessages.append(Chat.Message(role: role, content: text, images: messageImages))
            images.append(contentsOf: messageImages)
        }

        // Context window check: estimate token count and reject if over limit
        if contextLength > 0 {
            let totalChars = chatMessages.reduce(0) { $0 + $1.content.count }
            let estimatedTokens = totalChars * 10 / 35  // ~3.5 chars per token
            let needed = estimatedTokens + maxTokens
            if needed > contextLength {
                let errorBody = """
                {"error":{"message":"This model's maximum context length is \(contextLength) tokens. \
                However, your messages resulted in approximately \(estimatedTokens) tokens and \
                \(maxTokens) tokens were requested for the completion (\(needed) total). \
                Please reduce the length of the messages or completion.",\
                "type":"invalid_request_error","code":"context_length_exceeded"}}
                """
                sendResponse(connection: connection, status: 400, body: errorBody)
                return
            }
        }

        let generateParams = GenerateParameters(
            maxTokens: maxTokens,
            temperature: Float(temperature),
            topP: Float(topP)
        )

        // Feed all messages except the last as history, then send the last as the prompt
        let allButLast = Array(chatMessages.dropLast())
        let lastMessage = chatMessages.last ?? Chat.Message(role: .user, content: "")

        // KV cache reuse: check if the cached session's history matches
        let currentModelId = modelManager.currentModel?.id
        let canReuse = cachedSession != nil
            && cachedModelId == currentModelId
            && cachedMessages != nil
            && messagesMatch(cachedMessages!, allButLast)

        let session: ChatSession
        if canReuse {
            print("[APIServer] Reusing cached session (\(allButLast.count) history messages)")
            session = cachedSession!
            session.generateParameters = generateParams
        } else {
            if cachedSession != nil {
                print("[APIServer] History diverged, creating fresh session")
            }
            if !allButLast.isEmpty {
                session = ChatSession(
                    container,
                    history: allButLast,
                    generateParameters: generateParams
                )
            } else {
                session = ChatSession(
                    container,
                    generateParameters: generateParams
                )
            }
        }

        // Extract images from the last message only (ChatSession.streamDetails takes images separately)
        let lastImages = lastMessage.images

        inferenceStats.requestStarted(contextLength: contextLength)

        if isStream {
            await handleStreamingResponse(
                connection: connection,
                session: session,
                prompt: lastMessage.content,
                images: lastImages,
                tools: request.tools,
                requestId: requestId,
                created: created,
                modelName: modelName
            )
        } else {
            await handleNonStreamingResponse(
                connection: connection,
                session: session,
                prompt: lastMessage.content,
                images: lastImages,
                tools: request.tools,
                requestId: requestId,
                created: created,
                modelName: modelName
            )
        }

        // Cache the session for reuse on next request
        // allButLast + lastMessage (user) + assistant response = new cached history
        cachedSession = session
        cachedMessages = chatMessages  // full messages including the one just sent
        cachedModelId = currentModelId
    }

    /// Decode a base64 data URI (data:image/png;base64,...) into a UserInput.Image.
    private func decodeBase64Image(_ urlString: String) -> UserInput.Image? {
        // Handle data URIs: data:image/png;base64,<data>
        let base64String: String
        if urlString.hasPrefix("data:") {
            guard let commaIndex = urlString.firstIndex(of: ",") else { return nil }
            base64String = String(urlString[urlString.index(after: commaIndex)...])
        } else {
            // Could be a plain base64 string
            base64String = urlString
        }

        guard let data = Data(base64Encoded: base64String),
              let nsImage = NSImage(data: data),
              let cgImage = nsImage.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
            return nil
        }

        return .ciImage(CIImage(cgImage: cgImage))
    }

    // MARK: - Non-streaming response

    private func handleNonStreamingResponse(
        connection: NWConnection,
        session: ChatSession,
        prompt: String,
        images: [UserInput.Image],
        tools: [APIToolDefinition]?,
        requestId: String,
        created: Int,
        modelName: String
    ) async {
        do {
            var fullText = ""
            var promptTokens = 0
            var completionTokens = 0
            var frameworkToolCalls: [MLXLMCommon.ToolCall] = []

            let stream = session.streamDetails(
                to: prompt,
                images: images,
                videos: []
            )

            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    fullText += text
                    completionTokens += 1
                    inferenceStats.tokenGenerated(tokensPerSecond: 0, totalGenerated: completionTokens)
                case .info(let info):
                    promptTokens = info.promptTokenCount
                    completionTokens = info.generationTokenCount
                    inferenceStats.prefillCompleted(promptTokens: promptTokens)
                    if info.tokensPerSecond > 0 {
                        inferenceStats.tokenGenerated(tokensPerSecond: info.tokensPerSecond, totalGenerated: completionTokens)
                    }
                case .toolCall(let call):
                    frameworkToolCalls.append(call)
                }
            }

            inferenceStats.requestCompleted(promptTokens: promptTokens, generationTokens: completionTokens)

            // Parse tool calls: first check framework-detected ones, then our own text parser
            var finishReason = "stop"
            var responseContent: String? = fullText
            var apiToolCalls: [APIToolCall]? = nil

            if !frameworkToolCalls.isEmpty {
                // Framework natively detected tool calls (e.g. Qwen)
                finishReason = "tool_calls"
                apiToolCalls = frameworkToolCalls.enumerated().map { i, tc in
                    let argsJSON: String
                    let argsDict = tc.function.arguments.mapValues { $0.anyValue }
                    if let data = try? JSONSerialization.data(withJSONObject: argsDict),
                       let str = String(data: data, encoding: .utf8) {
                        argsJSON = str
                    } else {
                        argsJSON = "{}"
                    }
                    let callId = String(format: "call_%d_%08d", i, abs(tc.function.name.hashValue) % 100_000_000)
                    return APIToolCall(
                        index: i,
                        id: callId,
                        type: "function",
                        function: APIFunctionCall(name: tc.function.name, arguments: argsJSON)
                    )
                }
                responseContent = fullText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : fullText
            } else if let tools, !tools.isEmpty {
                // Try our own text parser (e.g. Gemma tool_code blocks)
                let (cleanText, parsedCalls) = ToolCallParser.parse(text: fullText, tools: tools)
                if !parsedCalls.isEmpty {
                    finishReason = "tool_calls"
                    apiToolCalls = parsedCalls.enumerated().map { i, tc in
                        APIToolCall(
                            index: i,
                            id: tc.id,
                            type: "function",
                            function: APIFunctionCall(name: tc.name, arguments: tc.arguments)
                        )
                    }
                    responseContent = cleanText.isEmpty ? nil : cleanText
                }
            }

            let response = APIChatCompletionResponse(
                id: requestId,
                object: "chat.completion",
                created: created,
                model: modelName,
                choices: [
                    APIChoice(
                        index: 0,
                        message: APIChoiceMessage(
                            role: "assistant",
                            content: responseContent,
                            tool_calls: apiToolCalls
                        ),
                        finish_reason: finishReason
                    )
                ],
                usage: APIUsageInfo(
                    prompt_tokens: promptTokens,
                    completion_tokens: completionTokens,
                    total_tokens: promptTokens + completionTokens
                )
            )

            if let json = try? JSONEncoder().encode(response) {
                sendResponse(connection: connection, status: 200, body: String(data: json, encoding: .utf8) ?? "{}")
            }
        } catch {
            inferenceStats.requestCompleted(promptTokens: 0, generationTokens: 0)
            sendResponse(connection: connection, status: 500, body: #"{"error":"\#(error.localizedDescription)"}"#)
        }
    }

    // MARK: - Streaming (SSE) response

    private func handleStreamingResponse(
        connection: NWConnection,
        session: ChatSession,
        prompt: String,
        images: [UserInput.Image],
        tools: [APIToolDefinition]?,
        requestId: String,
        created: Int,
        modelName: String
    ) async {
        // Send SSE headers
        let header = [
            "HTTP/1.1 200 OK",
            "Content-Type: text/event-stream",
            "Cache-Control: no-cache",
            "Connection: keep-alive",
            "Access-Control-Allow-Origin: *",
            "",
            "",
        ].joined(separator: "\r\n")

        let headerSent = await withCheckedContinuation { continuation in
            connection.send(content: header.data(using: .utf8), completion: .contentProcessed({ _ in
                continuation.resume(returning: true)
            }))
        }
        guard headerSent else { return }

        // Send initial role chunk
        sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
            id: requestId,
            object: "chat.completion.chunk",
            created: created,
            model: modelName,
            choices: [APIStreamChoice(index: 0, delta: APIDeltaMessage(role: "assistant", content: nil, tool_calls: nil), finish_reason: nil)],
            usage: nil
        ))

        // When tools are available, buffer full response to parse tool calls
        // (otherwise raw tool-call markup leaks into streamed text)
        let bufferForTools = tools != nil && !(tools?.isEmpty ?? true)

        var promptTokens = 0
        var completionTokens = 0
        var fullText = ""
        var frameworkToolCalls: [MLXLMCommon.ToolCall] = []

        do {
            let stream = session.streamDetails(
                to: prompt,
                images: images,
                videos: []
            )

            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    completionTokens += 1
                    fullText += text
                    inferenceStats.tokenGenerated(tokensPerSecond: 0, totalGenerated: completionTokens)

                    if !bufferForTools {
                        sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
                            id: requestId,
                            object: "chat.completion.chunk",
                            created: created,
                            model: modelName,
                            choices: [APIStreamChoice(index: 0, delta: APIDeltaMessage(role: nil, content: text, tool_calls: nil), finish_reason: nil)],
                            usage: nil
                        ))
                    }

                case .info(let info):
                    promptTokens = info.promptTokenCount
                    completionTokens = info.generationTokenCount
                    inferenceStats.prefillCompleted(promptTokens: promptTokens)
                    if info.tokensPerSecond > 0 {
                        inferenceStats.tokenGenerated(tokensPerSecond: info.tokensPerSecond, totalGenerated: completionTokens)
                    }

                case .toolCall(let call):
                    frameworkToolCalls.append(call)
                }
            }
        } catch {
            inferenceStats.requestCompleted(promptTokens: promptTokens, generationTokens: completionTokens)
            let errorEvent = "data: {\"error\":\"\(error.localizedDescription)\"}\n\n"
            connection.send(content: errorEvent.data(using: .utf8), completion: .contentProcessed({ _ in }))
        }

        // Post-generation: handle tool calls (framework-detected or text-parsed)
        var finishReason = "stop"

        if !frameworkToolCalls.isEmpty {
            // Framework natively detected tool calls (e.g. Qwen)
            finishReason = "tool_calls"

            // Emit any buffered text content
            if !fullText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
                    id: requestId,
                    object: "chat.completion.chunk",
                    created: created,
                    model: modelName,
                    choices: [APIStreamChoice(index: 0, delta: APIDeltaMessage(role: nil, content: fullText, tool_calls: nil), finish_reason: nil)],
                    usage: nil
                ))
            }

            // Emit tool call chunks
            for (i, tc) in frameworkToolCalls.enumerated() {
                let argsDict = tc.function.arguments.mapValues { $0.anyValue }
                let argsJSON: String
                if let data = try? JSONSerialization.data(withJSONObject: argsDict),
                   let str = String(data: data, encoding: .utf8) {
                    argsJSON = str
                } else {
                    argsJSON = "{}"
                }
                let callId = String(format: "call_%d_%08d", i, abs(tc.function.name.hashValue) % 100_000_000)
                let apiToolCall = APIToolCall(
                    index: i,
                    id: callId,
                    type: "function",
                    function: APIFunctionCall(name: tc.function.name, arguments: argsJSON)
                )
                sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
                    id: requestId,
                    object: "chat.completion.chunk",
                    created: created,
                    model: modelName,
                    choices: [APIStreamChoice(index: 0, delta: APIDeltaMessage(role: nil, content: nil, tool_calls: [apiToolCall]), finish_reason: nil)],
                    usage: nil
                ))
            }
        } else if bufferForTools {
            // Text-parsed tool calls (e.g. Gemma tool_code blocks)
            let (cleanText, parsed) = ToolCallParser.parse(text: fullText, tools: tools)
            if !parsed.isEmpty {
                finishReason = "tool_calls"
                fullText = cleanText
            }

            // Emit buffered content (cleaned of tool-call markup)
            if !fullText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
                    id: requestId,
                    object: "chat.completion.chunk",
                    created: created,
                    model: modelName,
                    choices: [APIStreamChoice(index: 0, delta: APIDeltaMessage(role: nil, content: fullText, tool_calls: nil), finish_reason: nil)],
                    usage: nil
                ))
            }

            // Emit tool call chunks
            for (i, tc) in parsed.enumerated() {
                let apiToolCall = APIToolCall(
                    index: i,
                    id: tc.id,
                    type: "function",
                    function: APIFunctionCall(name: tc.name, arguments: tc.arguments)
                )
                sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
                    id: requestId,
                    object: "chat.completion.chunk",
                    created: created,
                    model: modelName,
                    choices: [APIStreamChoice(index: 0, delta: APIDeltaMessage(role: nil, content: nil, tool_calls: [apiToolCall]), finish_reason: nil)],
                    usage: nil
                ))
            }
        }

        // Final chunk with finish_reason and usage
        sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
            id: requestId,
            object: "chat.completion.chunk",
            created: created,
            model: modelName,
            choices: [APIStreamChoice(index: 0, delta: APIDeltaMessage(role: nil, content: nil, tool_calls: nil), finish_reason: finishReason)],
            usage: APIUsageInfo(
                prompt_tokens: promptTokens,
                completion_tokens: completionTokens,
                total_tokens: promptTokens + completionTokens
            )
        ))

        inferenceStats.requestCompleted(promptTokens: promptTokens, generationTokens: completionTokens)

        // Send [DONE] and close
        let done = "data: [DONE]\n\n"
        connection.send(content: done.data(using: .utf8), completion: .contentProcessed({ _ in
            connection.cancel()
        }))
    }

    private func sendSSEEvent(connection: NWConnection, chunk: APIChatCompletionChunk) {
        guard let json = try? JSONEncoder().encode(chunk),
              let jsonString = String(data: json, encoding: .utf8) else { return }
        let event = "data: \(jsonString)\n\n"
        connection.send(content: event.data(using: .utf8), completion: .contentProcessed({ _ in }))
    }

    // MARK: - HTTP helpers

    private func sendResponse(
        connection: NWConnection,
        status: Int,
        body: String,
        extraHeaders: [String] = []
    ) {
        let statusText = switch status {
        case 200: "OK"
        case 400: "Bad Request"
        case 404: "Not Found"
        case 500: "Internal Server Error"
        case 503: "Service Unavailable"
        default: "Error"
        }

        var headers = [
            "HTTP/1.1 \(status) \(statusText)",
            "Content-Type: application/json",
            "Access-Control-Allow-Origin: *",
            "Access-Control-Allow-Methods: GET, POST, OPTIONS",
            "Access-Control-Allow-Headers: Content-Type, Authorization",
            "Content-Length: \(body.utf8.count)",
        ]
        headers.append(contentsOf: extraHeaders)
        headers.append("")
        headers.append("")

        let response = headers.joined(separator: "\r\n") + body
        connection.send(content: response.data(using: .utf8), completion: .contentProcessed({ _ in
            connection.cancel()
        }))
    }

    private func corsHeaders() -> [String] {
        [
            "Access-Control-Allow-Methods: GET, POST, OPTIONS",
            "Access-Control-Allow-Headers: Content-Type, Authorization",
            "Access-Control-Max-Age: 86400",
        ]
    }

    /// Check if cached messages are a prefix of new messages (for KV cache reuse).
    /// The cached messages include the full history from the previous request.
    /// For cache reuse, all but the last message of the new request must match
    /// all but the last message of the cached messages (the cached last was the
    /// previous user prompt, which is now part of the history).
    private func messagesMatch(_ cached: [Chat.Message], _ newHistory: [Chat.Message]) -> Bool {
        // The cached messages are the full chatMessages from the previous request.
        // For the cache to be reusable, the new history (allButLast) must match
        // exactly what the session has already processed.
        // After a request, the session has seen: cachedMessages' history + prompt + response.
        // So on the next request, if newHistory == cachedMessages, the session already
        // contains all of those turns and we can just send the new last message.
        guard cached.count == newHistory.count else { return false }
        for (a, b) in zip(cached, newHistory) {
            if a.role != b.role || a.content != b.content { return false }
        }
        return true
    }
}

// MARK: - HTTP request parser

private struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data?

    /// Parse raw HTTP data into a structured request.
    /// Uses raw Data operations to correctly handle binary body content.
    static func parse(_ data: Data) -> HTTPRequest? {
        // Find \r\n\r\n boundary between headers and body
        let separator: [UInt8] = [0x0D, 0x0A, 0x0D, 0x0A] // \r\n\r\n
        guard let separatorRange = data.firstRange(of: Data(separator)) else {
            // No complete header yet — might need more data
            // But if data is large enough, treat as malformed
            return data.count > 65536 ? nil : nil
        }

        let headerData = data[data.startIndex..<separatorRange.lowerBound]
        guard let headerString = String(data: headerData, encoding: .utf8) else { return nil }

        let lines = headerString.components(separatedBy: "\r\n")
        guard let requestLine = lines.first else { return nil }

        let parts = requestLine.split(separator: " ", maxSplits: 2)
        guard parts.count >= 2 else { return nil }

        let method = String(parts[0])
        let fullPath = String(parts[1])
        let path = fullPath.components(separatedBy: "?").first ?? fullPath

        var headers: [String: String] = [:]
        for line in lines.dropFirst() {
            let kv = line.split(separator: ":", maxSplits: 1)
            if kv.count == 2 {
                headers[String(kv[0]).trimmingCharacters(in: .whitespaces).lowercased()] =
                    String(kv[1]).trimmingCharacters(in: .whitespaces)
            }
        }

        // Body is everything after \r\n\r\n
        let bodyStart = separatorRange.upperBound
        let body: Data? = bodyStart < data.endIndex ? data[bodyStart..<data.endIndex] : nil

        return HTTPRequest(method: method, path: path, headers: headers, body: body)
    }
}
