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
    private var cachedInstructions: String = ""

    func start(modelManager: ModelManager, port: Int = 1234) {
        guard !isRunning else { return }
        self.modelManager = modelManager
        self.port = port

        do {
            let params = NWParameters.tcp
            params.allowLocalEndpointReuse = true
            // Disable Nagle's algorithm so small SSE events go out immediately
            if let tcpOptions = params.defaultProtocolStack.transportProtocol as? NWProtocolTCP.Options {
                tcpOptions.noDelay = true
            }
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
        cachedInstructions = ""
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
                    cachedInstructions = ""
                    await modelManager.loadModel(targetConfig)
                }
            }
            // If we can't resolve the model, continue with whatever is loaded
        }

        // Reload model if it was idle-unloaded
        if modelManager.modelContainer == nil, let lastModelId = Preferences.lastModelId,
           let config = ModelConfig.resolve(lastModelId) {
            print("[APIServer] Reloading idle-unloaded model: \(config.repoId)")
            cachedSession = nil
            cachedMessages = nil
            cachedModelId = nil
            cachedInstructions = ""
            await modelManager.loadModel(config)
        }

        guard modelManager.isReady, let container = modelManager.modelContainer else {
            sendResponse(connection: connection, status: 503, body: #"{"error":"No model loaded"}"#)
            return
        }

        modelManager.touchActivity()

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

        // Build the instructions string (system prompt + tool definitions).
        // This is passed to ChatSession via `instructions:` rather than injected
        // as history messages, so it avoids an expensive history-replay prefill.
        var instructions: String = ""

        // Collect system message text from the request
        for msg in request.messages where msg.role == "system" {
            let text = msg.content?.textContent ?? ""
            if !text.isEmpty {
                if !instructions.isEmpty { instructions += "\n\n" }
                instructions += text
            }
        }

        // Append tool definitions to instructions
        if let tools = request.tools, !tools.isEmpty {
            let toolSystemPrompt = ToolPromptBuilder.buildSystemPrompt(tools: tools, modelId: currentModelRepoId)
            if !instructions.isEmpty { instructions += "\n\n" }
            instructions += toolSystemPrompt
        }

        let toolsForInjection = request.tools
        let isQwen = currentModelRepoId.lowercased().contains("qwen")

        // Convert non-system messages to Chat.Message
        for msg in request.messages where msg.role != "system" {
            let role: Chat.Message.Role = switch msg.role {
            case "assistant": .assistant
            case "tool": .user
            default: .user
            }

            var text = msg.content?.textContent ?? ""

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
            && cachedInstructions == instructions
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
            // Use `instructions:` for system/tool prompt (matches internal chat pattern).
            // Only conversation turns go in `history:` — this avoids replaying the
            // large tool prompt as history on every new session.
            let instr = instructions.isEmpty ? nil : instructions
            if !allButLast.isEmpty {
                session = ChatSession(
                    container,
                    instructions: instr,
                    history: allButLast,
                    generateParameters: generateParams
                )
            } else {
                session = ChatSession(
                    container,
                    instructions: instr,
                    generateParameters: generateParams
                )
            }
        }

        // Extract images from the last message only (ChatSession.streamDetails takes images separately)
        let lastImages = lastMessage.images

        LiveCounters.shared.requestStarted(contextLength: contextLength)

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
        cachedInstructions = instructions
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
                    LiveCounters.shared.tokenGenerated(tokensPerSecond: 0, totalGenerated: completionTokens)
                case .info(let info):
                    promptTokens = info.promptTokenCount
                    completionTokens = info.generationTokenCount
                    LiveCounters.shared.prefillCompleted(promptTokens: promptTokens)
                    if info.tokensPerSecond > 0 {
                        LiveCounters.shared.tokenGenerated(tokensPerSecond: info.tokensPerSecond, totalGenerated: completionTokens)
                    }
                case .toolCall(let call):
                    frameworkToolCalls.append(call)
                }
            }

            LiveCounters.shared.requestCompleted(generationTokens: completionTokens)

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
            LiveCounters.shared.requestCompleted(generationTokens: 0)
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

        await Self.sendData(connection: connection, data: header.data(using: .utf8)!)

        // Send initial role chunk
        await Self.sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
            id: requestId,
            object: "chat.completion.chunk",
            created: created,
            model: modelName,
            choices: [APIStreamChoice(index: 0, delta: APIDeltaMessage(role: "assistant", content: nil, tool_calls: nil), finish_reason: nil)],
            usage: nil
        ))

        let hasTools = tools != nil && !(tools?.isEmpty ?? true)

        // Run the generation loop OFF MainActor.
        // ChatSession and NWConnection don't need MainActor.
        // Running on MainActor caused every token to compete with SwiftUI
        // rendering, creating back-pressure that coalesced all output.
        let stream = session.streamDetails(
            to: prompt,
            images: images,
            videos: []
        )
        // Transfer non-Sendable values to the nonisolated loop.
        // Safe because we don't touch session/images again until after the loop.
        let result = await {
            nonisolated(unsafe) let stream = stream
            return await Self.runStreamingLoop(
                connection: connection,
                stream: stream,
                requestId: requestId,
                created: created,
                modelName: modelName
            )
        }()

        let (promptTokens, completionTokens, fullText, frameworkToolCalls) = result

        // Stats were already updated by LiveCounters inside the loop

        // Post-generation: handle tool calls (framework-detected or text-parsed)
        var finishReason = "stop"

        if !frameworkToolCalls.isEmpty {
            finishReason = "tool_calls"
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
                await Self.sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
                    id: requestId,
                    object: "chat.completion.chunk",
                    created: created,
                    model: modelName,
                    choices: [APIStreamChoice(index: 0, delta: APIDeltaMessage(role: nil, content: nil, tool_calls: [apiToolCall]), finish_reason: nil)],
                    usage: nil
                ))
            }
        } else if hasTools {
            let (_, parsed) = ToolCallParser.parse(text: fullText, tools: tools)
            if !parsed.isEmpty {
                finishReason = "tool_calls"
            }
            for (i, tc) in parsed.enumerated() {
                let apiToolCall = APIToolCall(
                    index: i,
                    id: tc.id,
                    type: "function",
                    function: APIFunctionCall(name: tc.name, arguments: tc.arguments)
                )
                await Self.sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
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
        await Self.sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
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

        LiveCounters.shared.requestCompleted(generationTokens: completionTokens)

        // Send [DONE] and close
        await Self.sendData(connection: connection, data: "data: [DONE]\n\n".data(using: .utf8)!)
        connection.cancel()
    }

    /// Run the token generation + SSE send loop entirely off MainActor.
    /// This is critical: if the loop runs on MainActor, every token requires
    /// multiple actor hops competing with SwiftUI, causing all output to batch.
    nonisolated private static func runStreamingLoop(
        connection: NWConnection,
        stream: AsyncThrowingStream<Generation, any Error>,
        requestId: String,
        created: Int,
        modelName: String
    ) async -> (Int, Int, String, [MLXLMCommon.ToolCall]) {
        var promptTokens = 0
        var completionTokens = 0
        var fullText = ""
        var frameworkToolCalls: [MLXLMCommon.ToolCall] = []

        do {
            for try await generation in stream {
                switch generation {
                case .chunk(let text):
                    completionTokens += 1
                    fullText += text

                    // Update live counters directly — no MainActor hop needed
                    LiveCounters.shared.tokenGenerated(tokensPerSecond: 0, totalGenerated: completionTokens)

                    // Send directly — no MainActor hop.
                    await sendSSEEvent(connection: connection, chunk: APIChatCompletionChunk(
                        id: requestId,
                        object: "chat.completion.chunk",
                        created: created,
                        model: modelName,
                        choices: [APIStreamChoice(index: 0, delta: APIDeltaMessage(role: nil, content: text, tool_calls: nil), finish_reason: nil)],
                        usage: nil
                    ))

                case .info(let info):
                    promptTokens = info.promptTokenCount
                    completionTokens = info.generationTokenCount
                    LiveCounters.shared.prefillCompleted(promptTokens: promptTokens)
                    if info.tokensPerSecond > 0 {
                        LiveCounters.shared.tokenGenerated(tokensPerSecond: info.tokensPerSecond, totalGenerated: completionTokens)
                    }

                case .toolCall(let call):
                    frameworkToolCalls.append(call)
                }
            }
        } catch {
            let errorEvent = "data: {\"error\":\"\(error.localizedDescription)\"}\n\n"
            await sendData(connection: connection, data: errorEvent.data(using: .utf8)!)
        }

        return (promptTokens, completionTokens, fullText, frameworkToolCalls)
    }

    /// Send an SSE event and wait for the protocol stack to process it.
    nonisolated private static func sendSSEEvent(connection: NWConnection, chunk: APIChatCompletionChunk) async {
        guard let json = try? JSONEncoder().encode(chunk),
              let jsonString = String(data: json, encoding: .utf8) else { return }
        let event = "data: \(jsonString)\n\n"
        await sendData(connection: connection, data: event.data(using: .utf8)!)
    }

    /// Send raw data on the connection and wait for the protocol stack to process it.
    nonisolated private static func sendData(connection: NWConnection, data: Data) async {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
            connection.send(content: data, completion: .contentProcessed({ _ in
                continuation.resume()
            }))
        }
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

    /// Check if the cached session can be reused for the new history.
    ///
    /// After a request the session's KV cache contains:
    ///   cachedMessages (history + user prompt) + the generated assistant response.
    /// On the next request the client sends back the full conversation, so
    /// `newHistory` (allButLast) is typically `cachedMessages` + 1 assistant reply.
    /// We allow reuse when `cached` is a prefix of `newHistory` and there is at most
    /// one extra message (the assistant response the session already generated).
    /// More than one extra message (e.g. injected tool results) means the session
    /// hasn't processed them, so we must create a fresh session.
    private func messagesMatch(_ cached: [Chat.Message], _ newHistory: [Chat.Message]) -> Bool {
        guard cached.count <= newHistory.count,
              newHistory.count <= cached.count + 1 else { return false }
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
