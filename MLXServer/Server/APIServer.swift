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
        TokenPrefixCache.shared.invalidateAll()
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
                    TokenPrefixCache.shared.invalidateAll()
                    await modelManager.loadModel(targetConfig)
                }
            }
            // If we can't resolve the model, continue with whatever is loaded
        }

        // Reload model if it was idle-unloaded
        if modelManager.modelContainer == nil, let lastModelId = Preferences.lastModelId,
           let config = ModelConfig.resolve(lastModelId) {
            print("[APIServer] Reloading idle-unloaded model: \(config.repoId)")
            TokenPrefixCache.shared.invalidateAll()
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
        let currentModel = modelManager.currentModel
        let contextLength = modelManager.currentModel?.contextLength ?? 0

        if let tools = request.tools, !tools.isEmpty, currentModel?.supportsTools != true {
            sendResponse(
                connection: connection,
                status: 400,
                body: #"{"error":{"message":"The currently selected model does not support tool calls.","type":"invalid_request_error","code":"tools_not_supported"}}"#
            )
            return
        }

        LiveCounters.shared.requestStarted(requestId: requestId, contextLength: contextLength)
        let currentModelRepoId = currentModel?.repoId ?? modelName

        let preparedPrompt = PromptBuilder.build(
            from: request,
            modelId: currentModelRepoId,
            thinkingEnabled: Preferences.enableThinking
        )
        let isQwen = currentModelRepoId.lowercased().contains("qwen")

        if preparedPrompt.containsImages, currentModel?.supportsImages != true {
            LiveCounters.shared.requestCompleted(requestId: requestId, generationTokens: 0)
            sendResponse(
                connection: connection,
                status: 400,
                body: #"{"error":{"message":"The currently selected model does not support image inputs.","type":"invalid_request_error","code":"vision_not_supported"}}"#
            )
            return
        }

        // Context window check: estimate token count and reject if over limit
        let estimatedPromptTokens = preparedPrompt.estimatedPromptTokens
        if contextLength > 0 {
            let needed = estimatedPromptTokens + maxTokens
            if needed > contextLength {
                let errorBody = """
                {"error":{"message":"This model's maximum context length is \(contextLength) tokens. \
                However, your messages resulted in approximately \(estimatedPromptTokens) tokens and \
                \(maxTokens) tokens were requested for the completion (\(needed) total). \
                Please reduce the length of the messages or completion.",\
                "type":"invalid_request_error","code":"context_length_exceeded"}}
                """
                LiveCounters.shared.requestCompleted(requestId: requestId, generationTokens: 0)
                sendResponse(connection: connection, status: 400, body: errorBody)
                return
            }
        }

        let generateParams = GenerateParameters(
            maxTokens: maxTokens,
            temperature: Float(temperature),
            topP: Float(topP)
        )
        let currentModelId = modelManager.currentModel?.id ?? modelName
        let engine = InferenceEngine(container: container)
        let preparedInference: InferenceEngine.PreparedInference
        do {
            preparedInference = try await engine.prepare(preparedPrompt.userInput)
        } catch {
            LiveCounters.shared.requestCompleted(requestId: requestId, generationTokens: 0)
            sendResponse(connection: connection, status: 500, body: #"{"error":"\#(error.localizedDescription)"}"#)
            return
        }

        // Vision requests stay uncached until image fingerprinting lands.
        let cacheKey = preparedInference.hasImages ? nil : preparedInference.tokens
        let lease = cacheKey.map { TokenPrefixCache.shared.lookup(cacheKey: $0, modelId: currentModelId) }
            ?? TokenPrefixCache.CacheLease(entryId: UUID(), kvCache: nil, matchedTokenCount: 0, isHit: false)

        LiveCounters.shared.recordPrefillReuse(
            requestId: requestId,
            matchedPromptTokens: lease.matchedTokenCount,
            promptTokenCount: preparedInference.tokens.count
        )

        LiveCounters.shared.requestPhaseChanged(requestId: requestId, phase: .prefilling)

        let cancellation = CancellationToken()
        let streamHandle: InferenceEngine.StreamHandle
        do {
            streamHandle = try await engine.stream(
                InferenceEngine.InferenceRequest(
                    input: preparedInference.lmInput,
                    tokens: preparedInference.tokens,
                    parameters: generateParams,
                    cachedKV: lease.kvCache,
                    cachedTokenCount: lease.matchedTokenCount
                ),
                cancellation: cancellation
            )
        } catch {
            LiveCounters.shared.requestCompleted(requestId: requestId, generationTokens: 0)
            sendResponse(connection: connection, status: 500, body: #"{"error":"\#(error.localizedDescription)"}"#)
            return
        }

        let result: GenerationOutcome
        if isStream {
            result = await handleStreamingResponse(
                connection: connection,
                requestId: requestId,
                cancellation: cancellation,
                stream: streamHandle.stream,
                tools: request.tools,
                created: created,
                modelName: modelName
            )
        } else {
            result = await handleNonStreamingResponse(
                connection: connection,
                requestId: requestId,
                stream: streamHandle.stream,
                tools: request.tools,
                created: created,
                modelName: modelName
            )
        }

        if let cacheKey,
           result.succeeded || result.cancelled {
            Self.storePromptCache(
                streamHandle.workingCache,
                promptTokenCount: preparedInference.tokens.count,
                entryId: lease.entryId,
                cacheKey: cacheKey,
                modelId: currentModelId
            )
        }

        LiveCounters.shared.requestCompleted(requestId: requestId, generationTokens: result.completionTokens)
        modelManager.touchActivity()
    }

    // MARK: - Non-streaming response

    private func handleNonStreamingResponse(
        connection: NWConnection,
        requestId: String,
        stream: AsyncStream<Generation>,
        tools: [APIToolDefinition]?,
        created: Int,
        modelName: String
    ) async -> GenerationOutcome {
        do {
            let outcome = await Self.collectGenerationOutcome(
                stream: stream,
                requestId: requestId,
                cancellation: nil
            )
            let resolved = Self.resolveAssistantResponse(
                fullText: outcome.fullText,
                frameworkToolCalls: outcome.frameworkToolCalls,
                tools: tools
            )

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
                            content: resolved.content,
                            tool_calls: resolved.toolCalls
                        ),
                        finish_reason: resolved.finishReason
                    )
                ],
                usage: APIUsageInfo(
                    prompt_tokens: outcome.promptTokens,
                    completion_tokens: outcome.completionTokens,
                    total_tokens: outcome.promptTokens + outcome.completionTokens
                )
            )

            if let json = try? JSONEncoder().encode(response) {
                sendResponse(connection: connection, status: 200, body: String(data: json, encoding: .utf8) ?? "{}")
            }
            return GenerationOutcome(
                promptTokens: outcome.promptTokens,
                completionTokens: outcome.completionTokens,
                fullText: outcome.fullText,
                frameworkToolCalls: outcome.frameworkToolCalls,
                succeeded: true,
                cancelled: false
            )
        } catch {
            sendResponse(connection: connection, status: 500, body: #"{"error":"\#(error.localizedDescription)"}"#)
            return GenerationOutcome(promptTokens: 0, completionTokens: 0, fullText: "", frameworkToolCalls: [], succeeded: false, cancelled: false)
        }
    }

    // MARK: - Streaming (SSE) response

    private func handleStreamingResponse(
        connection: NWConnection,
        requestId: String,
        cancellation: CancellationToken,
        stream: AsyncStream<Generation>,
        tools: [APIToolDefinition]?,
        created: Int,
        modelName: String
    ) async -> GenerationOutcome {
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
        connection.stateUpdateHandler = { state in
            switch state {
            case .cancelled, .failed:
                cancellation.cancel()
            default:
                break
            }
        }

        let encoder = StreamingSSEEncoder(requestId: requestId, created: created, modelName: modelName)
        await Self.sendData(connection: connection, data: encoder.encodeRoleDelta("assistant"))

        let result = await Self.runStreamingLoop(
            connection: connection,
            stream: stream,
            cancellation: cancellation,
            requestId: requestId,
            encoder: encoder
        )

        if result.cancelled {
            connection.cancel()
            return result
        }

        let resolved = Self.resolveAssistantResponse(
            fullText: result.fullText,
            frameworkToolCalls: result.frameworkToolCalls,
            tools: tools
        )

        if let toolCalls = resolved.toolCalls {
            for apiToolCall in toolCalls {
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
            choices: [APIStreamChoice(index: 0, delta: APIDeltaMessage(role: nil, content: nil, tool_calls: nil), finish_reason: resolved.finishReason)],
            usage: APIUsageInfo(
                prompt_tokens: result.promptTokens,
                completion_tokens: result.completionTokens,
                total_tokens: result.promptTokens + result.completionTokens
            )
        ))

        // Send [DONE] and close
        await Self.sendData(connection: connection, data: "data: [DONE]\n\n".data(using: .utf8)!)
        connection.cancel()
        return result
    }

    /// Run the token generation + SSE send loop entirely off MainActor.
    /// This is critical: if the loop runs on MainActor, every token requires
    /// multiple actor hops competing with SwiftUI, causing all output to batch.
    nonisolated private static func runStreamingLoop(
        connection: NWConnection,
        stream: AsyncStream<Generation>,
        cancellation: CancellationToken,
        requestId: String,
        encoder: StreamingSSEEncoder
    ) async -> GenerationOutcome {
        var outcome = await collectGenerationOutcome(
            stream: stream,
            requestId: requestId,
            cancellation: cancellation
        ) { text in
            await sendData(connection: connection, data: encoder.encodeContentDelta(text))
        }
        outcome.succeeded = !outcome.cancelled
        return outcome
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

    nonisolated private static func collectGenerationOutcome(
        stream: AsyncStream<Generation>,
        requestId: String,
        cancellation: CancellationToken?,
        onChunk: ((String) async -> Void)? = nil
    ) async -> GenerationOutcome {
        var promptTokens = 0
        var completionTokens = 0
        var fullText = ""
        var frameworkToolCalls: [MLXLMCommon.ToolCall] = []
        var cancelled = false

        for await generation in stream {
            if let cancellation, cancellation.isCancelled {
                cancelled = true
                break
            }

            switch generation {
            case .chunk(let text):
                completionTokens += 1
                fullText += text
                LiveCounters.shared.tokenGenerated(tokensPerSecond: 0, totalGenerated: completionTokens)
                if let onChunk {
                    await onChunk(text)
                }
            case .info(let info):
                promptTokens = info.promptTokenCount
                completionTokens = info.generationTokenCount
                LiveCounters.shared.prefillCompleted(requestId: requestId, promptTokens: promptTokens)
                if info.tokensPerSecond > 0 {
                    LiveCounters.shared.tokenGenerated(tokensPerSecond: info.tokensPerSecond, totalGenerated: completionTokens)
                }
            case .toolCall(let call):
                frameworkToolCalls.append(call)
            }
        }

        return GenerationOutcome(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            fullText: fullText,
            frameworkToolCalls: frameworkToolCalls,
            succeeded: !cancelled,
            cancelled: cancelled
        )
    }

    private static func storePromptCache(
        _ cache: [KVCache],
        promptTokenCount: Int,
        entryId: UUID,
        cacheKey: [Int],
        modelId: String
    ) {
        guard trimGeneratedTokens(cache, promptTokenCount: promptTokenCount) else {
            return
        }
        TokenPrefixCache.shared.store(
            entryId: entryId,
            kvCache: cache,
            cacheKey: cacheKey,
            modelId: modelId
        )
    }

    private static func trimGeneratedTokens(_ cache: [KVCache], promptTokenCount: Int) -> Bool {
        for layer in cache {
            let excess = layer.offset - promptTokenCount
            guard excess <= 0 || layer.isTrimmable else {
                return false
            }
            if excess > 0 {
                let trimmed = layer.trim(excess)
                guard trimmed == excess else {
                    return false
                }
            }
        }
        return true
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

    private static func messageSignature(role: Chat.Message.Role, content: String, imageURLs: [String]) -> UInt64 {
        var hash: UInt64 = 14_695_981_039_346_656_037

        func mix(_ text: String) {
            for byte in text.utf8 {
                hash ^= UInt64(byte)
                hash &*= 1_099_511_628_211
            }
        }

        switch role {
        case .assistant:
            mix("assistant")
        case .system:
            mix("system")
        case .user:
            mix("user")
        @unknown default:
            mix("unknown")
        }
        mix("|")
        mix(content)
        for imageURL in imageURLs {
            mix("|")
            mix(imageURL)
        }

        return hash
    }

    private static func normalizedAssistantHistoryContent(
        content: String?,
        toolCalls: [APIToolCall]?,
        isQwen: Bool
    ) -> String? {
        var text = content?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        if let toolCalls, !toolCalls.isEmpty {
            let formattedCalls = isQwen
                ? ToolPromptBuilder.formatQwenToolCalls(toolCalls)
                : ToolPromptBuilder.formatGemmaToolCalls(toolCalls)
            text = text.isEmpty ? formattedCalls : text + "\n" + formattedCalls
        }
        return text.isEmpty ? nil : text
    }

    private static func resolveAssistantResponse(
        fullText: String,
        frameworkToolCalls: [MLXLMCommon.ToolCall],
        tools: [APIToolDefinition]?
    ) -> (content: String?, toolCalls: [APIToolCall]?, finishReason: String) {
        var finishReason = "stop"
        var responseContent: String? = fullText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : fullText
        var apiToolCalls: [APIToolCall]? = nil

        if !frameworkToolCalls.isEmpty {
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
        } else if let tools, !tools.isEmpty {
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

        return (responseContent, apiToolCalls, finishReason)
    }
}

private struct GenerationOutcome {
    var promptTokens: Int
    var completionTokens: Int
    var fullText: String
    var frameworkToolCalls: [MLXLMCommon.ToolCall]
    var succeeded: Bool
    var cancelled: Bool
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
