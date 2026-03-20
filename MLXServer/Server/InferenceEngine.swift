import MLX
import MLXLMCommon

/// Stateless inference wrapper for the API path.
final class InferenceEngine: @unchecked Sendable {
    private let container: ModelContainer

    init(container: ModelContainer) {
        self.container = container
    }

    struct InferenceRequest: @unchecked Sendable {
        let input: LMInput
        let tokens: [Int]
        let parameters: GenerateParameters
        let cachedKV: [KVCache]?
        let cachedTokenCount: Int
    }

    struct StreamHandle: @unchecked Sendable {
        let stream: AsyncStream<Generation>
        let workingCache: [KVCache]
    }

    struct PreparedInference: @unchecked Sendable {
        let lmInput: LMInput
        let tokens: [Int]
        let hasImages: Bool
    }

    func stream(
        _ request: InferenceRequest,
        cancellation: CancellationToken
    ) async throws -> StreamHandle {
        _ = cancellation
        nonisolated(unsafe) let input = request.input
        nonisolated(unsafe) let cachedKV = request.cachedKV
        let parameters = request.parameters

        return try await container.perform { context in
            let workingCache = cachedKV ?? context.model.newCache(parameters: parameters)
            let stream = try MLXLMCommon.generate(
                input: input,
                cache: workingCache,
                parameters: parameters,
                context: context
            )
            return StreamHandle(stream: stream, workingCache: workingCache)
        }
    }

    func prepare(_ userInput: UserInput) async throws -> PreparedInference {
        nonisolated(unsafe) let input = userInput
        let lmInput = try await container.prepare(input: input)
        nonisolated(unsafe) let preparedInput = lmInput
        let tokenArray: [Int] = await container.perform { _ in
            preparedInput.text.tokens.asArray(Int.self)
        }

        return PreparedInference(
            lmInput: lmInput,
            tokens: tokenArray,
            hasImages: userInput.images.count > 0
        )
    }
}