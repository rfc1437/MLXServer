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
        let cacheKey: [Int]
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

    func prepare(_ userInput: UserInput, imageFingerprints: [UInt64] = []) async throws -> PreparedInference {
        nonisolated(unsafe) let input = userInput
        let lmInput = try await container.prepare(input: input)
        nonisolated(unsafe) let preparedInput = lmInput
        let tokenArray: [Int] = await container.perform { _ in
            preparedInput.text.tokens.asArray(Int.self)
        }
        let cacheKey = await buildCacheKey(tokens: tokenArray, imageFingerprints: imageFingerprints)

        return PreparedInference(
            lmInput: lmInput,
            tokens: tokenArray,
            cacheKey: cacheKey,
            hasImages: userInput.images.count > 0
        )
    }

    private func buildCacheKey(tokens: [Int], imageFingerprints: [UInt64]) async -> [Int] {
        guard !imageFingerprints.isEmpty else {
            return tokens
        }

        let modelIdentifier = await container.configuration.name.lowercased()

        if modelIdentifier.contains("gemma"),
           let key = Self.buildGemmaCacheKey(tokens: tokens, imageFingerprints: imageFingerprints) {
            return key
        }

        return await container.perform { context in
            let visionStartTokens = context.tokenizer.encode(text: "<|vision_start|>")
            let imagePadTokens = context.tokenizer.encode(text: "<|image_pad|>")
            let visionEndTokens = context.tokenizer.encode(text: "<|vision_end|>")

            if let key = Self.buildQwenCacheKey(
                tokens: tokens,
                imageFingerprints: imageFingerprints,
                visionStartTokens: visionStartTokens,
                imagePadTokens: imagePadTokens,
                visionEndTokens: visionEndTokens
            ) {
                return key
            }

            return Self.buildFallbackVisionCacheKey(tokens: tokens, imageFingerprints: imageFingerprints)
        }
    }

    private static func buildGemmaCacheKey(tokens: [Int], imageFingerprints: [UInt64]) -> [Int]? {
        let imageTokenId = 262_144
        let totalImageTokenCount = tokens.reduce(into: 0) { count, token in
            if token == imageTokenId {
                count += 1
            }
        }

        guard totalImageTokenCount > 0,
              totalImageTokenCount % imageFingerprints.count == 0
        else {
            return nil
        }

        let tokensPerImage = totalImageTokenCount / imageFingerprints.count
        guard tokensPerImage > 0 else {
            return nil
        }

        var key: [Int] = []
        key.reserveCapacity(tokens.count + imageFingerprints.count * 2)

        var currentImageTokenCount = 0
        var currentImageIndex = 0

        for token in tokens {
            key.append(token)
            guard token == imageTokenId else { continue }

            currentImageTokenCount += 1
            if currentImageTokenCount == tokensPerImage,
               currentImageIndex < imageFingerprints.count {
                key.append(contentsOf: fingerprintSentinels(imageFingerprints[currentImageIndex]))
                currentImageIndex += 1
                currentImageTokenCount = 0
            }
        }

        guard currentImageIndex == imageFingerprints.count else {
            return nil
        }

        return key
    }

    private static func buildQwenCacheKey(
        tokens: [Int],
        imageFingerprints: [UInt64],
        visionStartTokens: [Int],
        imagePadTokens: [Int],
        visionEndTokens: [Int]
    ) -> [Int]? {
        guard !visionStartTokens.isEmpty,
              !imagePadTokens.isEmpty,
              !visionEndTokens.isEmpty
        else {
            return nil
        }

        var key: [Int] = []
        key.reserveCapacity(tokens.count + imageFingerprints.count * 2)

        var tokenIndex = 0
        var imageIndex = 0

        while tokenIndex < tokens.count {
            if matches(tokens: tokens, sequence: visionStartTokens, at: tokenIndex) {
                let imageRegionStart = tokenIndex
                var scanIndex = tokenIndex + visionStartTokens.count
                var sawImagePad = false

                while matches(tokens: tokens, sequence: imagePadTokens, at: scanIndex) {
                    sawImagePad = true
                    scanIndex += imagePadTokens.count
                }

                if sawImagePad,
                   matches(tokens: tokens, sequence: visionEndTokens, at: scanIndex),
                   imageIndex < imageFingerprints.count {
                    let imageRegionEnd = scanIndex + visionEndTokens.count
                    key.append(contentsOf: tokens[imageRegionStart..<imageRegionEnd])
                    key.append(contentsOf: fingerprintSentinels(imageFingerprints[imageIndex]))
                    tokenIndex = imageRegionEnd
                    imageIndex += 1
                    continue
                }
            }

            key.append(tokens[tokenIndex])
            tokenIndex += 1
        }

        guard imageIndex == imageFingerprints.count else {
            return nil
        }

        return key
    }

    private static func buildFallbackVisionCacheKey(tokens: [Int], imageFingerprints: [UInt64]) -> [Int] {
        var key: [Int] = []
        key.reserveCapacity(tokens.count + imageFingerprints.count * 2)
        for fingerprint in imageFingerprints {
            key.append(contentsOf: fingerprintSentinels(fingerprint))
        }
        key.append(contentsOf: tokens)
        return key
    }

    private static func fingerprintSentinels(_ fingerprint: UInt64) -> [Int] {
        let upper = Int(UInt32(truncatingIfNeeded: fingerprint >> 32))
        let lower = Int(UInt32(truncatingIfNeeded: fingerprint))
        return [-(upper + 1), -(lower + 1)]
    }

    private static func matches(tokens: [Int], sequence: [Int], at start: Int) -> Bool {
        guard start + sequence.count <= tokens.count else {
            return false
        }

        for (offset, token) in sequence.enumerated() where tokens[start + offset] != token {
            return false
        }

        return true
    }
}