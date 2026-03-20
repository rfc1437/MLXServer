import XCTest
@testable import MLX_Server

@MainActor
final class ChatViewModelTests: XCTestCase {
    func testGemmaChatViewModelSendProducesAssistantReply() async throws {
        let modelManager = ModelManager()
        let config = try XCTUnwrap(ModelConfig.resolve("gemma"))
        await modelManager.loadModel(config)
        defer { modelManager.unloadModel() }

        XCTAssertTrue(modelManager.isReady)

        let viewModel = ChatViewModel(modelManager: modelManager)
        viewModel.inputText = "Say hello in one word."
        viewModel.send()

        XCTAssertTrue(viewModel.isGenerating)

        try await waitUntil(timeoutSeconds: 15) {
            !viewModel.isGenerating
        }

        XCTAssertEqual(viewModel.conversation.messages.count, 2)
        XCTAssertEqual(viewModel.conversation.messages[0].role, .user)
        XCTAssertEqual(viewModel.conversation.messages[0].content, "Say hello in one word.")
        XCTAssertEqual(viewModel.conversation.messages[1].role, .assistant)
        XCTAssertFalse(viewModel.conversation.messages[1].sessionContent.isEmpty)
        XCTAssertGreaterThan(viewModel.promptTokens, 0)
    }

    private func waitUntil(
        timeoutSeconds: TimeInterval,
        intervalNanoseconds: UInt64 = 100_000_000,
        condition: @escaping @MainActor () -> Bool
    ) async throws {
        let deadline = Date().addingTimeInterval(timeoutSeconds)
        while Date() < deadline {
            if condition() {
                return
            }
            try await Task.sleep(nanoseconds: intervalNanoseconds)
        }
        XCTFail("Condition not met before timeout")
    }
}
