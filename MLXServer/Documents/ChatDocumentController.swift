import Foundation

@Observable
@MainActor
final class ChatDocumentController {
    static let shared = ChatDocumentController()

    private(set) var pendingOpenURLs: [URL] = []
    private(set) var openRequestNonce = UUID()

    func enqueueOpenRequests(_ urls: [URL]) {
        guard !urls.isEmpty else { return }
        pendingOpenURLs.append(contentsOf: urls)
        openRequestNonce = UUID()
    }

    func consumeNextOpenRequest() -> URL? {
        guard !pendingOpenURLs.isEmpty else { return nil }
        return pendingOpenURLs.removeFirst()
    }

    var hasPendingOpenRequests: Bool {
        !pendingOpenURLs.isEmpty
    }
}
