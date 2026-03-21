import os

/// Thread-safe cancellation flag for cooperative stream shutdown.
final class CancellationToken: @unchecked Sendable {
    private let lock = OSAllocatedUnfairLock(initialState: false)

    var isCancelled: Bool {
        lock.withLock { $0 }
    }

    func cancel() {
        lock.withLock { $0 = true }
    }
}