import MarkdownUI
import SwiftUI

struct ChatMessagesView: View {
    let viewModel: ChatViewModel

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if viewModel.conversation.messages.isEmpty {
                        emptyState
                    } else {
                        ForEach(viewModel.conversation.messages) { message in
                            MessageBubbleView(message: message)
                                .id(message.id)
                        }
                    }
                    Color.clear
                        .frame(height: 1)
                        .id("bottom")
                }
                .padding()
            }
            .onChange(of: viewModel.conversation.messages.last?.content) {
                // During streaming, scroll without animation to avoid overlapping animations
                proxy.scrollTo("bottom", anchor: .bottom)
            }
            .onChange(of: viewModel.conversation.messages.count) {
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo("bottom", anchor: .bottom)
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Spacer()
            Image(systemName: "message")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text("Start a conversation")
                .font(.title3)
                .foregroundStyle(.secondary)
            if viewModel.modelManager.currentModel == nil {
                Text("Select a model from the toolbar to begin")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
        }
        .frame(maxWidth: .infinity, minHeight: 300)
    }

}

struct MessageBubbleView: View {
    let message: ChatMessage
    @State private var showThinking = false

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
                // Show attached images
                if !message.images.isEmpty {
                    HStack(spacing: 4) {
                        ForEach(Array(message.images.enumerated()), id: \.offset) { _, image in
                            Image(nsImage: image)
                                .resizable()
                                .aspectRatio(contentMode: .fill)
                                .frame(width: 80, height: 80)
                                .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }

                // Thinking block (collapsible)
                if !message.thinkingContent.isEmpty || message.isThinking {
                    thinkingView
                }

                // Message content
                if !message.content.isEmpty || (message.isStreaming && !message.isThinking) {
                    Group {
                        if message.role == .assistant {
                            Markdown(message.content + (message.isStreaming && !message.isThinking ? " ●" : ""))
                                .textSelection(.enabled)
                        } else {
                            Text(message.content)
                                .textSelection(.enabled)
                        }
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        message.role == .user
                            ? Color.accentColor.opacity(0.15)
                            : Color.secondary.opacity(0.1)
                    )
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }
            }

            if message.role == .assistant { Spacer(minLength: 60) }
        }
    }

    private var thinkingView: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeInOut(duration: 0.15)) {
                    showThinking.toggle()
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: showThinking ? "chevron.down" : "chevron.right")
                        .font(.caption2)
                    if message.isThinking {
                        ProgressView()
                            .controlSize(.mini)
                        Text("Thinking…")
                    } else {
                        Image(systemName: "brain")
                        Text("Thought")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            .buttonStyle(.plain)

            if showThinking {
                Text(message.thinkingContent + (message.isThinking ? " ●" : ""))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .textSelection(.enabled)
                    .padding(.top, 4)
                    .padding(.leading, 14)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.purple.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}
