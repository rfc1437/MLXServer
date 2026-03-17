import SwiftUI
import UniformTypeIdentifiers

struct ChatInputView: View {
    @Bindable var viewModel: ChatViewModel

    var body: some View {
        VStack(spacing: 8) {
            // Image preview strip
            if !viewModel.attachedImages.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(viewModel.attachedImages.enumerated()), id: \.offset) { index, image in
                            ZStack(alignment: .topTrailing) {
                                Image(nsImage: image)
                                    .resizable()
                                    .aspectRatio(contentMode: .fill)
                                    .frame(width: 60, height: 60)
                                    .clipShape(RoundedRectangle(cornerRadius: 8))

                                Button {
                                    viewModel.removeImage(at: index)
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .font(.caption)
                                        .foregroundStyle(.white)
                                        .background(Circle().fill(.black.opacity(0.5)))
                                }
                                .buttonStyle(.plain)
                                .offset(x: 4, y: -4)
                            }
                        }
                    }
                    .padding(.horizontal, 12)
                }
            }

            // Input row
            HStack(alignment: .bottom, spacing: 8) {
                // Image attach button
                Button {
                    pickImage()
                } label: {
                    Image(systemName: "photo.badge.plus")
                        .font(.title3)
                }
                .buttonStyle(.plain)
                .disabled(!viewModel.modelManager.isReady)

                // Text field
                TextField("Message…", text: $viewModel.inputText, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...8)
                    .onSubmit {
                        if !NSEvent.modifierFlags.contains(.shift) {
                            viewModel.send()
                        }
                    }

                // Send or Stop button
                if viewModel.isGenerating {
                    Button {
                        viewModel.stop()
                    } label: {
                        Image(systemName: "stop.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.red)
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut(.escape, modifiers: [])
                } else {
                    Button {
                        viewModel.send()
                    } label: {
                        Image(systemName: "arrow.up.circle.fill")
                            .font(.title2)
                            .foregroundStyle(Color.accentColor)
                    }
                    .buttonStyle(.plain)
                    .disabled(viewModel.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !viewModel.modelManager.isReady)
                    .keyboardShortcut(.return, modifiers: .command)
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
        }
        .padding(.top, 4)
        .onDrop(of: [.image], isTargeted: nil) { providers in
            for provider in providers {
                _ = provider.loadObject(ofClass: NSImage.self) { image, _ in
                    if let image = image as? NSImage {
                        Task { @MainActor in
                            viewModel.attachImage(image)
                        }
                    }
                }
            }
            return true
        }
        // Cmd+V paste for images
        .onPasteCommand(of: [.image, .png, .jpeg, .tiff]) { providers in
            for provider in providers {
                _ = provider.loadObject(ofClass: NSImage.self) { image, _ in
                    if let image = image as? NSImage {
                        Task { @MainActor in
                            viewModel.attachImage(image)
                        }
                    }
                }
            }
        }
    }

    private func pickImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.allowsMultipleSelection = true
        panel.canChooseDirectories = false

        if panel.runModal() == .OK {
            for url in panel.urls {
                if let image = NSImage(contentsOf: url) {
                    viewModel.attachImage(image)
                }
            }
        }
    }
}
