import SwiftUI
import UniformTypeIdentifiers

struct ChatInputView: View {
    @Bindable var viewModel: ChatViewModel
    @State private var pasteMonitor: Any?

    private var supportsImages: Bool {
        viewModel.modelManager.currentModel?.supportsImages == true
    }

    var body: some View {
        VStack(spacing: 8) {
            // Image preview strip
            if supportsImages && !viewModel.attachedImages.isEmpty {
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
                .disabled(!viewModel.modelManager.isReady || !supportsImages)

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
        .onDrop(of: [.image, .fileURL], isTargeted: nil) { providers in
            guard supportsImages else { return false }
            for provider in providers {
                if provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                    provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { data, _ in
                        guard let urlData = data as? Data,
                              let url = URL(dataRepresentation: urlData, relativeTo: nil),
                              let image = NSImage(contentsOf: url) else { return }
                        Task { @MainActor in
                            viewModel.attachImage(image)
                        }
                    }
                } else {
                    _ = provider.loadObject(ofClass: NSImage.self) { image, _ in
                        if let image = image as? NSImage {
                            Task { @MainActor in
                                viewModel.attachImage(image)
                            }
                        }
                    }
                }
            }
            return true
        }
        .onAppear { installPasteMonitor() }
        .onDisappear { removePasteMonitor() }
    }

    // MARK: - Paste monitor

    /// Intercepts Cmd+V before the TextField to handle image file URLs from Finder.
    /// If the pasteboard contains file URLs pointing to images, attaches them and
    /// consumes the event. Otherwise lets the TextField handle normal text paste.
    private func installPasteMonitor() {
        guard pasteMonitor == nil else { return }
        pasteMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { event in
            guard supportsImages else { return event }
            // Check for Cmd+V
            guard event.modifierFlags.contains(.command),
                  event.charactersIgnoringModifiers == "v" else {
                return event
            }

            let pasteboard = NSPasteboard.general
            let images = loadImagesFromPasteboard(pasteboard)
            guard !images.isEmpty else { return event }

            // Attach images and consume the event
            Task { @MainActor in
                for image in images {
                    viewModel.attachImage(image)
                }
            }
            return nil // consume the event
        }
    }

    private func removePasteMonitor() {
        if let monitor = pasteMonitor {
            NSEvent.removeMonitor(monitor)
            pasteMonitor = nil
        }
    }

    /// Tries to load images from the pasteboard.
    /// Handles: Finder file copies (file URLs), screenshot clipboard data, image data from other apps.
    private func loadImagesFromPasteboard(_ pasteboard: NSPasteboard) -> [NSImage] {
        var images: [NSImage] = []

        // 1. Check for file URLs (Finder copy)
        if let urls = pasteboard.readObjects(forClasses: [NSURL.self], options: [
            .urlReadingFileURLsOnly: true,
            .urlReadingContentsConformToTypes: [UTType.image.identifier],
        ]) as? [URL] {
            for url in urls {
                if let image = NSImage(contentsOf: url) {
                    images.append(image)
                }
            }
        }

        if !images.isEmpty { return images }

        // 2. Check for direct image data (screenshots, copy from Preview, etc.)
        if let image = NSImage(pasteboard: pasteboard) {
            images.append(image)
        }

        return images
    }

    // MARK: - File picker

    private func pickImage() {
        guard supportsImages else { return }
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
