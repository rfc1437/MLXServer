import SwiftUI

struct ModelManagementView: View {
    @Environment(ModelManager.self) private var modelManager

    @State private var newRepoId = ""
    @State private var pendingDelete: ModelConfig?
    @State private var editingMetadataModel: ModelConfig?
    @FocusState private var isRepoIdFieldFocused: Bool

    private let sizeFormatter: ByteCountFormatter = {
        let formatter = ByteCountFormatter()
        formatter.allowedUnits = [.useGB, .useMB, .useKB]
        formatter.countStyle = .file
        formatter.includesUnit = true
        formatter.isAdaptive = true
        return formatter
    }()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                GroupBox("Add Model") {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Enter a HuggingFace model ID. The app will download it, load it once, and then keep it available in the regular model picker.")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        HStack {
                            TextField("owner/repo", text: $newRepoId)
                                .textFieldStyle(.roundedBorder)
                                .focused($isRepoIdFieldFocused)
                                .onSubmit {
                                    downloadEnteredModel()
                                }

                            Button("Download & Select") {
                                downloadEnteredModel()
                            }
                            .disabled(modelManager.isLoading || newRepoId.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }
                    }
                }

                GroupBox("Recommended Defaults") {
                    VStack(spacing: 0) {
                        ForEach(modelManager.curatedModels) { model in
                            curatedRow(model)
                            if model.id != modelManager.curatedModels.last?.id {
                                Divider()
                            }
                        }
                    }
                }

                GroupBox("Models On Disk") {
                    if modelManager.localModelsOnDisk.isEmpty {
                        ContentUnavailableView(
                            "No Local Models",
                            systemImage: "externaldrive",
                            description: Text("Downloaded models will appear here with their summed file sizes.")
                        )
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 20)
                    } else {
                        VStack(spacing: 0) {
                            ForEach(modelManager.localModelsOnDisk) { model in
                                localRow(model)
                                if model.id != modelManager.localModelsOnDisk.last?.id {
                                    Divider()
                                }
                            }
                        }
                    }
                }
            }
            .padding(20)
        }
        .navigationTitle("Models")
        .frame(minWidth: 760, minHeight: 520)
        .sheet(item: $editingMetadataModel) { model in
            ModelMetadataEditorView(
                model: model,
                baselineModel: modelManager.baselineModel(repoId: model.repoId) ?? model,
                detectedLocalModel: modelManager.discoveredLocalModelInfo(repoId: model.repoId),
                hasSavedOverride: Preferences.hasModelMetadataOverride(forRepoId: model.repoId),
                onSave: { override in
                    modelManager.saveMetadataOverride(override, for: model)
                },
                onReset: {
                    modelManager.clearMetadataOverride(for: model)
                }
            )
        }
        .alert(
            "Delete Local Model?",
            isPresented: Binding(
                get: { pendingDelete != nil },
                set: { if !$0 { pendingDelete = nil } }
            )
        ) {
            Button("Delete", role: .destructive) {
                if let pendingDelete {
                    modelManager.deleteModel(pendingDelete)
                }
                self.pendingDelete = nil
            }
            Button("Cancel", role: .cancel) {
                pendingDelete = nil
            }
        } message: {
            if let pendingDelete {
                Text("This removes the local files for \(pendingDelete.repoId).")
            }
        }
        .onAppear {
            modelManager.refreshAvailableModels()
            if newRepoId.isEmpty {
                isRepoIdFieldFocused = true
            }
        }
    }

    @ViewBuilder
    private func curatedRow(_ model: ModelConfig) -> some View {
        HStack(alignment: .top, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(model.displayName)
                        .font(.headline)
                    if modelManager.currentModel?.repoId == model.repoId {
                        Text("Loaded")
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(.green.opacity(0.15), in: Capsule())
                    }
                }

                Text(model.repoId)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Label(
                model.isLocal ? "On Disk" : "Not Downloaded",
                systemImage: model.isLocal ? "checkmark.circle.fill" : "arrow.down.circle"
            )
            .font(.caption)
            .foregroundStyle(model.isLocal ? .green : .secondary)

            Button(model.isLocal ? "Load" : "Download") {
                Task {
                    await modelManager.loadModel(model)
                }
            }
            .disabled(modelManager.isLoading)

            Button("Metadata…") {
                editingMetadataModel = model
            }
        }
        .padding(.vertical, 10)
    }

    @ViewBuilder
    private func localRow(_ model: ModelConfig) -> some View {
        HStack(alignment: .top, spacing: 14) {
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(model.displayName)
                        .font(.headline)
                    if !model.isCurated {
                        Text("Custom")
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(.secondary.opacity(0.14), in: Capsule())
                    }
                    if modelManager.currentModel?.repoId == model.repoId {
                        Text("Loaded")
                            .font(.caption.weight(.semibold))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(.green.opacity(0.15), in: Capsule())
                    }
                }

                Text(model.repoId)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if let localSizeBytes = model.localSizeBytes {
                Text(sizeFormatter.string(fromByteCount: localSizeBytes))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 90, alignment: .trailing)
            }

            Button("Load") {
                Task {
                    await modelManager.loadModel(model)
                }
            }
            .disabled(modelManager.isLoading)

            Button("Metadata…") {
                editingMetadataModel = model
            }

            Button("Delete", role: .destructive) {
                pendingDelete = model
            }
            .disabled(modelManager.isLoading)
        }
        .padding(.vertical, 10)
    }

    private func downloadEnteredModel() {
        let repoId = newRepoId.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repoId.isEmpty else { return }

        Task {
            await modelManager.addModel(repoId: repoId)
            if modelManager.errorMessage == nil {
                newRepoId = ""
            }
        }
    }
}

private struct ModelMetadataEditorView: View {
    @Environment(\.dismiss) private var dismiss

    let model: ModelConfig
    let baselineModel: ModelConfig
    let detectedLocalModel: LocalModelResolver.LocalModelInfo?
    let hasSavedOverride: Bool
    let onSave: (ModelMetadataOverride) -> Void
    let onReset: () -> Void

    @State private var contextLengthText: String
    @State private var primaryLoaderKind: ModelConfig.LoaderKind
    @State private var supportsImages: Bool
    @State private var supportsTools: Bool

    init(
        model: ModelConfig,
        baselineModel: ModelConfig,
        detectedLocalModel: LocalModelResolver.LocalModelInfo?,
        hasSavedOverride: Bool,
        onSave: @escaping (ModelMetadataOverride) -> Void,
        onReset: @escaping () -> Void
    ) {
        self.model = model
        self.baselineModel = baselineModel
        self.detectedLocalModel = detectedLocalModel
        self.hasSavedOverride = hasSavedOverride
        self.onSave = onSave
        self.onReset = onReset
        _contextLengthText = State(initialValue: String(model.contextLength))
        _primaryLoaderKind = State(initialValue: model.primaryLoaderKind)
        _supportsImages = State(initialValue: model.supportsImages)
        _supportsTools = State(initialValue: model.supportsTools)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Metadata") {
                    TextField("Context length", text: $contextLengthText)
                        .textFieldStyle(.roundedBorder)

                    Picker("Primary loader", selection: $primaryLoaderKind) {
                        ForEach(ModelConfig.LoaderKind.allCases, id: \.self) { loaderKind in
                            Text(loaderKind.displayName).tag(loaderKind)
                        }
                    }

                    Toggle("Supports images", isOn: $supportsImages)
                    Toggle("Supports tools", isOn: $supportsTools)
                }

                Section("Comparison") {
                    Text(defaultsSummary)
                        .foregroundStyle(.secondary)

                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                        GridRow {
                            Text("")
                            Text("Effective")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                            Text(baselineHeading)
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                        }

                        comparisonRow(
                            label: "Context",
                            effective: currentOverride?.contextLength.description ?? "Invalid",
                            baseline: baselineModel.contextLength > 0 ? "\(baselineModel.contextLength)" : "Unknown"
                        )
                        comparisonRow(
                            label: "Loader",
                            effective: primaryLoaderKind.displayName,
                            baseline: baselineModel.primaryLoaderKind.displayName
                        )
                        comparisonRow(
                            label: "Images",
                            effective: yesNo(supportsImages),
                            baseline: yesNo(baselineModel.supportsImages)
                        )
                        comparisonRow(
                            label: "Tools",
                            effective: yesNo(supportsTools),
                            baseline: yesNo(baselineModel.supportsTools)
                        )
                    }
                }

                if let detectedLocalModel {
                    Section("Discovered Source") {
                        LabeledContent("Detected context") {
                            Text(detectedLocalModel.contextLength > 0 ? "\(detectedLocalModel.contextLength)" : "Unknown")
                        }
                        LabeledContent("Detected loader order") {
                            Text(detectedLocalModel.loaderKinds.map(\.displayName).joined(separator: ", "))
                        }
                        LabeledContent("Detected vision") {
                            Text(yesNo(detectedLocalModel.supportsImages))
                        }
                    }
                }
            }
            .formStyle(.grouped)
            .navigationTitle(model.displayName)
            .frame(minWidth: 520, minHeight: 380)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        dismiss()
                    }
                }

                ToolbarItem(placement: .primaryAction) {
                    Button("Save") {
                        guard let currentOverride else { return }
                        onSave(currentOverride)
                        dismiss()
                    }
                    .disabled(currentOverride == nil)
                }

                if hasSavedOverride {
                    ToolbarItem(placement: .automatic) {
                        Button("Reset to Detected") {
                            onReset()
                            dismiss()
                        }
                    }
                }
            }
        }
    }

    private var currentOverride: ModelMetadataOverride? {
        guard let contextLength = Int(contextLengthText.trimmingCharacters(in: .whitespacesAndNewlines)), contextLength >= 0 else {
            return nil
        }

        return ModelMetadataOverride(
            contextLength: contextLength,
            primaryLoaderKind: primaryLoaderKind,
            supportsImages: supportsImages,
            supportsTools: supportsTools
        )
    }

    private var defaultsSummary: String {
        if detectedLocalModel != nil {
            if hasSavedOverride {
                return "The editable fields show the effective overridden metadata. The comparison column shows the discovered baseline from the local model files."
            }
            return "The editable fields currently match the discovered baseline from the local model files. Save to store an override for this repo ID."
        }

        if model.isCurated {
            return hasSavedOverride
                ? "The editable fields show the effective overridden metadata. The comparison column shows the curated built-in baseline."
                : "The editable fields currently match the curated built-in baseline. Save to store an override for this repo ID."
        }

        if hasSavedOverride {
            return "The editable fields show the effective overridden metadata. The comparison column shows the inferred baseline for this repo ID."
        }

        return "The editable fields currently match the inferred baseline for this repo ID. Save to store an override."
    }

    private var baselineHeading: String {
        if detectedLocalModel != nil {
            return "Detected"
        }
        if model.isCurated {
            return "Built-in"
        }
        return "Inferred"
    }

    @ViewBuilder
    private func comparisonRow(label: String, effective: String, baseline: String) -> some View {
        GridRow {
            Text(label)
            Text(effective)
                .monospaced()
            Text(baseline)
                .foregroundStyle(.secondary)
                .monospaced()
        }
    }

    private func yesNo(_ value: Bool) -> String {
        value ? "Yes" : "No"
    }
}