import SwiftUI

struct SceneManagementView: View {
    @Environment(SceneStore.self) private var sceneStore

    @State private var selectedSceneId: UUID?
    @State private var renamingSceneId: UUID?
    @State private var renameDraft = ""
    @State private var pendingDeleteSceneIDs: [UUID] = []
    @FocusState private var focusedRenameSceneId: UUID?

    var body: some View {
        NavigationSplitView {
            Group {
                if sceneStore.scenes.isEmpty {
                    ContentUnavailableView(
                        "No Scenes Yet",
                        systemImage: "theatermasks",
                        description: Text("Use the add button in the toolbar to create a scene.")
                    )
                } else {
                    List(selection: $selectedSceneId) {
                        ForEach(sceneStore.scenes) { scene in
                            VStack(alignment: .leading, spacing: 2) {
                                if renamingSceneId == scene.id {
                                    TextField("Scene Name", text: $renameDraft)
                                        .textFieldStyle(.roundedBorder)
                                        .focused($focusedRenameSceneId, equals: scene.id)
                                        .onSubmit {
                                            commitRename(for: scene.id)
                                        }
                                } else {
                                    Text(scene.displayName)
                                }

                                Text(scene.resolvedModel?.displayName ?? "Current model")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .tag(scene.id)
                            .onTapGesture(count: 2) {
                                beginRename(scene)
                            }
                            .contextMenu {
                                Button("Rename") {
                                    beginRename(scene)
                                }
                                Button("Duplicate") {
                                    duplicateScene(scene)
                                }
                                Divider()
                                Button("Delete", role: .destructive) {
                                    confirmDelete(sceneIDs: [scene.id])
                                }
                            }
                        }
                        .onDelete(perform: deleteScenes)
                    }
                    .navigationTitle("Scenes")
                    .listStyle(.sidebar)
                }
            }
        } detail: {
            if let selectedScene = sceneStore.scene(id: selectedSceneId) {
                SceneEditorView(scene: selectedScene)
            } else {
                ContentUnavailableView(
                    "No Scene Selected",
                    systemImage: "slider.horizontal.3",
                    description: Text("Select a scene in the sidebar or create one from the toolbar.")
                )
            }
        }
        .navigationTitle("Scenes")
        .frame(minWidth: 760, minHeight: 480)
        .toolbar {
            ToolbarItemGroup {
                Button {
                    createAndSelectScene()
                } label: {
                    Label("New Scene", systemImage: "plus")
                }

                Button {
                    duplicateSelectedScene()
                } label: {
                    Label("Duplicate Scene", systemImage: "plus.square.on.square")
                }
                .disabled(sceneStore.scene(id: selectedSceneId) == nil)

                Button(role: .destructive) {
                    if let selectedSceneId {
                        confirmDelete(sceneIDs: [selectedSceneId])
                    }
                } label: {
                    Label("Delete Scene", systemImage: "trash")
                }
                .disabled(sceneStore.scene(id: selectedSceneId) == nil)
            }
        }
        .confirmationDialog(
            deleteDialogTitle,
            isPresented: deleteConfirmationBinding,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                performConfirmedDelete()
            }
            Button("Cancel", role: .cancel) {
                pendingDeleteSceneIDs = []
            }
        } message: {
            Text(deleteDialogMessage)
        }
        .onAppear {
            if selectedSceneId == nil {
                selectedSceneId = sceneStore.scenes.first?.id
            }
        }
        .onChange(of: sceneStore.scenes.count) {
            if sceneStore.scene(id: selectedSceneId) == nil {
                selectedSceneId = sceneStore.scenes.first?.id
            }
        }
    }

    private func deleteScenes(at offsets: IndexSet) {
        let sceneIDs = offsets.map { sceneStore.scenes[$0].id }
        confirmDelete(sceneIDs: sceneIDs)
    }

    private func beginRename(_ scene: ChatScene) {
        selectedSceneId = scene.id
        renamingSceneId = scene.id
        renameDraft = scene.displayName
        focusedRenameSceneId = scene.id
    }

    private func commitRename(for id: UUID) {
        let trimmedName = renameDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        sceneStore.updateScene(id: id) {
            $0.name = trimmedName.isEmpty ? "Untitled Scene" : trimmedName
        }
        renamingSceneId = nil
        focusedRenameSceneId = nil
    }

    private func confirmDelete(sceneIDs: [UUID]) {
        pendingDeleteSceneIDs = sceneIDs
    }

    private func performConfirmedDelete() {
        let idsToDelete = Set(pendingDeleteSceneIDs)
        pendingDeleteSceneIDs = []
        sceneStore.deleteScenes(ids: idsToDelete)
        if let selectedSceneId, idsToDelete.contains(selectedSceneId) {
            self.selectedSceneId = sceneStore.scenes.first?.id
        }
        if let renamingSceneId, idsToDelete.contains(renamingSceneId) {
            self.renamingSceneId = nil
            focusedRenameSceneId = nil
            renameDraft = ""
        }
    }

    private func createAndSelectScene() {
        let created = sceneStore.addScene()
        selectedSceneId = created.id
    }

    private func duplicateSelectedScene() {
        guard let selectedScene = sceneStore.scene(id: selectedSceneId) else { return }
        duplicateScene(selectedScene)
    }

    private func duplicateScene(_ scene: ChatScene) {
        let duplicated = sceneStore.addScene(copying: scene)
        selectedSceneId = duplicated.id
    }

    private func deleteScene(_ id: UUID) {
        sceneStore.deleteScene(id: id)
        if selectedSceneId == id {
            selectedSceneId = sceneStore.scenes.first?.id
        }
    }

    private var deleteConfirmationBinding: Binding<Bool> {
        Binding(
            get: { !pendingDeleteSceneIDs.isEmpty },
            set: { isPresented in
                if !isPresented {
                    pendingDeleteSceneIDs = []
                }
            }
        )
    }

    private var deleteDialogTitle: String {
        pendingDeleteSceneIDs.count == 1 ? "Delete Scene?" : "Delete Scenes?"
    }

    private var deleteDialogMessage: String {
        if pendingDeleteSceneIDs.count == 1,
           let scene = sceneStore.scene(id: pendingDeleteSceneIDs.first) {
            return "\"\(scene.displayName)\" will be removed from your saved scenes."
        }
        return "\(pendingDeleteSceneIDs.count) scenes will be removed from your saved scenes."
    }
}

private struct SceneEditorView: View {
    @Environment(SceneStore.self) private var sceneStore

    let scene: ChatScene

    var body: some View {
        Form {
            Section("Details") {
                TextField("Name", text: binding(for: \.name))

                Picker("Model", selection: modelBinding) {
                    Text("Current model").tag(Optional<String>.none)
                    ForEach(ModelConfig.availableModels) { model in
                        Text(model.displayName).tag(Optional(model.id))
                    }
                }
            }

            Section("Scene Prompt") {
                TextEditor(text: binding(for: \.systemPrompt))
                    .font(.body.monospaced())
                    .frame(minHeight: 150)

                Text("Appended after the base system prompt when this scene starts a new chat.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Starter Prompt") {
                TextEditor(text: binding(for: \.starterPrompt))
                    .font(.body.monospaced())
                    .frame(minHeight: 120)

                Text("Sent automatically as the first user message when this scene starts a new chat.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Generation Overrides") {
                GenerationOverridesEditor(
                    overrides: generationOverridesBinding,
                    inheritedSettings: inheritedGenerationSettings,
                    inheritedSource: inheritedGenerationSource
                )
            }
        }
        .formStyle(.grouped)
        .navigationTitle(scene.displayName)
    }

    private var modelBinding: Binding<String?> {
        Binding(
            get: { sceneStore.scene(id: scene.id)?.modelId },
            set: { newValue in
                sceneStore.updateScene(id: scene.id) {
                    $0.modelId = newValue
                }
            }
        )
    }

    private func binding(for keyPath: WritableKeyPath<ChatScene, String>) -> Binding<String> {
        Binding(
            get: { sceneStore.scene(id: scene.id)?[keyPath: keyPath] ?? scene[keyPath: keyPath] },
            set: { newValue in
                sceneStore.updateScene(id: scene.id) {
                    $0[keyPath: keyPath] = newValue
                }
            }
        )
    }

    private var generationOverridesBinding: Binding<GenerationSettingsOverride> {
        Binding(
            get: { sceneStore.scene(id: scene.id)?.generationOverrides ?? scene.generationOverrides },
            set: { newValue in
                sceneStore.updateScene(id: scene.id) {
                    $0.generationOverrides = newValue
                }
            }
        )
    }

    private var effectiveModelId: String {
        sceneStore.scene(id: scene.id)?.modelId
            ?? scene.modelId
            ?? Preferences.defaultModelId
            ?? Preferences.lastModelId
            ?? ModelConfig.default.id
    }

    private var inheritedGenerationSettings: GenerationSettings {
        Preferences.generationSettings(forModelId: effectiveModelId)
    }

    private var inheritedGenerationSource: String {
        let modelName = ModelConfig.resolve(effectiveModelId)?.displayName ?? effectiveModelId
        if Preferences.hasGenerationSettings(forModelId: effectiveModelId) {
            return "saved \(modelName) defaults"
        }
        return "built-in \(modelName) defaults"
    }
}