import Foundation

@Observable
@MainActor
final class SceneStore {
    var scenes: [ChatScene]

    init() {
        self.scenes = Preferences.scenes
    }

    func addScene(copying scene: ChatScene? = nil) -> ChatScene {
        let nextScene: ChatScene
        if let scene {
            nextScene = ChatScene(
                name: scene.displayName,
                modelId: scene.modelId,
                systemPrompt: scene.systemPrompt,
                starterPrompt: scene.starterPrompt
            )
        } else {
            nextScene = .empty
        }
        scenes.append(nextScene)
        persist()
        return nextScene
    }

    func updateScene(id: UUID, _ mutate: (inout ChatScene) -> Void) {
        guard let index = scenes.firstIndex(where: { $0.id == id }) else { return }
        mutate(&scenes[index])
        persist()
    }

    func deleteScene(id: UUID) {
        scenes.removeAll { $0.id == id }
        persist()
    }

    func deleteScenes(ids: some Sequence<UUID>) {
        let idsToDelete = Set(ids)
        scenes.removeAll { idsToDelete.contains($0.id) }
        persist()
    }

    func deleteScenes(at offsets: IndexSet) {
        scenes.remove(atOffsets: offsets)
        persist()
    }

    func scene(id: UUID?) -> ChatScene? {
        guard let id else { return nil }
        return scenes.first(where: { $0.id == id })
    }

    private func persist() {
        Preferences.scenes = scenes
    }
}