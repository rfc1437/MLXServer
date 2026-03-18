import SwiftUI

struct SceneSelectionView: View {
    let scenes: [ChatScene]
    let activeSceneId: UUID?
    let currentModelName: String?
    let onSelectNeutral: () -> Void
    let onSelectScene: (ChatScene) -> Void
    let onManageScenes: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Start New Chat")
                .font(.headline)

            sceneButton(
                title: "Neutral",
                subtitle: currentModelName.map { "Keeps \($0) and only uses the base system prompt." }
                    ?? "Keeps the current model and only uses the base system prompt.",
                isSelected: activeSceneId == nil,
                action: onSelectNeutral
            )

            if !scenes.isEmpty {
                Divider()

                ForEach(scenes) { scene in
                    sceneButton(
                        title: scene.displayName,
                        subtitle: sceneSubtitle(for: scene),
                        isSelected: activeSceneId == scene.id,
                        action: { onSelectScene(scene) }
                    )
                }
            }

            Divider()

            Button("Manage Scenes…", action: onManageScenes)
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .frame(width: 320)
    }

    @ViewBuilder
    private func sceneButton(
        title: String,
        subtitle: String,
        isSelected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isSelected ? Color.accentColor : Color.secondary.opacity(0.45))
                    .padding(.top, 2)

                VStack(alignment: .leading, spacing: 3) {
                    Text(title)
                        .foregroundStyle(.primary)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.leading)
                }

                Spacer(minLength: 0)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10)
            .background(isSelected ? Color.accentColor.opacity(0.08) : Color.secondary.opacity(0.06))
            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
        }
        .buttonStyle(.plain)
    }

    private func sceneSubtitle(for scene: ChatScene) -> String {
        let modelText = scene.resolvedModel?.displayName ?? "Current model"
        if scene.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "\(modelText) • No extra scene prompt"
        }
        return "\(modelText) • Adds scene prompt"
    }
}