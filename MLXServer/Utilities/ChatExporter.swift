import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// A FileDocument that exports a chat conversation as Markdown or RTF.
struct ChatExportDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.plainText] }
    static var writableContentTypes: [UTType] {
        [UTType(filenameExtension: "md") ?? .plainText, .rtf]
    }

    let messages: [ChatMessage]
    let modelName: String?

    init(messages: [ChatMessage], modelName: String?) {
        self.messages = messages
        self.modelName = modelName
    }

    init(configuration: ReadConfiguration) throws {
        self.messages = []
        self.modelName = nil
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let contentType = configuration.contentType

        if contentType == .rtf, let data = ChatExporter.exportRTF(messages: messages, modelName: modelName) {
            return FileWrapper(regularFileWithContents: data)
        } else {
            let md = ChatExporter.exportMarkdown(messages: messages, modelName: modelName)
            return FileWrapper(regularFileWithContents: Data(md.utf8))
        }
    }
}

/// Exports a chat conversation to Markdown or RTF (Pages-compatible) format.
enum ChatExporter {

    // MARK: - Markdown export

    static func exportMarkdown(messages: [ChatMessage], modelName: String?) -> String {
        var lines: [String] = []

        // Header
        lines.append("# Chat Session")
        if let modelName {
            lines.append("**Model:** \(modelName)")
        }
        let formatter = DateFormatter()
        formatter.dateStyle = .long
        formatter.timeStyle = .short
        if let first = messages.first {
            lines.append("**Date:** \(formatter.string(from: first.timestamp))")
        }
        lines.append("")
        lines.append("---")
        lines.append("")

        for message in messages {
            guard message.role != .system else { continue }

            if message.role == .user {
                // User messages as blockquotes
                lines.append("**You:**")
                lines.append("")
                for line in message.content.components(separatedBy: "\n") {
                    lines.append("> \(line)")
                }
            } else {
                // Assistant messages: carry over original markdown
                lines.append("**Assistant:**")
                lines.append("")
                lines.append(message.content)
            }

            lines.append("")
            lines.append("---")
            lines.append("")
        }

        return lines.joined(separator: "\n")
    }

    // MARK: - RTF export

    static func exportRTF(messages: [ChatMessage], modelName: String?) -> Data? {
        let doc = NSMutableAttributedString()

        let bodyFont = NSFont.systemFont(ofSize: 13)
        let bodyBoldFont = NSFont.boldSystemFont(ofSize: 13)
        let titleFont = NSFont.boldSystemFont(ofSize: 20)
        let metaFont = NSFont.systemFont(ofSize: 11)
        let codeFont = NSFont.monospacedSystemFont(ofSize: 12, weight: .regular)

        let bodyParagraph = NSMutableParagraphStyle()
        bodyParagraph.paragraphSpacing = 8
        bodyParagraph.lineSpacing = 2

        let userParagraph = NSMutableParagraphStyle()
        userParagraph.paragraphSpacing = 8
        userParagraph.lineSpacing = 2
        userParagraph.headIndent = 20
        userParagraph.firstLineHeadIndent = 20

        // Title
        doc.append(NSAttributedString(
            string: "Chat Session\n",
            attributes: [.font: titleFont, .paragraphStyle: bodyParagraph]
        ))

        // Metadata
        let formatter = DateFormatter()
        formatter.dateStyle = .long
        formatter.timeStyle = .short
        var metaText = ""
        if let modelName { metaText += "Model: \(modelName)  " }
        if let first = messages.first {
            metaText += "Date: \(formatter.string(from: first.timestamp))"
        }
        if !metaText.isEmpty {
            doc.append(NSAttributedString(
                string: metaText + "\n\n",
                attributes: [.font: metaFont, .foregroundColor: NSColor.secondaryLabelColor]
            ))
        }

        for message in messages {
            guard message.role != .system else { continue }

            if message.role == .user {
                doc.append(NSAttributedString(
                    string: "You\n",
                    attributes: [
                        .font: bodyBoldFont,
                        .foregroundColor: NSColor.systemBlue,
                    ]
                ))
                doc.append(NSAttributedString(
                    string: message.content + "\n\n",
                    attributes: [
                        .font: bodyFont,
                        .paragraphStyle: userParagraph,
                        .foregroundColor: NSColor.labelColor,
                    ]
                ))
            } else {
                doc.append(NSAttributedString(
                    string: "Assistant\n",
                    attributes: [
                        .font: bodyBoldFont,
                        .foregroundColor: NSColor.labelColor,
                    ]
                ))
                let rendered = renderMarkdown(message.content, bodyFont: bodyFont, codeFont: codeFont, paragraph: bodyParagraph)
                doc.append(rendered)
                doc.append(NSAttributedString(string: "\n\n"))
            }

            doc.append(NSAttributedString(
                string: "\n",
                attributes: [
                    .strikethroughStyle: NSUnderlineStyle.single.rawValue,
                    .strikethroughColor: NSColor.separatorColor,
                    .font: NSFont.systemFont(ofSize: 4),
                ]
            ))
        }

        return doc.rtf(from: NSRange(location: 0, length: doc.length), documentAttributes: [
            .documentType: NSAttributedString.DocumentType.rtf,
        ])
    }

    // MARK: - Markdown → NSAttributedString (basic)

    private static func renderMarkdown(
        _ text: String,
        bodyFont: NSFont,
        codeFont: NSFont,
        paragraph: NSParagraphStyle
    ) -> NSAttributedString {
        let result = NSMutableAttributedString()
        let lines = text.components(separatedBy: "\n")
        var inCodeBlock = false
        var codeBlockLines: [String] = []

        for line in lines {
            if line.hasPrefix("```") {
                if inCodeBlock {
                    let code = codeBlockLines.joined(separator: "\n")
                    let codePara = NSMutableParagraphStyle()
                    codePara.paragraphSpacing = 4
                    codePara.headIndent = 12
                    codePara.firstLineHeadIndent = 12
                    result.append(NSAttributedString(
                        string: code + "\n",
                        attributes: [
                            .font: codeFont,
                            .foregroundColor: NSColor.secondaryLabelColor,
                            .backgroundColor: NSColor.quaternaryLabelColor,
                            .paragraphStyle: codePara,
                        ]
                    ))
                    codeBlockLines = []
                    inCodeBlock = false
                } else {
                    inCodeBlock = true
                }
                continue
            }

            if inCodeBlock {
                codeBlockLines.append(line)
                continue
            }

            if line.hasPrefix("### ") {
                result.append(NSAttributedString(
                    string: String(line.dropFirst(4)) + "\n",
                    attributes: [.font: NSFont.boldSystemFont(ofSize: 14), .paragraphStyle: paragraph]
                ))
            } else if line.hasPrefix("## ") {
                result.append(NSAttributedString(
                    string: String(line.dropFirst(3)) + "\n",
                    attributes: [.font: NSFont.boldSystemFont(ofSize: 15), .paragraphStyle: paragraph]
                ))
            } else if line.hasPrefix("# ") {
                result.append(NSAttributedString(
                    string: String(line.dropFirst(2)) + "\n",
                    attributes: [.font: NSFont.boldSystemFont(ofSize: 17), .paragraphStyle: paragraph]
                ))
            } else {
                let styled = applyInlineFormatting(line, bodyFont: bodyFont, codeFont: codeFont)
                result.append(styled)
                result.append(NSAttributedString(string: "\n", attributes: [.font: bodyFont]))
            }
        }

        return result
    }

    private static func applyInlineFormatting(
        _ text: String,
        bodyFont: NSFont,
        codeFont: NSFont
    ) -> NSAttributedString {
        let result = NSMutableAttributedString()
        var remaining = text[text.startIndex...]

        while !remaining.isEmpty {
            if remaining.hasPrefix("`"), let end = remaining.dropFirst().firstIndex(of: "`") {
                let code = String(remaining[remaining.index(after: remaining.startIndex)..<end])
                result.append(NSAttributedString(
                    string: code,
                    attributes: [
                        .font: codeFont,
                        .foregroundColor: NSColor.secondaryLabelColor,
                        .backgroundColor: NSColor.quaternaryLabelColor,
                    ]
                ))
                remaining = remaining[remaining.index(after: end)...]
            } else if remaining.hasPrefix("**"), let end = remaining.dropFirst(2).range(of: "**") {
                let bold = String(remaining[remaining.index(remaining.startIndex, offsetBy: 2)..<end.lowerBound])
                result.append(NSAttributedString(
                    string: bold,
                    attributes: [.font: NSFont.boldSystemFont(ofSize: bodyFont.pointSize)]
                ))
                remaining = remaining[end.upperBound...]
            } else if remaining.hasPrefix("*"), let end = remaining.dropFirst().firstIndex(of: "*") {
                let italic = String(remaining[remaining.index(after: remaining.startIndex)..<end])
                result.append(NSAttributedString(
                    string: italic,
                    attributes: [.font: NSFontManager.shared.convert(bodyFont, toHaveTrait: .italicFontMask)]
                ))
                remaining = remaining[remaining.index(after: end)...]
            } else {
                let ch = remaining[remaining.startIndex]
                result.append(NSAttributedString(
                    string: String(ch),
                    attributes: [.font: bodyFont]
                ))
                remaining = remaining[remaining.index(after: remaining.startIndex)...]
            }
        }

        return result
    }
}
