#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
CONFIG="${1:-Debug}"
APP_NAME="MLX Server"

echo "==> Building $APP_NAME ($CONFIG)"

# Regenerate Xcode project from project.yml (picks up any new/removed files)
if command -v xcodegen &>/dev/null; then
    xcodegen generate --spec "$PROJECT_DIR/project.yml" --project "$PROJECT_DIR" 2>&1 | grep -v '^$'
fi

# Build — filter to show only app source compilation, errors, and result
xcodebuild \
    -project "$PROJECT_DIR/MLXServer.xcodeproj" \
    -scheme MLXServer \
    -destination 'platform=macOS' \
    -configuration "$CONFIG" \
    SYMROOT="$BUILD_DIR" \
    build 2>&1 | \
    grep -E "(CompileSwift .* 'MLXServer'|error:|warning:.*MLXServer/|BUILD )" | \
    sed "s|.*CompileSwift normal arm64 Compiling ||" | \
    sed "s| (in target 'MLXServer' from project 'MLXServer')||"

APP_PATH="$BUILD_DIR/$CONFIG/$APP_NAME.app"

if [ -d "$APP_PATH" ] && [ -f "$APP_PATH/Contents/MacOS/$APP_NAME" ]; then
    echo ""
    echo "==> Build succeeded"
    echo "    $APP_PATH"
    echo ""
    echo "    Run:  open \"$APP_PATH\""
    echo "    Or:   \"$APP_PATH/Contents/MacOS/$APP_NAME\""
else
    echo ""
    echo "==> Build failed"
    exit 1
fi
