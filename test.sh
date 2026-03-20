#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"
CONFIG="${1:-Debug}"
APP_NAME="MLX Server"
DESTINATION="${TEST_DESTINATION:-platform=macOS,arch=arm64}"

echo "==> Testing $APP_NAME ($CONFIG)"

# Regenerate Xcode project from project.yml (picks up any new/removed files)
if command -v xcodegen &>/dev/null; then
    xcodegen generate --spec "$PROJECT_DIR/project.yml" --project "$PROJECT_DIR" 2>&1 | grep -v '^$'
fi

# Run tests — filter to test progress, app warnings, build failures, and final result
xcodebuild \
    -project "$PROJECT_DIR/MLXServer.xcodeproj" \
    -scheme MLXServer \
    -destination "$DESTINATION" \
    -configuration "$CONFIG" \
    SYMROOT="$BUILD_DIR" \
    test 2>&1 | \
    grep -E "(Test Suite|Test Case|Executed [0-9]+ tests|Testing started|Testing failed|Testing passed|error:|warning:.*MLXServer/|\*\* TEST|BUILD )"

echo ""
echo "==> Tests passed"