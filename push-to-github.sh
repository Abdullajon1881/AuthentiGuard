#!/usr/bin/env bash
# Push AuthentiGuard to your GitHub account
# Usage: ./push-to-github.sh <github-username> [repo-name]
set -e

GITHUB_USER="${1:-}"
REPO_NAME="${2:-authentiguard}"

if [ -z "$GITHUB_USER" ]; then
    echo "Usage: ./push-to-github.sh <your-github-username> [repo-name]"
    echo "Example: ./push-to-github.sh johndoe authentiguard"
    exit 1
fi

echo "Creating GitHub repository: $GITHUB_USER/$REPO_NAME"
echo ""

# Option A: Use GitHub CLI (recommended)
if command -v gh >/dev/null 2>&1; then
    echo "Using GitHub CLI..."
    gh repo create "$REPO_NAME" \
        --private \
        --description "AI content authenticity detection platform — 5-modal (text/audio/video/image/code)" \
        --homepage "https://authentiguard.io"
    git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"
    git push -u origin main
    git push origin develop
    echo "✓ Pushed to https://github.com/$GITHUB_USER/$REPO_NAME"
else
    # Option B: Manual steps
    echo "GitHub CLI not found. Create the repo manually, then run:"
    echo ""
    echo "  git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
    echo "  git push -u origin main"
    echo "  git push origin develop"
    echo ""
    echo "Or create via API (requires a Personal Access Token in GH_TOKEN):"
    echo ""
    echo "  curl -X POST https://api.github.com/user/repos \\"
    echo "    -H \"Authorization: Bearer \$GH_TOKEN\" \\"
    echo "    -H \"Content-Type: application/json\" \\"
    echo "    -d '{\"name\":\"$REPO_NAME\",\"private\":true,\"description\":\"AI authenticity detection\"}'"
    echo ""
    echo "  git remote add origin https://github.com/$GITHUB_USER/$REPO_NAME.git"
    echo "  git push -u origin main"
    echo "  git push origin develop"
fi
