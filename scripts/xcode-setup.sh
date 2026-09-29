#!/usr/bin/env bash
set -euo pipefail

# Vestige Xcode Setup
# Gives Xcode's AI agent persistent memory in 30 seconds.
# https://github.com/samvallad33/vestige

VESTIGE_VERSION="latest"
BINARY_NAME="vestige-mcp"
MCP_CONFIG='.mcp.json'

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

print_step() { echo -e "\n${BLUE}▸${NC} ${BOLD}$1${NC}"; }
print_ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
print_warn() { echo -e "  ${YELLOW}!${NC} $1"; }
print_err()  { echo -e "  ${RED}✗${NC} $1"; }

echo -e "${BOLD}"
echo "  ╦  ╦┌─┐┌─┐┌┬┐┬┌─┐┌─┐"
echo "  ╚╗╔╝├┤ └─┐ │ ││ ┬├┤ "
echo "   ╚╝ └─┘└─┘ ┴ ┴└─┘└─┘"
echo -e "${NC}"
echo "  Memory for Xcode's AI Agent"
echo ""

# --- Step 1: Detect or install vestige-mcp ---
print_step "Checking for vestige-mcp..."

VESTIGE_PATH=""
for p in /usr/local/bin/vestige-mcp "$HOME/.local/bin/vestige-mcp" "$HOME/.cargo/bin/vestige-mcp"; do
    if [ -x "$p" ]; then
        VESTIGE_PATH="$p"
        break
    fi
done

if [ -n "$VESTIGE_PATH" ]; then
    VERSION=$("$VESTIGE_PATH" --version 2>/dev/null || echo "unknown")
    print_ok "Found: $VESTIGE_PATH ($VERSION)"
else
    print_warn "vestige-mcp not found. Installing..."

    ARCH=$(uname -m)
    OS=$(uname -s)

    if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
        TARBALL="vestige-mcp-aarch64-apple-darwin.tar.gz"
    elif [ "$OS" = "Darwin" ] && [ "$ARCH" = "x86_64" ]; then
        TARBALL="vestige-mcp-x86_64-apple-darwin.tar.gz"
    elif [ "$OS" = "Linux" ] && [ "$ARCH" = "x86_64" ]; then
        TARBALL="vestige-mcp-x86_64-unknown-linux-gnu.tar.gz"
    else
        print_err "Unsupported platform: $OS/$ARCH"
        echo "  Install manually: https://github.com/samvallad33/vestige#install"
        exit 1
    fi

    URL="https://github.com/samvallad33/vestige/releases/latest/download/$TARBALL"
    CHECKSUM_URL="${URL}.sha256"
    VESTIGE_TMPDIR=$(mktemp -d)
    trap 'rm -rf "$VESTIGE_TMPDIR"' EXIT

    echo "  Downloading $TARBALL..."
    curl -fsSL "$URL" -o "$VESTIGE_TMPDIR/$TARBALL"

    # Verify checksum if available
    if curl -fsSL "$CHECKSUM_URL" -o "$VESTIGE_TMPDIR/$TARBALL.sha256" 2>/dev/null; then
        echo "  Verifying checksum..."
        (cd "$VESTIGE_TMPDIR" && shasum -a 256 -c "$TARBALL.sha256")
        print_ok "Checksum verified"
    else
        print_warn "No checksum file found — skipping verification"
    fi

    # Extract and verify binary
    tar -xz -f "$VESTIGE_TMPDIR/$TARBALL" -C "$VESTIGE_TMPDIR"

    if [ ! -f "$VESTIGE_TMPDIR/vestige-mcp" ] || [ ! -s "$VESTIGE_TMPDIR/vestige-mcp" ]; then
        print_err "Download appears corrupt — vestige-mcp not found in tarball"
        exit 1
    fi

    if ! file "$VESTIGE_TMPDIR/vestige-mcp" | grep -q "Mach-O\|ELF"; then
        print_err "Downloaded file is not a valid binary"
        exit 1
    fi

    chmod +x "$VESTIGE_TMPDIR/vestige-mcp"

    # Prefer user-local install, fall back to /usr/local/bin with sudo
    INSTALL_DIR="$HOME/.local/bin"
    if [ -w "/usr/local/bin" ]; then
        INSTALL_DIR="/usr/local/bin"
    elif [ ! -d "$INSTALL_DIR" ]; then
        mkdir -p "$INSTALL_DIR"
    fi

    if [ "$INSTALL_DIR" = "/usr/local/bin" ] && [ ! -w "$INSTALL_DIR" ]; then
        echo ""
        echo "  Install location: $INSTALL_DIR (requires sudo)"
        read -rp "  Continue? (y/N): " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            INSTALL_DIR="$HOME/.local/bin"
            mkdir -p "$INSTALL_DIR"
            print_warn "Installing to $INSTALL_DIR instead"
        else
            sudo mv "$VESTIGE_TMPDIR/vestige-mcp" "$INSTALL_DIR/"
            [ -f "$VESTIGE_TMPDIR/vestige" ] && sudo mv "$VESTIGE_TMPDIR/vestige" "$INSTALL_DIR/"
            [ -f "$VESTIGE_TMPDIR/vestige-restore" ] && sudo mv "$VESTIGE_TMPDIR/vestige-restore" "$INSTALL_DIR/"
            [ -f "$VESTIGE_TMPDIR/vestige-upgrade" ] && sudo mv "$VESTIGE_TMPDIR/vestige-upgrade" "$INSTALL_DIR/"
        fi
    fi

    if [ "$INSTALL_DIR" != "/usr/local/bin" ] || [ -w "$INSTALL_DIR" ]; then
        mv "$VESTIGE_TMPDIR/vestige-mcp" "$INSTALL_DIR/" 2>/dev/null || true
        [ -f "$VESTIGE_TMPDIR/vestige" ] && mv "$VESTIGE_TMPDIR/vestige" "$INSTALL_DIR/" 2>/dev/null || true
        [ -f "$VESTIGE_TMPDIR/vestige-restore" ] && mv "$VESTIGE_TMPDIR/vestige-restore" "$INSTALL_DIR/" 2>/dev/null || true
        [ -f "$VESTIGE_TMPDIR/vestige-upgrade" ] && mv "$VESTIGE_TMPDIR/vestige-upgrade" "$INSTALL_DIR/" 2>/dev/null || true
    fi

    VESTIGE_PATH="$INSTALL_DIR/vestige-mcp"
    VERSION=$("$VESTIGE_PATH" --version 2>/dev/null || echo "unknown")
    print_ok "Installed: $VESTIGE_PATH ($VERSION)"
fi

# --- Helper: generate .mcp.json content ---
generate_mcp_json() {
    local cmd_path="$1"
    # Escape backslashes and double quotes for valid JSON
    local escaped_path
    escaped_path=$(printf '%s' "$cmd_path" | sed 's/\\/\\\\/g; s/"/\\"/g')
    cat << MCPEOF
{
  "mcpServers": {
    "vestige": {
      "type": "stdio",
      "command": "$escaped_path",
      "args": [],
      "env": {
        "PATH": "/usr/local/bin:/usr/bin:/bin"
      }
    }
  }
}
MCPEOF
}

# --- Step 2: Find or select project ---
print_step "Finding Xcode projects..."

PROJECT_DIR=""

if [ -n "${1:-}" ]; then
    # User passed a project path
    PROJECT_DIR="$(cd "$1" && pwd)"
    print_ok "Using: $PROJECT_DIR"
elif ls "$(pwd)/"*.xcodeproj >/dev/null 2>&1 || [ -f "$(pwd)/Package.swift" ]; then
    PROJECT_DIR="$(pwd)"
    print_ok "Current directory: $PROJECT_DIR"
else
    # Search for projects
    PROJECTS=$(find "$HOME/Developer" -maxdepth 4 \( -name "*.xcodeproj" -o -name "Package.swift" \) 2>/dev/null | head -20)

    if [ -z "$PROJECTS" ]; then
        print_warn "No Xcode projects found in ~/Developer"
        echo ""
        echo "  Usage: $0 /path/to/your/xcode/project"
        echo ""
        echo "  Or run from inside your project directory:"
        echo "  cd /path/to/project && $0"
        exit 1
    fi

    echo "  Found projects:"
    i=1
    declare -a PROJECT_LIST=()
    while IFS= read -r proj; do
        dir=$(dirname "$proj")
        if [ -d "$dir" ]; then
            PROJECT_LIST+=("$dir")
            echo "    $i) $dir"
            ((i++))
        fi
    done <<< "$PROJECTS"

    max_choice=$((i - 1))
    echo ""
    read -rp "  Select project (1-$max_choice), or 'a' for all: " choice

    if [ "$choice" = "a" ]; then
        for dir in "${PROJECT_LIST[@]}"; do
            if [ ! -f "$dir/$MCP_CONFIG" ]; then
                generate_mcp_json "$VESTIGE_PATH" > "$dir/$MCP_CONFIG"
                print_ok "$dir/$MCP_CONFIG"
            else
                print_warn "$dir/$MCP_CONFIG (already exists, skipped)"
            fi
        done

        echo ""
        echo -e "${GREEN}${BOLD}Done!${NC} Vestige added to ${#PROJECT_LIST[@]} projects."
        echo ""
        echo "  Restart Xcode (Cmd+Q) and type /context to verify."
        exit 0
    elif [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "$max_choice" ]; then
        idx=$((choice - 1))
        PROJECT_DIR="${PROJECT_LIST[$idx]}"
        print_ok "Selected: $PROJECT_DIR"
    else
        print_err "Invalid selection: $choice"
        exit 1
    fi
fi

# --- Step 3: Create .mcp.json ---
print_step "Creating $MCP_CONFIG..."

CONFIG_PATH="$PROJECT_DIR/$MCP_CONFIG"

if [ -f "$CONFIG_PATH" ]; then
    print_warn "$CONFIG_PATH already exists"
    read -rp "  Overwrite? (y/N): " overwrite
    if [ "$overwrite" != "y" ] && [ "$overwrite" != "Y" ]; then
        echo "  Skipped."
        exit 0
    fi
    cp "$CONFIG_PATH" "$CONFIG_PATH.bak"
    print_ok "Backed up existing config to $CONFIG_PATH.bak"
fi

generate_mcp_json "$VESTIGE_PATH" > "$CONFIG_PATH"
print_ok "Created $CONFIG_PATH"

# --- Step 4: Verify ---
print_step "Verifying vestige-mcp starts..."

if "$VESTIGE_PATH" --version >/dev/null 2>&1; then
    print_ok "vestige-mcp binary OK"
else
    print_err "vestige-mcp failed to start — check the binary"
    exit 1
fi

# --- Done ---
echo ""
echo -e "${GREEN}${BOLD}Vestige is ready for Xcode.${NC}"
echo ""
echo "  Next steps:"
echo "    1. Restart Xcode (Cmd+Q, then reopen)"
echo "    2. Open your project"
echo "    3. Type /context in the Agent panel"
echo "    4. You should see vestige listed with 19 tools"
echo ""
echo "  Try it:"
echo "    \"Remember that this project uses SwiftUI with MVVM architecture\""
echo "    (new session) → \"What architecture does this project use?\""
echo "    It remembers."
echo ""
echo -e "  ${BLUE}https://github.com/samvallad33/vestige${NC}"
echo ""
