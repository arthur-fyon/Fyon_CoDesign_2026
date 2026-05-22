#!/usr/bin/env bash
# =============================================================================
#  clean.sh – Remove generated / ephemeral artefacts from the project.
#
#  Usage (from any directory):
#    bash experiments/lib/clean.sh [--dry-run]
#
#  What is removed:
#    - __pycache__ directories (project-wide)
#    - wandb/      directories (project-wide)
#    - logs/       directory contents
#    - configs/generated/ contents  (preserves .gitkeep)
#    - models/trained/ contents     (preserves .gitkeep)
#    - *_wandb_id.txt files inside models/trained/
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(cd "$SCRIPT_DIR/../.." && pwd)"

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

_remove() {
    # _remove <description> <path-or-glob...>
    local desc="$1"; shift
    for target in "$@"; do
        if [[ -e "$target" || -d "$target" ]]; then
            echo "  [remove] $target"
            [[ "$DRY_RUN" -eq 0 ]] && rm -rf "$target"
        fi
    done
}

_remove_find() {
    # _remove_find <description> <find-args...>
    local desc="$1"; shift
    echo "[clean] $desc"
    while IFS= read -r -d '' target; do
        echo "  [remove] $target"
        [[ "$DRY_RUN" -eq 0 ]] && rm -rf "$target"
    done < <(find "$BASE" "$@" -print0 2>/dev/null)
}

_remove_dir_contents() {
    # Keep the directory itself and any .gitkeep inside it.
    local desc="$1"
    local dir="$2"
    echo "[clean] $desc  ($dir)"
    if [[ -d "$dir" ]]; then
        find "$dir" -mindepth 1 -not -name '.gitkeep' -print0 2>/dev/null \
        | while IFS= read -r -d '' target; do
            echo "  [remove] $target"
            [[ "$DRY_RUN" -eq 0 ]] && rm -rf "$target"
        done
    else
        echo "  (directory not found, skipping)"
    fi
}

echo "============================================================"
echo "  clean.sh  –  base: $BASE"
[[ "$DRY_RUN" -eq 1 ]] && echo "  DRY-RUN mode: nothing will be deleted"
echo "============================================================"

_remove_find "__pycache__ directories"   -type d -name '__pycache__'
_remove_find "wandb directories"         -type d -name 'wandb'
_remove_dir_contents "logs"              "$BASE/logs"
_remove_dir_contents "configs/generated" "$BASE/configs/generated"
_remove_dir_contents "models/trained"    "$BASE/models/trained"
_remove_dir_contents "models/wandb_ids"  "$BASE/models/wandb_ids"

echo ""
echo "[clean] Done."
