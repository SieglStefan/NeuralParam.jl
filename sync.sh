#!/bin/bash
#
# Sync this repo with PIK HPC2024. Run from MSYS2 (needs rsync).
#
#   bash sync.sh up                 Windows -> HPC
#   bash sync.sh down               HPC -> Windows
#   bash sync.sh up --dry-run       preview, changes nothing
#   bash sync.sh up --delete        also remove files missing on the source
#
# Everything is synced EXCEPT data/reference (10+ GB, produced and used
# only on HPC) and purely local stuff (.git, .claude, sandbox).
#
# No --delete by default: syncing can only add or update files, never
# remove them. Pass --delete explicitly when you want a true mirror.

set -euo pipefail

REMOTE=hpc:/p/tmp/stefansi/NeuralParam.jl
LOCAL="$(cd "$(dirname "$0")" && pwd)"

EXCLUDES=(
  --exclude '.git/'
  --exclude '.claude/'
  --exclude 'sandbox/'
  --exclude 'data/reference/'
)

MODE=${1:?Usage: sync.sh up|down [--dry-run] [--delete]}
shift || true

case "$MODE" in
  up)   rsync -azhP "${EXCLUDES[@]}" "$@" "$LOCAL/"  "$REMOTE/" ;;
  down) rsync -azhP "${EXCLUDES[@]}" "$@" "$REMOTE/" "$LOCAL/"  ;;
  *)    echo "Usage: bash sync.sh up|down [--dry-run] [--delete]" >&2; exit 1 ;;
esac
