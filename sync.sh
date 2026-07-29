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
# only on HPC), evaluation/ (local analysis notebooks, never run on HPC)
# and .git (syncing it would let `down` overwrite your local git history
# with the cluster's).
#
# No --delete by default: syncing can only add or update files, never
# remove them. Pass --delete explicitly when you want a true mirror.

set -euo pipefail

REMOTE=hpc:/p/tmp/stefansi/NeuralParam.jl
LOCAL="$(cd "$(dirname "$0")" && pwd)"

EXCLUDES=(
  --exclude '.git/'
  --exclude 'data/reference/'
  --exclude 'evaluation/'
)

# -i lists one line per changed item; --omit-dir-times stops directories
# from counting as "changed" just because their timestamp moved.
# --no-o --no-g: Windows has no matching Unix owner/group, so without these
# rsync reports a spurious ownership change on every single file.
FLAGS=(-azh --partial --omit-dir-times --no-o --no-g -i)

MODE=${1:?Usage: sync.sh up|down [--dry-run] [--delete]}
shift || true

case "$MODE" in
  up)   rsync "${FLAGS[@]}" "${EXCLUDES[@]}" "$@" "$LOCAL/"  "$REMOTE/" ;;
  down) rsync "${FLAGS[@]}" "${EXCLUDES[@]}" "$@" "$REMOTE/" "$LOCAL/"  ;;
  *)    echo "Usage: bash sync.sh up|down [--dry-run] [--delete]" >&2; exit 1 ;;
esac





# ------------------------------ Explanation of rsync output ------------------------------
# Reading the -i (itemize) output: each line is an 11-character code, YXcstpoguax
#
#   >f+++++++++ path/to/file
#   ||\________/
#   ||    └─ which attributes differ
#   |└─ type of item
#   └─ direction / whether anything is transferred
#
# Char 1 — what happens:
#   >   file is being RECEIVED (down)
#   <   file is being SENT (up)
#   c   item is being created locally (directory, symlink)
#   .   nothing is transferred, only metadata differs
#   *   a message follows instead of a code, e.g. "*deleting"
#
# Char 2 — what it is:
#   f   file      d   directory      L   symlink
#
# Chars 3-11 — which attributes differ (c s t p o g u a x):
#   c   checksum / content differs
#   s   size differs
#   t   modification time differs
#   p   permissions differ
#   o   owner differs        (suppressed here by --no-o)
#   g   group differs        (suppressed here by --no-g)
#   u   access time differs
#   a   ACL differs          x   extended attributes differ
#   +   the item is brand new, so every attribute counts as new
#   .   this attribute is unchanged
#
# Common lines:
#   >f+++++++++   new file, transferred
#   >f..t......   existing file, only the timestamp changed -> re-sent
#   >f.st......   existing file, size and time changed -> updated
#   cd+++++++++   directory created
#   .f.....g...   NOTHING transferred, only group metadata (noise)
#   *deleting     file removed (only appears with --delete)
#
# Rule of thumb: a leading '.' means no data moves. Only '>' and '<' transfer.
# ---------------------------------------------------------------------------