#!/usr/bin/env bash
# Usage:
#   ./scripts/sync.sh push   — send trained_models/ to cluster
#   ./scripts/sync.sh pull   — pull trained_models/ from cluster
#   ./scripts/sync.sh status — compare trained_models/ with cluster
#
# Set CLUSTER_HOST and CLUSTER_PATH before running, or export them
# in your shell profile:
#   export CLUSTER_HOST=user@cluster.example.com
#   export CLUSTER_PATH=/path/to/project

set -euo pipefail

CLUSTER_HOST="${CLUSTER_HOST:-}"
CLUSTER_PATH="${CLUSTER_PATH:-}"

if [[ -z "$CLUSTER_HOST" || -z "$CLUSTER_PATH" ]]; then
  echo "Error: set CLUSTER_HOST and CLUSTER_PATH before running."
  echo "  export CLUSTER_HOST=user@cluster.example.com"
  echo "  export CLUSTER_PATH=/path/to/project"
  exit 1
fi

REMOTE="${CLUSTER_HOST}:${CLUSTER_PATH}"
RSYNC="rsync -avz --progress --human-readable"

case "${1:-}" in

  push)
    echo "==> Pushing trained_models/ to ${REMOTE}/trained_models/ ..."
    $RSYNC trained_models/ "${REMOTE}/trained_models/"
    ;;

  pull)
    echo "==> Pulling trained_models/ from ${REMOTE}/trained_models/ ..."
    $RSYNC "${REMOTE}/trained_models/" trained_models/
    ;;

  status)
    echo "==> Local trained_models/:"
    find trained_models/ -name '*.pth' | sort
    echo ""
    echo "==> Remote trained_models/:"
    ssh "$CLUSTER_HOST" "find ${CLUSTER_PATH}/trained_models/ -name '*.pth' | sort"
    ;;

  *)
    echo "Usage: $0 {push|pull|status}"
    exit 1
    ;;
esac
