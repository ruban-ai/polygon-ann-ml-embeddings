#!/usr/bin/env bash
# Copy custom nmslib (WeightedJaccard) from DGX to Orion/local.
#
# Run from Orion (or any machine that can SSH to the DGX):
#   bash copy_nmslib_weighted_from_dgx.sh
#
# Optional env overrides:
#   DGX_HOST=ruban@dgxa100.cs.utsarr.edu
#   DGX_SRC=/raid/ruban/nmslib_weighted
#   LOCAL_DEST=/mnt/data1/ruban/nmslib_weighted

set -euo pipefail

DGX_HOST="${DGX_HOST:-ruban@dgxa100.cs.utsarr.edu}"
DGX_SRC="${DGX_SRC:-/raid/ruban/nmslib_weighted}"
LOCAL_DEST="${LOCAL_DEST:-/mnt/data1/ruban/nmslib_weighted}"

echo "Source:      ${DGX_HOST}:${DGX_SRC}"
echo "Destination: ${LOCAL_DEST}"
echo

ssh -o ConnectTimeout=15 "${DGX_HOST}" "test -d '${DGX_SRC}' && du -sh '${DGX_SRC}'"

mkdir -p "$(dirname "${LOCAL_DEST}")"
rsync -avz --progress "${DGX_HOST}:${DGX_SRC}/" "${LOCAL_DEST}/"

echo
echo "Done. Installed tree at: ${LOCAL_DEST}"
echo
echo "Install into your Python env (example):"
echo "  cd ${LOCAL_DEST}"
echo "  pip install -e ."
echo
echo "If notebooks expect /raid/ruban/nmslib_weighted, either:"
echo "  sudo mkdir -p /raid/ruban && sudo ln -sfn ${LOCAL_DEST} /raid/ruban/nmslib_weighted"
echo "or set PYTHONPATH before running notebooks."
