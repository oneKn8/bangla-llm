#!/usr/bin/env bash
# Pack pipeline code into a tar.gz for SCP to a GCP VM.
#
# Creates pipeline_code.tar.gz containing only the files needed to run
# the normalize + SHA-256 dedup pipeline on a remote machine.
#
# Usage:
#   bash pack_pipeline.sh
#
# Output:
#   /home/oneknight/projects/bangla-llm/data-pipeline/pipeline_code.tar.gz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT="${SCRIPT_DIR}/pipeline_code.tar.gz"

# Files required for the pipeline to run
REQUIRED_FILES=(
    "pipeline_fast.py"
    "config.py"
    "processing/__init__.py"
    "processing/normalize.py"
    "processing/stats.py"
    "collectors/__init__.py"
    "collectors/hf_corpus.py"
    "collectors/wikipedia.py"
    "collectors/base.py"
)

# Verify all required files exist
echo "[pack] Checking required files..."
missing=0
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${SCRIPT_DIR}/${f}" ]]; then
        echo "[pack] ERROR: missing ${f}"
        missing=1
    fi
done

if [[ $missing -eq 1 ]]; then
    echo "[pack] Aborting due to missing files."
    exit 1
fi

echo "[pack] All files present. Creating archive..."

# Build tar.gz from the data-pipeline directory
# Files are stored with paths relative to data-pipeline/
tar -czf "${OUTPUT}" \
    -C "${SCRIPT_DIR}" \
    "${REQUIRED_FILES[@]}"

# Print summary
echo "[pack] Created: ${OUTPUT}"
echo "[pack] Contents:"
tar -tzf "${OUTPUT}" | while read -r line; do
    echo "  ${line}"
done

SIZE=$(stat --printf="%s" "${OUTPUT}" 2>/dev/null || stat -f "%z" "${OUTPUT}" 2>/dev/null)
echo "[pack] Size: $(( SIZE / 1024 )) KB"
echo ""
echo "[pack] To deploy to a GCP VM:"
echo "  gcloud compute scp ${OUTPUT} gcp_vm_setup.sh INSTANCE_NAME:~/"
echo "  gcloud compute ssh INSTANCE_NAME -- 'HF_TOKEN=hf_xxx bash ~/gcp_vm_setup.sh'"
