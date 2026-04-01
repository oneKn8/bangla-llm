#!/usr/bin/env bash
# GCP VM setup and full pipeline execution for all major Bangla datasets.
#
# Collects and processes:
#   1. CulturaX  (28 GB Parquet, 12.4M docs) -> pipeline_fast
#   2. Sangraha  (16 GB Parquet, 5.5M docs)  -> pipeline_fast
#   3. CC-100    (860 MB .txt.xz)             -> pipeline_parallel (full)
#   4. Wikipedia (bnwiki + bnwikisource dumps) -> pipeline_parallel (full)
#
# Prerequisites:
#   - c4-standard-96 Spot VM, 500GB SSD, Ubuntu 22.04
#   - pipeline_code.tar.gz SCP'd to ~/pipeline_code.tar.gz
#   - HF_TOKEN env var set
#
# Create VM:
#   gcloud compute instances create bangla-pipeline \
#     --zone=us-central1-a \
#     --machine-type=c4-standard-96 \
#     --provisioning-model=SPOT \
#     --instance-termination-action=STOP \
#     --boot-disk-size=500GB \
#     --boot-disk-type=pd-ssd \
#     --image-family=ubuntu-2204-lts \
#     --image-project=ubuntu-os-cloud
#
# Usage:
#   HF_TOKEN=hf_xxx nohup bash ~/gcp_vm_setup.sh > ~/pipeline.log 2>&1 &

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PIPELINE_DIR="/opt/bangla-pipeline"
VENV_DIR="${PIPELINE_DIR}/venv"
CODE_DIR="${PIPELINE_DIR}/code"
DOWNLOADS_DIR="${PIPELINE_DIR}/downloads"
OUTPUT_DIR="${PIPELINE_DIR}/output"
TARBALL="${HOME}/pipeline_code.tar.gz"

FAST_WORKERS=6           # for pipeline_fast (CulturaX, Sangraha)
FULL_WORKERS=6           # for pipeline_parallel (CC-100, Wikipedia)
WIKI_EXTRACTOR_PROCS=4   # wikiextractor parallel processes

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
die() { log "FATAL: $*"; exit 1; }

elapsed_since() {
    local start=$1
    local now
    now=$(date +%s)
    echo "$(( (now - start) / 60 ))m $(( (now - start) % 60 ))s"
}

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
SCRIPT_START=$(date +%s)
log "=== Bangla Pipeline - GCP VM Setup ==="

[[ -z "${HF_TOKEN:-}" ]] && die "HF_TOKEN environment variable is not set"
[[ ! -f "${TARBALL}" ]] && die "Pipeline code not found at ${TARBALL}"

# ---------------------------------------------------------------------------
# Step 1: Install system dependencies
# ---------------------------------------------------------------------------
log "Step 1/7: Installing Python 3.12 + system deps"

sudo apt-get update -qq
sudo apt-get install -y -qq \
    software-properties-common build-essential wget xz-utils
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev

python3.12 --version || die "Python 3.12 installation failed"
log "  Python 3.12 installed"

# ---------------------------------------------------------------------------
# Step 2: Create venv + install Python deps
# ---------------------------------------------------------------------------
log "Step 2/7: Setting up Python environment"

sudo mkdir -p "${PIPELINE_DIR}" "${OUTPUT_DIR}" "${DOWNLOADS_DIR}"
sudo chown -R "$(whoami):$(whoami)" "${PIPELINE_DIR}"

python3.12 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel 2>&1 | tail -1

pip install \
    "numpy<2.0" \
    orjson \
    tqdm \
    datasets \
    huggingface_hub \
    bnunicodenormalizer \
    requests \
    wikiextractor \
    pyarrow \
    fasttext-wheel \
    2>&1 | tail -5

log "  Dependencies installed"

# ---------------------------------------------------------------------------
# Step 3: Extract pipeline code + HF login
# ---------------------------------------------------------------------------
log "Step 3/7: Extracting pipeline code"

mkdir -p "${CODE_DIR}"
tar -xzf "${TARBALL}" -C "${CODE_DIR}"

# Verify critical files exist
for f in pipeline_fast.py pipeline_parallel.py config.py convert_to_jsonl.py \
         processing/normalize.py processing/lang_detect.py \
         processing/quality.py processing/dedup.py processing/stats.py \
         collectors/wikipedia.py collectors/base.py; do
    [[ -f "${CODE_DIR}/${f}" ]] || die "Missing file after extraction: ${f}"
done
log "  Pipeline code extracted to ${CODE_DIR}"

log "  Logging in to HuggingFace..."
python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" 2>&1 || die "HF login failed"
log "  HuggingFace login successful"

# ---------------------------------------------------------------------------
# Step 4: Download all datasets (parallel)
# ---------------------------------------------------------------------------
log "Step 4/7: Downloading all datasets in parallel"
DL_START=$(date +%s)
cd "${CODE_DIR}"

FAIL=0

# 4a: CulturaX Parquet (28 GB, needs HF auth)
log "  [CulturaX] Starting Parquet download (~28 GB)..."
python -u -c "
from huggingface_hub import snapshot_download
snapshot_download('uonlp/CulturaX', repo_type='dataset',
    allow_patterns=['data/bn/*'], local_dir='${DOWNLOADS_DIR}/culturax')
print('CulturaX download complete')
" > "${PIPELINE_DIR}/dl_culturax.log" 2>&1 &
PID_CX=$!

# 4b: Sangraha Parquet (16 GB, open access)
# Config name may be "ben" or "ben_Beng" -- match both
log "  [Sangraha] Starting Parquet download (~16 GB)..."
python -u -c "
from huggingface_hub import snapshot_download
snapshot_download('ai4bharat/sangraha', repo_type='dataset',
    allow_patterns=['data/ben/*', 'data/ben_Beng/*'], local_dir='${DOWNLOADS_DIR}/sangraha')
print('Sangraha download complete')
" > "${PIPELINE_DIR}/dl_sangraha.log" 2>&1 &
PID_SG=$!

# 4c: CC-100 raw text (860 MB compressed)
log "  [CC-100] Starting download (~860 MB)..."
wget -q --show-progress -O "${DOWNLOADS_DIR}/bn.txt.xz" \
    "https://data.statmt.org/cc-100/bn.txt.xz" \
    > "${PIPELINE_DIR}/dl_cc100.log" 2>&1 &
PID_CC=$!

# 4d: Wikipedia dumps via collector (downloads + extracts + converts to JSONL)
log "  [Wikipedia] Starting dump download + extraction..."
python -u -c "
import sys
sys.path.insert(0, '.')
from collectors.wikipedia import WikipediaCollector
# Patch wikiextractor to use more processes on this VM
import collectors.wikipedia as wm
orig = wm.WikipediaCollector._run_wikiextractor
def patched_run(self, dump_path):
    output_dir = dump_path.parent / 'extracted'
    output_dir.mkdir(exist_ok=True)
    import subprocess
    cmd = [
        'python', '-m', 'wikiextractor.WikiExtractor',
        str(dump_path), '--output', str(output_dir),
        '--json', '--no-templates',
        '--processes', '${WIKI_EXTRACTOR_PROCS}',
        '--min_text_length', '200',
    ]
    print(f'[wikipedia] Running: {\" \".join(cmd)}')
    subprocess.run(cmd, check=True)
    return output_dir
wm.WikipediaCollector._run_wikiextractor = patched_run
collector = WikipediaCollector()
collector.run()
" > "${PIPELINE_DIR}/dl_wiki.log" 2>&1 &
PID_WK=$!

# Wait for all downloads
log "  Waiting for all downloads to complete..."

wait $PID_CX || { log "  [CulturaX] FAILED -- see dl_culturax.log"; FAIL=1; }
log "  [CulturaX] download complete"

wait $PID_SG || { log "  [Sangraha] FAILED -- see dl_sangraha.log"; FAIL=1; }
log "  [Sangraha] download complete"

wait $PID_CC || { log "  [CC-100] FAILED -- see dl_cc100.log"; FAIL=1; }
log "  [CC-100] download complete"

wait $PID_WK || { log "  [Wikipedia] FAILED -- see dl_wiki.log"; FAIL=1; }
log "  [Wikipedia] download complete"

[[ $FAIL -eq 1 ]] && die "One or more downloads failed. Check logs in ${PIPELINE_DIR}/dl_*.log"

log "  All downloads finished in $(elapsed_since $DL_START)"
log "  Download sizes:"
du -sh "${DOWNLOADS_DIR}"/*/ 2>/dev/null || true
du -sh "${DOWNLOADS_DIR}/bn.txt.xz" 2>/dev/null || true
log "  Disk usage: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')"

# ---------------------------------------------------------------------------
# Step 5: Convert downloads to pipeline JSONL format
# ---------------------------------------------------------------------------
log "Step 5/7: Converting datasets to JSONL"
CONV_START=$(date +%s)

FAST_INPUT="${PIPELINE_DIR}/fast_input"
mkdir -p "${FAST_INPUT}"

# 5a: CulturaX Parquet -> JSONL
log "  [CulturaX] Parquet -> JSONL..."
python -u convert_to_jsonl.py parquet \
    "${DOWNLOADS_DIR}/culturax" \
    "${FAST_INPUT}/culturax.jsonl" \
    culturax

# Free disk: remove CulturaX Parquet files
rm -rf "${DOWNLOADS_DIR}/culturax"
log "  [CulturaX] Parquet files removed, $(du -sh "${FAST_INPUT}/culturax.jsonl" | cut -f1) JSONL"

# 5b: Sangraha Parquet -> JSONL
log "  [Sangraha] Parquet -> JSONL..."
python -u convert_to_jsonl.py parquet \
    "${DOWNLOADS_DIR}/sangraha" \
    "${FAST_INPUT}/sangraha.jsonl" \
    sangraha

rm -rf "${DOWNLOADS_DIR}/sangraha"
log "  [Sangraha] Parquet files removed, $(du -sh "${FAST_INPUT}/sangraha.jsonl" | cut -f1) JSONL"

# 5c: CC-100 .txt.xz -> JSONL (into raw dir for pipeline_parallel)
log "  [CC-100] .txt.xz -> JSONL..."
mkdir -p "${CODE_DIR}/data/raw/hf_corpus"
python -u convert_to_jsonl.py cc100 \
    "${DOWNLOADS_DIR}/bn.txt.xz" \
    "${CODE_DIR}/data/raw/hf_corpus/cc100.jsonl"

rm -f "${DOWNLOADS_DIR}/bn.txt.xz"
log "  [CC-100] Source removed, $(du -sh "${CODE_DIR}/data/raw/hf_corpus/cc100.jsonl" | cut -f1) JSONL"

# Wikipedia is already in JSONL format (collector handled it in step 4d)
log "  [Wikipedia] Already in JSONL from collector"
find "${CODE_DIR}/data/raw/wikipedia" "${CODE_DIR}/data/raw/wikisource" \
    -name "*.jsonl" -exec sh -c 'echo "    {} ($(wc -l < {} | tr -d " ") docs, $(du -h {} | cut -f1))"' \; 2>/dev/null

log "  Conversion finished in $(elapsed_since $CONV_START)"
log "  Disk usage: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')"

# ---------------------------------------------------------------------------
# Step 6: Process all datasets
# ---------------------------------------------------------------------------
log "Step 6/7: Processing all datasets"
PROC_START=$(date +%s)
mkdir -p "${OUTPUT_DIR}/reports"

# 6a: CulturaX -> pipeline_fast (normalize + SHA-256 dedup)
log "  [CulturaX] pipeline_fast with ${FAST_WORKERS} workers..."
python -u pipeline_fast.py \
    --input "${FAST_INPUT}/culturax.jsonl" \
    --output-dir "${OUTPUT_DIR}" \
    --workers "${FAST_WORKERS}"

# Save report before it gets overwritten
cp "${OUTPUT_DIR}/reports/quality_report_fast.json" \
   "${OUTPUT_DIR}/reports/quality_report_culturax.json" 2>/dev/null || true

# Free disk
rm -f "${FAST_INPUT}/culturax.jsonl"
log "  [CulturaX] Done. Output: $(du -sh "${OUTPUT_DIR}/culturax.jsonl" 2>/dev/null | cut -f1 || echo 'N/A')"

# 6b: Sangraha -> pipeline_fast
log "  [Sangraha] pipeline_fast with ${FAST_WORKERS} workers..."
python -u pipeline_fast.py \
    --input "${FAST_INPUT}/sangraha.jsonl" \
    --output-dir "${OUTPUT_DIR}" \
    --workers "${FAST_WORKERS}"

cp "${OUTPUT_DIR}/reports/quality_report_fast.json" \
   "${OUTPUT_DIR}/reports/quality_report_sangraha.json" 2>/dev/null || true

rm -f "${FAST_INPUT}/sangraha.jsonl"
log "  [Sangraha] Done. Output: $(du -sh "${OUTPUT_DIR}/sangraha.jsonl" 2>/dev/null | cut -f1 || echo 'N/A')"

# 6c: CC-100 + Wikipedia -> pipeline_parallel (full: normalize + fasttext + quality + dedup)
# NOTE: fasttext lid.176.bin auto-downloads on first use (~126 MB)
log "  [CC-100 + Wikipedia] pipeline_parallel with ${FULL_WORKERS} workers..."
python -u pipeline_parallel.py "${FULL_WORKERS}"

# Move pipeline_parallel output to OUTPUT_DIR
for f in "${CODE_DIR}/data/processed/"*.jsonl; do
    [[ -f "$f" ]] && mv "$f" "${OUTPUT_DIR}/"
done
cp "${CODE_DIR}/data/reports/quality_report.json" \
   "${OUTPUT_DIR}/reports/quality_report_full_pipeline.json" 2>/dev/null || true

log "  Processing finished in $(elapsed_since $PROC_START)"

# ---------------------------------------------------------------------------
# Step 7: Summary
# ---------------------------------------------------------------------------
log ""
log "========================================="
log "  ALL DONE in $(elapsed_since $SCRIPT_START)"
log "========================================="
log ""
log "Output files:"
ls -lhS "${OUTPUT_DIR}"/*.jsonl 2>/dev/null || echo "  (no output files)"
log ""
log "Reports:"
ls -lh "${OUTPUT_DIR}/reports/"*.json 2>/dev/null || echo "  (no reports)"

log ""
log "Total output size: $(du -sh "${OUTPUT_DIR}" | cut -f1)"
log "Disk usage: $(df -h / | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')"

# Print per-source stats if reports exist
for report in "${OUTPUT_DIR}/reports/"*.json; do
    [[ -f "$report" ]] || continue
    log ""
    log "Report: $(basename "$report")"
    python -c "
import json
with open('${report}') as f:
    r = json.load(f)
g = r.get('global', r)
print(f\"  Input:     {g.get('total_input_docs', g.get('total', 0)):>12,} docs\")
print(f\"  Output:    {g.get('total_output_docs', g.get('kept', 0)):>12,} docs\")
rate = g.get('retention_rate', 0)
if isinstance(rate, float) and rate <= 1:
    rate = f'{rate:.1%}'
print(f\"  Retention: {rate:>11}\")
" 2>/dev/null || true
done

log ""
log "=== DOWNLOAD INSTRUCTIONS ==="
log "To copy results to your local machine:"
log "  gcloud compute scp --recurse bangla-pipeline:${OUTPUT_DIR}/ ./cloud-output/"
log ""
log "To stop the VM (stop billing):"
log "  gcloud compute instances stop bangla-pipeline --zone=us-central1-a"
log ""
log "NOTE: CulturaX and CC-100 both originate from CommonCrawl."
log "Consider running a cross-source dedup pass on the combined output."
