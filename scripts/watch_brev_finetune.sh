#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  watch_brev_finetune.sh \
    --instance-name kotha-finetune \
    --remote-host 216.81.245.216 \
    --run-pid 15362 \
    --log-file logs/final_finetune_20260402_014303.log \
    --local-dest /abs/path/to/save/artifacts

Behavior:
  - Polls the remote Brev box until the pipeline process exits
  - Downloads the current run log and any produced checkpoints
  - Deletes the Brev instance after a successful run
  - Stops the Brev instance after a failed run
EOF
}

INSTANCE_NAME=""
REMOTE_USER="shadeform"
REMOTE_HOST=""
REMOTE_DIR="/home/shadeform/bangla-llm"
RUN_PID=""
LOG_FILE=""
LOCAL_DEST=""
POLL_SECONDS=60
DELETE_ON_SUCCESS="true"
STOP_ON_FAILURE="true"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --instance-name)
      INSTANCE_NAME="$2"
      shift 2
      ;;
    --remote-user)
      REMOTE_USER="$2"
      shift 2
      ;;
    --remote-host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --run-pid)
      RUN_PID="$2"
      shift 2
      ;;
    --log-file)
      LOG_FILE="$2"
      shift 2
      ;;
    --local-dest)
      LOCAL_DEST="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    --delete-on-success)
      DELETE_ON_SUCCESS="$2"
      shift 2
      ;;
    --stop-on-failure)
      STOP_ON_FAILURE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$INSTANCE_NAME" || -z "$REMOTE_HOST" || -z "$RUN_PID" || -z "$LOG_FILE" || -z "$LOCAL_DEST" ]]; then
  usage >&2
  exit 1
fi

REMOTE="${REMOTE_USER}@${REMOTE_HOST}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

remote_cmd() {
  ssh "$REMOTE" "$@"
}

remote_run_active() {
  remote_cmd "ps -p '$RUN_PID' -o cmd= | grep -q 'run_kotha_pipeline.sh'"
}

remote_pipeline_succeeded() {
  remote_cmd "grep -q 'PIPELINE COMPLETE' '$REMOTE_DIR/$LOG_FILE'"
}

remote_path_exists() {
  local rel_path="$1"
  remote_cmd "test -e '$REMOTE_DIR/$rel_path'"
}

download_artifact() {
  local rel_path="$1"
  local parent_dir
  parent_dir="$(dirname "$rel_path")"
  mkdir -p "$LOCAL_DEST/$parent_dir"
  rsync -az "$REMOTE:$REMOTE_DIR/$rel_path" "$LOCAL_DEST/$parent_dir/"
}

download_if_present() {
  local rel_path="$1"
  if remote_path_exists "$rel_path"; then
    log "Downloading $rel_path"
    download_artifact "$rel_path"
  else
    log "Skipping missing artifact: $rel_path"
  fi
}

cleanup_success() {
  if [[ "$DELETE_ON_SUCCESS" == "true" ]]; then
    log "Deleting Brev instance $INSTANCE_NAME"
    brev delete "$INSTANCE_NAME"
  else
    log "Delete-on-success disabled; leaving instance running"
  fi
}

cleanup_failure() {
  if [[ "$STOP_ON_FAILURE" == "true" ]]; then
    log "Stopping Brev instance $INSTANCE_NAME after unsuccessful run"
    brev stop "$INSTANCE_NAME"
  else
    log "Stop-on-failure disabled; leaving instance running"
  fi
}

main() {
  mkdir -p "$LOCAL_DEST"
  log "Watching instance $INSTANCE_NAME on $REMOTE_HOST"
  log "Run PID: $RUN_PID"
  log "Remote log: $LOG_FILE"
  log "Local destination: $LOCAL_DEST"

  while remote_run_active; do
    log "Training still running; sleeping for ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
  done

  if remote_pipeline_succeeded; then
    status="success"
    log "Training finished successfully"
  else
    status="failure"
    log "Training process exited without success marker"
  fi

  download_if_present "$LOG_FILE"
  download_if_present "checkpoints/kotha-cpt"
  download_if_present "checkpoints/kotha-1-sft"
  download_if_present "checkpoints/kotha-1-news-lora"

  if [[ "$status" == "success" ]]; then
    cleanup_success
  else
    cleanup_failure
  fi

  log "Watcher complete with status: $status"
}

main
