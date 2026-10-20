#!/bin/bash
# Train a list of bottle configs one after another on the single GPU.
#
#   bash robot_imitation_glue/ur5station/bottle/run_training_queue.sh \
#       robot_imitation_glue/ur5station/bottle/configs/bottle_generic_{50,70,90}ep.json
#
# Per config: skipped if its final checkpoint already exists (so re-running the queue after a crash
# resumes with the first unfinished run); otherwise `lerobot-train --config_path=...`, logged to
# outputs/train/bottle/logs/<config>.log. The queue STOPS at the first failed run (exit != 0) instead of
# starting the next one, so a broken setup cannot burn through every run. A failed run's output_dir must
# be moved aside before re-running (lerobot-train refuses an existing output_dir without resume, and only
# the final checkpoint is saved, so a crashed run restarts from step 0).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO"
LOG_DIR="outputs/train/bottle/logs"
mkdir -p "$LOG_DIR"
TRAIN=/home/rtalwar/miniconda3/envs/imitation_button_py312/bin/lerobot-train

for config in "$@"; do
  name="$(basename "$config" .json)"
  output_dir="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['output_dir'])" "$config")"
  steps="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['steps'])" "$config")"
  final="$output_dir/checkpoints/$(printf '%06d' "$steps")/pretrained_model"
  if [ -d "$final" ]; then
    echo "[queue] $(date '+%F %T') SKIP $name -- final checkpoint exists ($final)"
    continue
  fi
  if [ -e "$output_dir" ]; then
    echo "[queue] $(date '+%F %T') STOP $name -- $output_dir exists without a final checkpoint (crashed run?); move it aside and re-run the queue"
    exit 2
  fi
  echo "[queue] $(date '+%F %T') START $name -> $output_dir (log: $LOG_DIR/$name.log)"
  "$TRAIN" --config_path="$config" > "$LOG_DIR/$name.log" 2>&1
  status=$?
  if [ $status -ne 0 ] || [ ! -d "$final" ]; then
    echo "[queue] $(date '+%F %T') FAILED $name (exit $status, final checkpoint present: $([ -d "$final" ] && echo yes || echo no)) -- stopping the queue"
    exit 1
  fi
  echo "[queue] $(date '+%F %T') DONE $name"
done
echo "[queue] $(date '+%F %T') ALL DONE"
