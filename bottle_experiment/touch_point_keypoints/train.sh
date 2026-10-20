#!/bin/bash
# Train the touch-point keypoint detector on one split of datasets/touch_point_keypoints.
#   bash bottle_experiment/touch_point_keypoints/train.sh fold0     # a cross-validation fold
#   bash bottle_experiment/touch_point_keypoints/train.sh final     # the model the rig uses
# Logs to wandb offline; the best checkpoint is copied to datasets/touch_point_keypoints/models/<split>.ckpt.
set -euo pipefail
SPLIT=${1:?usage: train.sh <fold0..fold4|final>}
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA="$REPO/datasets/touch_point_keypoints"
export WANDB_MODE=${WANDB_MODE:-offline}
cd "$REPO/keypoint-detection"
STAMP=$(mktemp)
~/miniconda3/envs/keypoint-detection/bin/python keypoint_detection/tasks/train.py \
  --keypoint_channel_configuration "touch_point" \
  --json_dataset_path "$DATA/${SPLIT}_train.json" \
  --json_validation_dataset_path "$DATA/${SPLIT}_val.json" \
  --backbone_type ConvNeXtUnet \
  --heatmap_sigma 3 \
  --max_keypoints 1 \
  --maximal_gt_keypoint_pixel_distances "2 4 8" \
  --batch_size 8 --learning_rate 3e-4 \
  --max_epochs 80 --early_stopping_relative_threshold -1.0 \
  --ap_epoch_start 1 --ap_epoch_freq 1 \
  --augment_train \
  --accelerator gpu --devices 1 --precision 16 \
  --wandb_project touch-point-keypoints --wandb_name "$SPLIT"
BEST=$(find logging -name "*.ckpt" -newer "$STAMP" -printf "%T@ %p\n" | sort -n | tail -1 | cut -d" " -f2-)
rm -f "$STAMP"
mkdir -p "$DATA/models"
cp "$BEST" "$DATA/models/$SPLIT.ckpt"
echo "best checkpoint: $BEST -> $DATA/models/$SPLIT.ckpt"
