#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:?Usage: run_baseline.sh <main|vision|execution> <num_envs>}"
NUM_ENVS="${2:?Missing number of environments}"

PROJECT_ROOT="$HOME/vla-representation/vla-representation"
RLINF_ROOT="$HOME/vla-representation/RLinf"
MODEL="$PROJECT_ROOT/checkpoints/pi05_maniskill_sft"

case "$ENV_NAME" in
  main)
    ENV_ID="PutOnPlateInScene25Main-v3"
    OBJ_SET="train"
    ;;
  vision)
    ENV_ID="PutOnPlateInScene25VisionTexture05-v1"
    OBJ_SET="test"
    ;;
  execution)
    ENV_ID="PutOnPlateInScene25EEPose-v1"
    OBJ_SET="test"
    ;;
  *)
    echo "Unknown environment: $ENV_NAME" >&2
    exit 1
    ;;
esac

OUTPUT="${EVAL_OUTPUT_DIR:-$PROJECT_ROOT/results/baseline/${ENV_NAME}_${NUM_ENVS}}"
mkdir -p "$OUTPUT/videos"

export PATH="$RLINF_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$RLINF_ROOT:${PYTHONPATH:-}"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"

cd "$RLINF_ROOT"

exec "$RLINF_ROOT/.venv/bin/python" \
  evaluations/eval_embodied_agent.py \
  --config-path "$EMBODIED_PATH/config" \
  --config-name maniskill_ppo_openpi_pi05 \
  runner.only_eval=true \
  runner.logger.log_path="$OUTPUT" \
  actor.model.model_path="$MODEL" \
  'rollout.model=${actor.model}' \
  env.eval.total_num_envs="$NUM_ENVS" \
  env.eval.rollout_epoch="${EVAL_ROLLOUT_EPOCHS:-1}" \
  env.eval.max_episode_steps=80 \
  env.eval.max_steps_per_rollout_epoch=80 \
  env.eval.use_fixed_reset_state_ids="${EVAL_FIXED_RESET_IDS:-true}" \
  env.eval.init_params.id="$ENV_ID" \
  env.eval.init_params.obj_set="$OBJ_SET" \
  env.eval.video_cfg.save_video="${EVAL_SAVE_VIDEO:-true}" \
  env.eval.video_cfg.video_base_dir="$OUTPUT/videos"
