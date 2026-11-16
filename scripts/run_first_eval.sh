#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$HOME/vla-representation/vla-representation"
RLINF_ROOT="$HOME/vla-representation/RLinf"

MODEL="$PROJECT_ROOT/checkpoints/pi05_maniskill_sft"
OUTPUT="$PROJECT_ROOT/results/baseline/first_eval"

export PATH="$RLINF_ROOT/.venv/bin:$PATH"
export PYTHONPATH="$RLINF_ROOT:${PYTHONPATH:-}"
export EMBODIED_PATH="$RLINF_ROOT/examples/embodiment"
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false

# Use the NVIDIA Vulkan driver.
export VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json
export VK_ICD_FILENAMES="$VK_DRIVER_FILES"

mkdir -p "$OUTPUT/videos"

cd "$RLINF_ROOT"

exec "$RLINF_ROOT/.venv/bin/python" \
  evaluations/eval_embodied_agent.py \
  --config-path "$RLINF_ROOT/examples/embodiment/config" \
  --config-name maniskill_ppo_openpi_pi05 \
  runner.only_eval=true \
  runner.logger.log_path="$OUTPUT" \
  actor.model.model_path="$MODEL" \
  'rollout.model=${actor.model}' \
  env.eval.total_num_envs=1 \
  env.eval.rollout_epoch=1 \
  env.eval.max_episode_steps=80 \
  env.eval.max_steps_per_rollout_epoch=80 \
  env.eval.use_fixed_reset_state_ids=true \
  env.eval.init_params.id=PutOnPlateInScene25Main-v3 \
  env.eval.init_params.obj_set=train \
  env.eval.video_cfg.save_video=true \
  env.eval.video_cfg.video_base_dir="$OUTPUT/videos"
