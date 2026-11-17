import argparse
import os
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.modules.model import preprocess_observation


ROOT = Path.home() / "vla-representation/vla-representation"
RLINF = ROOT.parent / "RLinf"
SOURCE = ROOT / "results/probe/main_20"
OUTPUT = ROOT / "results/probe/features/pi05_main_20.npz"
CHECKPOINT = ROOT / "checkpoints/pi05_maniskill_sft"


def load_model():
    config_dir = RLINF / "examples/embodiment/config"
    os.environ["EMBODIED_PATH"] = str(RLINF / "examples/embodiment")

    with initialize_config_dir(
        config_dir=str(config_dir),
        version_base=None,
    ):
        cfg = compose(config_name="maniskill_ppo_openpi_pi05")

    model_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.actor.model, resolve=True)
    )
    model_cfg.model_path = str(CHECKPOINT)
    model_cfg.openpi.task = "eval"
    model_cfg.openpi.torch_compile = False

    model = get_model(model_cfg)
    model = model.to("cuda").eval()

    print("Loaded:", CHECKPOINT)
    print("Model:", type(model).__name__)
    return model


def load_paths(limit):
    paths = sorted(SOURCE.rglob("*.npz"))
    paths = [
        path for path in paths
        if int(path.stem.rsplit("_step_", 1)[1]) < 80
    ]

    if limit:
        paths = paths[:limit]

    if not paths:
        raise RuntimeError(f"No observations found in {SOURCE}")

    return paths


@torch.inference_mode()
def extract_feature(model, sample):
    env_obs = {
        "main_images": torch.from_numpy(sample["image"][None]).to("cuda"),
        "states": torch.from_numpy(sample["state"][None]).to("cuda"),
        "task_descriptions": [sample["instruction"].item()],
    }

    observation = model.env_obs_to_observation(env_obs)
    prepared = preprocess_observation(observation, train=False)

    prefix_output, prefix_mask, _ = model.build_prefix_cache(prepared)

    mask = prefix_mask.unsqueeze(-1).to(prefix_output.dtype)
    feature = (prefix_output * mask).sum(dim=1)
    feature = feature / mask.sum(dim=1).clamp(min=1)

    return feature[0].float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    paths = load_paths(args.limit)
    print(f"Observations: {len(paths)}")

    model = load_model()

    features = []
    states = []
    targets = []
    episodes = []
    sample_ids = []

    for index, path in enumerate(paths, start=1):
        with np.load(path) as sample:
            feature = extract_feature(model, sample)
            features.append(feature)
            states.append(sample["state"].copy())
            targets.append(sample["gripper_to_object"].copy())

        episodes.append(
            f"{path.parent.name}/{path.stem.split('_step_')[0]}"
        )
        sample_ids.append(str(path.relative_to(SOURCE)))

        if index == 1 or index % 20 == 0 or index == len(paths):
            print(f"Processed {index}/{len(paths)}", flush=True)

    features = np.stack(features)

    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite features detected")

    output = OUTPUT
    if args.limit:
        output = OUTPUT.with_name(f"pi05_test_{args.limit}.npz")

    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        features=features,
        states=np.stack(states),
        targets=np.stack(targets),
        episodes=np.array(episodes),
        sample_ids=np.array(sample_ids),
        representation="vlm_prefix_masked_mean",
        checkpoint=str(CHECKPOINT),
    )

    print("Saved:", output)
    print("Features:", features.shape)
    print("States:", (len(states), states[0].shape[0]))
    print("Targets:", (len(targets), 3))
    print("Episodes:", len(set(episodes)))


if __name__ == "__main__":
    main()
