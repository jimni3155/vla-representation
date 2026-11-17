import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, SiglipModel


ROOT = Path.home() / "vla-representation/vla-representation"
SOURCE = ROOT / "results/probe/main_20"
PI05 = ROOT / "results/probe/features/pi05_main_20.npz"
OUTPUT = ROOT / "results/probe/features"

MODELS = {
    "dinov2": "facebook/dinov2-base",
    "siglip": "google/siglip-base-patch16-224",
}


def letterbox(image, size=224):
    width, height = image.size
    scale = min(size / width, size / height)

    new_width = round(width * scale)
    new_height = round(height * scale)

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC,
    )

    canvas = Image.new("RGB", (size, size))
    x = (size - new_width) // 2
    y = (size - new_height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def load_data(limit):
    with np.load(PI05) as data:
        sample_ids = data["sample_ids"].copy()
        states = data["states"].copy()
        targets = data["targets"].copy()
        episodes = data["episodes"].copy()

    if limit:
        sample_ids = sample_ids[:limit]
        states = states[:limit]
        targets = targets[:limit]
        episodes = episodes[:limit]

    images = []

    for sample_id in sample_ids:
        with np.load(SOURCE / str(sample_id)) as sample:
            image = Image.fromarray(sample["image"])
            images.append(letterbox(image))

    return images, sample_ids, states, targets, episodes


@torch.inference_mode()
def extract_batch(name, model, pixels):
    if name == "siglip":
        output = model.vision_model(pixel_values=pixels)
        patches = output.last_hidden_state
        global_feature = patches.mean(dim=1)
    else:
        output = model(pixel_values=pixels)
        tokens = output.last_hidden_state
        global_feature = tokens[:, 0]
        patches = tokens[:, 1:]

    grid = int(patches.shape[1] ** 0.5)
    if grid * grid != patches.shape[1]:
        raise ValueError(f"Unexpected patch count: {patches.shape[1]}")

    spatial = patches.transpose(1, 2).reshape(
        len(pixels), -1, grid, grid
    )
    spatial = F.adaptive_avg_pool2d(spatial, (2, 2))
    spatial = spatial.flatten(1)

    return (
        global_feature.float().cpu().numpy(),
        spatial.float().cpu().numpy(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    images, sample_ids, states, targets, episodes = load_data(args.limit)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_id = MODELS[args.model]
    print(f"Model: {model_id}", flush=True)
    print(f"Samples: {len(images)}, device: {device}", flush=True)

    processor = AutoImageProcessor.from_pretrained(model_id)
    if args.model == "siglip":
        model = SiglipModel.from_pretrained(model_id)
    else:
        model = AutoModel.from_pretrained(model_id)

    model = model.to(device).eval()
    global_features = []
    spatial_features = []

    for start in range(0, len(images), args.batch_size):
        batch = images[start:start + args.batch_size]

        inputs = processor(
            images=batch,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixels = inputs["pixel_values"].to(device)

        global_vec, spatial_vec = extract_batch(
            args.model, model, pixels
        )
        global_features.append(global_vec)
        spatial_features.append(spatial_vec)

        done = min(start + args.batch_size, len(images))
        if done % 40 == 0 or done == len(images):
            print(f"Processed {done}/{len(images)}", flush=True)

    global_features = np.concatenate(global_features)
    spatial_features = np.concatenate(spatial_features)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    suffix = f"_test{args.limit}" if args.limit else ""
    path = OUTPUT / f"{args.model}_main_20{suffix}.npz"

    np.savez_compressed(
        path,
        global_features=global_features,
        spatial_features=spatial_features,
        states=states,
        targets=targets,
        episodes=episodes,
        sample_ids=sample_ids,
        model_name=model_id,
        image_transform="letterbox_224",
    )

    print("Saved:", path)
    print("Global:", global_features.shape)
    print("Spatial:", spatial_features.shape)


if __name__ == "__main__":
    main()
