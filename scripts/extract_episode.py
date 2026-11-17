import argparse
from pathlib import Path

import cv2


def extract_video(source, output, row, col, tile_size=512):
    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    x = col * tile_size
    y = row * tile_size

    if x + tile_size > width or y + tile_size > height:
        raise ValueError("Selected tile is outside the video frame")

    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (tile_size, tile_size),
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        crop = frame[y:y + tile_size, x:x + tile_size]
        writer.write(crop)

    cap.release()
    writer.release()
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", choices=["main", "vision", "execution"], required=True)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--col", type=int, default=0)
    args = parser.parse_args()

    root = Path.home() / "vla-representation/vla-representation"

    source = root / f"results/baseline/{args.env}_50/videos/seed_0/0.mp4"
    output = root / f"results/baseline/previews/{args.env}_r{args.row}_c{args.col}.mp4"

    extract_video(source, output, args.row, args.col)


if __name__ == "__main__":
    main()
