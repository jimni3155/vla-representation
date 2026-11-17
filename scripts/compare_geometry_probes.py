from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path("results/probe/features")
ALPHA = 100.0


def evaluate(features, targets, episodes):
    splitter = GroupKFold(n_splits=5)
    errors = []

    for train_idx, test_idx in splitter.split(features, targets, episodes):
        scaler = StandardScaler()
        train = scaler.fit_transform(features[train_idx])
        test = scaler.transform(features[test_idx])

        model = Ridge(alpha=ALPHA, solver="lsqr")
        model.fit(train, targets[train_idx])

        predictions = model.predict(test)
        distances = np.linalg.norm(
            predictions - targets[test_idx], axis=1
        )

        errors.append(distances.mean() * 100)

    return np.array(errors)


def main():
    with np.load(ROOT / "pi05_main_20.npz") as data:
        latent = data["features"].astype(np.float32)
        states = data["states"].astype(np.float32)
        targets = data["targets"].astype(np.float32)
        episodes = data["episodes"].copy()
        sample_ids = data["sample_ids"].copy()

    models = {
        "Robot State": states,
        "Pi05 Latent": latent,
        "Pi05 Latent + State": np.concatenate(
            [latent, states], axis=1
        ),
    }

    for name in ("dinov2", "siglip", "dinov3"):
        path = ROOT / f"{name}_main_20.npz"
        if not path.exists():
            print(f"Skipping {name}: file not found")
            continue

        with np.load(path) as data:
            if not np.array_equal(data["sample_ids"], sample_ids):
                raise ValueError(f"{name}: sample order mismatch")

            if not np.allclose(data["targets"], targets):
                raise ValueError(f"{name}: target mismatch")

            global_features = data["global_features"].astype(np.float32)
            spatial_features = data["spatial_features"].astype(np.float32)

        models[f"{name} Global + State"] = np.concatenate(
            [global_features, states], axis=1
        )
        models[f"{name} Spatial + State"] = np.concatenate(
            [spatial_features, states], axis=1
        )

    print(f"\nSamples: {len(targets)}")
    print(f"Episodes: {len(set(episodes))}")
    print("Metric: Mean 3D position error (cm)")
    print(f"Ridge alpha: {ALPHA}\n")

    results = []

    for name, features in models.items():
        errors = evaluate(features, targets, episodes)

        results.append((name, errors.mean(), errors.std(ddof=1)))

        print(
            f"{name:<28} "
            f"{errors.mean():5.2f} ± {errors.std(ddof=1):4.2f} cm"
        )
        print("  Folds:", np.round(errors, 2).tolist())

    print("\n===== Ranking =====")
    for rank, (name, mean, std) in enumerate(
        sorted(results, key=lambda x: x[1]), start=1
    ):
        print(f"{rank}. {name}: {mean:.2f} cm")


if __name__ == "__main__":
    main()
