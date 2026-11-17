from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path.home() / "vla-representation/vla-representation"
SOURCE = ROOT / "results/probe/features/pi05_main_20.npz"
ALPHA = 100.0


def evaluate(features, targets, episodes):
    splitter = GroupKFold(n_splits=5)
    errors = []

    for train_idx, test_idx in splitter.split(features, targets, episodes):
        scaler = StandardScaler()
        x_train = scaler.fit_transform(features[train_idx])
        x_test = scaler.transform(features[test_idx])

        model = Ridge(alpha=ALPHA, solver="lsqr")
        model.fit(x_train, targets[train_idx])

        predictions = model.predict(x_test)
        distances = np.linalg.norm(
            predictions - targets[test_idx], axis=1
        )

        errors.append(distances.mean() * 100)

    return np.array(errors)


def evaluate_mean_baseline(targets, episodes):
    splitter = GroupKFold(n_splits=5)
    errors = []

    for train_idx, test_idx in splitter.split(targets, targets, episodes):
        prediction = targets[train_idx].mean(axis=0)
        distances = np.linalg.norm(
            targets[test_idx] - prediction, axis=1
        )
        errors.append(distances.mean() * 100)

    return np.array(errors)


def main():
    with np.load(SOURCE) as data:
        latent = data["features"].astype(np.float32)
        states = data["states"].astype(np.float32)
        targets = data["targets"].astype(np.float32)
        episodes = data["episodes"].copy()

    assert len(latent) == len(states) == len(targets) == len(episodes)
    assert len(set(episodes)) == 20
    assert np.isfinite(latent).all()
    assert np.isfinite(states).all()
    assert np.isfinite(targets).all()

    models = {
        "Mean baseline": None,
        "Robot State": states,
        "Pi05 Latent": latent,
        "Pi05 Latent + State": np.concatenate(
            [latent, states], axis=1
        ),
    }

    print(f"Samples: {len(targets)}")
    print(f"Episodes: {len(set(episodes))}")
    print(f"Target: Gripper -> Object (3D)")
    print(f"Ridge alpha: {ALPHA}")
    print("\nMean 3D position error (cm)")
    print("-" * 60)

    for name, features in models.items():
        if features is None:
            errors = evaluate_mean_baseline(targets, episodes)
        else:
            errors = evaluate(features, targets, episodes)

        print(
            f"{name:<24} "
            f"{errors.mean():6.2f} ± {errors.std(ddof=1):5.2f}"
        )
        print("  Folds:", np.round(errors, 2).tolist())


if __name__ == "__main__":
    main()
