from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path("results/probe/features")
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]


def fit_and_error(x, y, train_idx, test_idx, alpha):
    scaler = StandardScaler()
    train = scaler.fit_transform(x[train_idx])
    test = scaler.transform(x[test_idx])

    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(train, y[train_idx])

    predictions = model.predict(test)
    return np.linalg.norm(predictions - y[test_idx], axis=1) * 100


def evaluate(x, y, groups):
    outer = GroupKFold(n_splits=5)
    errors = []
    chosen_alphas = []

    for outer_train, outer_test in outer.split(x, y, groups):
        inner_groups = groups[outer_train]
        inner = GroupKFold(n_splits=4)

        alpha_scores = []

        for alpha in ALPHAS:
            fold_errors = []

            for inner_train, inner_val in inner.split(
                x[outer_train], y[outer_train], inner_groups
            ):
                train_idx = outer_train[inner_train]
                val_idx = outer_train[inner_val]

                error = fit_and_error(
                    x, y, train_idx, val_idx, alpha
                )
                fold_errors.extend(error.tolist())

            alpha_scores.append(np.mean(fold_errors))

        best_alpha = ALPHAS[int(np.argmin(alpha_scores))]
        chosen_alphas.append(best_alpha)

        error = fit_and_error(
            x, y, outer_train, outer_test, best_alpha
        )
        errors.extend(error.tolist())

    return np.mean(errors), chosen_alphas


def main():
    with np.load(ROOT / "pi05_main_20.npz") as data:
        latent = data["features"].astype(np.float32)
        states = data["states"].astype(np.float32)
        targets = data["targets"].astype(np.float32)
        groups = data["episodes"].copy()
        sample_ids = data["sample_ids"].copy()

    models = {
        "Robot State": states,
        "Pi05 Latent": latent,
        "Pi05 + State": np.concatenate([latent, states], axis=1),
    }

    for name in ("dinov2", "siglip", "dinov3"):
        path = ROOT / f"{name}_main_20.npz"
        if not path.exists():
            continue

        with np.load(path) as data:
            if not np.array_equal(data["sample_ids"], sample_ids):
                raise ValueError(f"{name}: sample IDs do not match")
            if not np.allclose(data["targets"], targets):
                raise ValueError(f"{name}: targets do not match")

            for field in ("global_features", "spatial_features"):
                features = data[field].astype(np.float32)
                label = "Global" if field == "global_features" else "Spatial"

                models[f"{name} {label} + State"] = np.concatenate(
                    [features, states], axis=1
                )

    print(f"Samples: {len(targets)}")
    print(f"Episodes: {len(set(groups))}")
    print("Nested GroupKFold: outer=5, inner=4\n")

    for name, features in models.items():
        error, alphas = evaluate(features, targets, groups)
        print(f"{name:<28} {error:.2f} cm")
        print(f"  Selected alphas: {alphas}", flush=True)


if __name__ == "__main__":
    main()
