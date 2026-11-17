from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


ROOT = Path("results/probe/features")
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]


def load_data(env):
    with np.load(ROOT / f"pi05_{env}_20.npz") as data:
        return {
            "latent": data["features"].astype(np.float32),
            "state": data["states"].astype(np.float32),
            "target": data["targets"].astype(np.float32),
            "episode": data["episodes"].copy(),
        }


def get_features(data, name):
    if name == "State":
        return data["state"]
    if name == "Pi05":
        return data["latent"]
    if name == "Pi05 + State":
        return np.concatenate(
            [data["latent"], data["state"]], axis=1
        )
    raise ValueError(name)


def position_errors(predictions, targets):
    return np.linalg.norm(predictions - targets, axis=1) * 100


def select_alpha(x, y, groups):
    splitter = GroupKFold(n_splits=5)
    best_alpha = None
    best_error = float("inf")

    for alpha in ALPHAS:
        errors = []

        for train_idx, val_idx in splitter.split(x, y, groups):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x[train_idx])
            x_val = scaler.transform(x[val_idx])

            model = Ridge(alpha=alpha, solver="lsqr")
            model.fit(x_train, y[train_idx])

            predictions = model.predict(x_val)
            errors.extend(
                position_errors(predictions, y[val_idx])
            )

        mean_error = np.mean(errors)
        if mean_error < best_error:
            best_error = mean_error
            best_alpha = alpha

    return best_alpha, best_error


def main():
    datasets = {
        env: load_data(env)
        for env in ("main", "vision", "execution")
    }

    train_data = datasets["main"]
    y_train = train_data["target"]
    groups = train_data["episode"]

    print("Training: Main (320 observations, 20 episodes)")
    print("Evaluation: Vision / Execution (no refitting)")
    print("Metric: mean 3D position error in cm\n")

    for env, data in datasets.items():
        if len(data["target"]) != 320:
            raise ValueError(f"{env}: unexpected sample count")
        if not np.isfinite(data["target"]).all():
            raise ValueError(f"{env}: invalid targets")

        distances = np.linalg.norm(data["target"], axis=1) * 100
        print(
            f"{env:>9} | GT distance mean={distances.mean():.2f} cm"
            f", std={distances.std():.2f} cm"
        )

    print()

    for name in ("State", "Pi05", "Pi05 + State"):
        x_train = get_features(train_data, name)

        alpha, cv_error = select_alpha(
            x_train, y_train, groups
        )

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_train)

        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(x_scaled, y_train)

        print(f"===== {name} =====")
        print(f"Alpha: {alpha}")
        print(f"Main CV: {cv_error:.2f} cm")

        for env in ("vision", "execution"):
            data = datasets[env]
            x_test = get_features(data, name)

            predictions = model.predict(
                scaler.transform(x_test)
            )
            errors = position_errors(
                predictions, data["target"]
            )

            print(
                f"{env.capitalize():>9} OOD: "
                f"{errors.mean():.2f} cm "
                f"(median={np.median(errors):.2f}, "
                f"p90={np.percentile(errors, 90):.2f})"
            )

        print()


if __name__ == "__main__":
    main()
