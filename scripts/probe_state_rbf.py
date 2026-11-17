from pathlib import Path

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


source = Path("results/probe/features/pi05_main_20.npz")

with np.load(source) as data:
    x = data["states"].astype(np.float64)
    y = data["targets"].astype(np.float64)
    groups = data["episodes"].copy()

alphas = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1, 10]


def predict_error(train, test, alpha, gamma):
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train])
    x_test = scaler.transform(x[test])

    model = KernelRidge(alpha=alpha, gamma=gamma, kernel="rbf")
    model.fit(x_train, y[train])

    pred = model.predict(x_test)
    return np.linalg.norm(pred - y[test], axis=1) * 100


outer = GroupKFold(n_splits=5)
results = []

for fold, (train, test) in enumerate(outer.split(x, y, groups), 1):
    inner = GroupKFold(n_splits=4)
    best_score = float("inf")
    best_params = None

    for alpha in alphas:
        for gamma in gammas:
            errors = []

            for tr, val in inner.split(x[train], y[train], groups[train]):
                error = predict_error(train[tr], train[val], alpha, gamma)
                errors.extend(error.tolist())

            score = np.mean(errors)
            if score < best_score:
                best_score = score
                best_params = (alpha, gamma)

    alpha, gamma = best_params
    error = predict_error(train, test, alpha, gamma)
    results.extend(error.tolist())

    print(
        f"Fold {fold}: {error.mean():.2f} cm "
        f"(alpha={alpha}, gamma={gamma})"
    )

print(f"\nState RBF: {np.mean(results):.2f} cm")
print("Pi05 Latent + State: 2.87 cm (previous result)")
