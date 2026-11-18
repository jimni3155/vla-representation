import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path.home() / "vla-representation/vla-representation"
RUN = ROOT / "results/residual/td3_v2/train_50"
CKPT = RUN / "latest_td3_v2.pt"

sys.path.insert(0, str(ROOT / "scripts/residual_rl"))
from td3 import TD3


def summary(name, x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        print(f"{name}: no data")
        return

    print(
        f"{name}: "
        f"mean={x.mean():.6f}, "
        f"min={x.min():.6f}, "
        f"max={x.max():.6f}"
    )


print("===== TRAINING LOG =====")

files = sorted(RUN.glob("worker_*.csv"))
assert len(files) == 1, f"Expected one worker CSV, found {len(files)}"

with files[0].open() as f:
    rows = list(csv.DictReader(f))

print("Logged chunks:", len(rows))
print("Last replay size:", rows[-1]["replay_size"])

for name in (
    "mean_reward",
    "mean_abs_residual",
    "critic_loss",
    "actor_loss",
):
    values = [
        float(row[name])
        for row in rows
        if row[name] not in ("", "None")
    ]

    print(f"\n{name}")
    summary("  first quarter", values[:max(1, len(values)//4)])
    summary("  last quarter", values[-max(1, len(values)//4):])


print("\n===== CHECKPOINT =====")

agent = TD3(
    state_dim=21,
    action_dim=3,
    max_action=0.002,
    chunk_size=5,
    device="cpu",
)

agent.load(CKPT)
agent.actor.eval()

print("TD3 updates:", agent.total_it)

print("\n===== ACTOR WEIGHTS =====")

for name, param in agent.actor.named_parameters():
    values = param.detach().cpu().numpy()
    print(
        f"{name}: "
        f"mean={values.mean():.6f}, "
        f"std={values.std():.6f}, "
        f"norm={np.linalg.norm(values):.6f}"
    )

print("\n===== FINAL OUTPUT LAYER =====")

last = agent.actor.net[-1]
print("Bias:", last.bias.detach().cpu().numpy())
print("Weight norm:", last.weight.detach().norm().item())

print("\n===== CHECK COMPLETE =====")
print("Note: State dependence cannot be established from weights alone.")
