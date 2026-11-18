import builtins
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

MODULE = "rlinf.workers.env.env_worker"
original_import = builtins.__import__


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_gt_state(task):
    """[B,14]: object-gripper, plate-object, qpos."""
    qpos = as_numpy(task.agent.robot.get_qpos()).astype(np.float32)

    gripper = as_numpy(
        (task.agent.finger1_link.pose.p
         + task.agent.finger2_link.pose.p) / 2
    )

    source_ids = as_numpy(task.select_carrot_ids).reshape(-1)
    target_ids = as_numpy(task.select_plate_ids).reshape(-1)

    rows = []
    for i in range(len(qpos)):
        obj_name = task.carrot_names[int(source_ids[i])]
        plate_name = task.plate_names[int(target_ids[i])]

        obj = as_numpy(task.objs_carrot[obj_name].pose.p)[i]
        plate = as_numpy(task.objs_plate[plate_name].pose.p)[i]

        rows.append(np.concatenate([
            obj - gripper[i],
            plate - obj,
            qpos[i],
        ]))

    states = np.asarray(rows, dtype=np.float32)
    if states.shape[1] != 14:
        raise RuntimeError(f"Unexpected GT state shape: {states.shape}")
    if not np.isfinite(states).all():
        raise RuntimeError("Non-finite GT state")
    return states


def install():
    module = sys.modules.get(MODULE)
    if module is None:
        return

    cls = getattr(module, "EnvWorker", None)
    if cls is None or getattr(cls, "_online_td3_installed", False):
        return

    original = cls.env_evaluate_step

    def wrapped(self, raw_actions, stage_id):
        from td3 import TD3, ReplayBuffer

        env = self.eval_env_list[stage_id]
        task = env
        for _ in range(10):
            if (
                hasattr(task, "agent")
                and hasattr(task, "select_carrot_ids")
            ):
                break

            inner = getattr(task, "env", None)
            if inner is None:
                raise RuntimeError(
                    f"Cannot locate ManiSkill task: {type(task)}"
                )

            next_task = getattr(inner, "unwrapped", inner)
            if next_task is task:
                raise RuntimeError(
                    f"Environment unwrapping stalled: {type(task)}"
                )
            task = next_task
        else:
            raise RuntimeError("ManiSkill task not found")

        if not hasattr(self, "_online_td3"):
            torch.set_num_threads(2)

            self._online_td3 = TD3(
                state_dim=14,
                action_dim=3,
                max_action=0.005,
                chunk_size=5,
                device="cpu",
            )
            self._online_replay = ReplayBuffer(
                state_dim=14,
                action_dim=3,
                capacity=100_000,
            )
            self._online_count = 0

            self._online_dir = Path(
                os.environ["ONLINE_TD3_DIR"]
            )
            self._online_dir.mkdir(parents=True, exist_ok=True)

            self._online_log = self._online_dir / (
                f"train_{os.getpid()}.csv"
            )

            with self._online_log.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "chunk", "transitions",
                    "mean_reward", "mean_abs_residual",
                    "critic_loss", "actor_loss",
                ])

            print("[TD3] Initialized online learner", flush=True)

        agent = self._online_td3
        replay = self._online_replay
        count = self._online_count

        original_chunk_step = env.chunk_step

        def learning_chunk_step(actions):
            state = get_gt_state(task)
            batch_size = len(state)

            # Initial exploration; then learned policy + noise.
            if replay.size < 64:
                residual = np.random.uniform(
                    -0.005, 0.005, size=(batch_size, 3)
                ).astype(np.float32)
            else:
                residual = agent.select_action(
                    state, noise_std=0.001
                )

            modified = actions.clone()
            delta = torch.as_tensor(
                residual,
                device=modified.device,
                dtype=modified.dtype,
            )

            # One residual decision per 5-step action chunk.
            modified[:, 0, :3] += delta

            result = original_chunk_step(modified)

            (
                obs_list,
                chunk_rewards,
                terminations,
                truncations,
                infos_list,
            ) = result

            next_state = get_gt_state(task)

            rewards = as_numpy(chunk_rewards)
            terminated = as_numpy(
                terminations.any(dim=1)
            ).astype(bool)
            truncated = as_numpy(
                truncations.any(dim=1)
            ).astype(bool)

            done = np.logical_or(terminated, truncated)

            gamma = 0.99
            weights = gamma ** np.arange(
                rewards.shape[1], dtype=np.float32
            )
            chunk_return = (rewards * weights[None, :]).sum(axis=1)

            # Done transitions do not bootstrap, so auto-reset
            # next-state rows are harmless for this pilot.
            replay.add(
                state,
                residual,
                chunk_return.astype(np.float32),
                next_state,
                done.astype(np.float32),
            )

            metrics = None
            if replay.size >= 64:
                # Approximately one gradient update per transition.
                for _ in range(batch_size):
                    metrics = agent.train_step(
                        replay, batch_size=64
                    )

            self._online_count += 1
            current = self._online_count

            critic = (
                metrics["critic_loss"] if metrics else None
            )
            actor = (
                metrics["actor_loss"] if metrics else None
            )

            with self._online_log.open("a", newline="") as f:
                csv.writer(f).writerow([
                    current,
                    replay.size,
                    float(np.mean(chunk_return)),
                    float(np.mean(np.abs(residual))),
                    critic,
                    actor,
                ])

            if current == 1 or current % 4 == 0:
                agent.save(
                    self._online_dir / "latest_td3.pt"
                )
                print(
                    f"[TD3] chunk={current} "
                    f"replay={replay.size} "
                    f"reward={chunk_return.mean():.5f} "
                    f"critic={critic} actor={actor}",
                    flush=True,
                )

            return result

        env.chunk_step = learning_chunk_step
        try:
            return original(self, raw_actions, stage_id)
        finally:
            env.chunk_step = original_chunk_step

    cls.env_evaluate_step = wrapped
    cls._online_td3_installed = True
    print("[TD3] Online hook installed", flush=True)


def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
    result = original_import(name, globals, locals, fromlist, level)
    if MODULE in sys.modules:
        install()
    return result


if os.environ.get("ONLINE_TD3_DIR"):
    builtins.__import__ = hooked_import
