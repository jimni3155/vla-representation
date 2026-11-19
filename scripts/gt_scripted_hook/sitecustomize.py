import builtins
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

MODULE = "rlinf.workers.env.env_worker"
original_import = builtins.__import__

STATE_DIM = 21
MAX_RESIDUAL = 0.002
GAMMA = 0.99
WARMUP = 256
BATCH_SIZE = 128


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def find_task(env):
    task = env

    for _ in range(10):
        if (
            hasattr(task, "agent")
            and hasattr(task, "select_carrot_ids")
        ):
            return task

        inner = getattr(task, "env", None)
        if inner is None:
            raise RuntimeError(
                f"ManiSkill task not found: {type(task)}"
            )

        nxt = getattr(inner, "unwrapped", inner)

        if nxt is task:
            raise RuntimeError("Environment unwrapping stalled")

        task = nxt

    raise RuntimeError("ManiSkill task not found")


def get_state(task, base_actions):
    """
    21D:
      0:3   object - gripper
      3:6   plate - object
      6:14  robot qpos
      14:21 first base action of current VLA chunk
    """
    qpos = as_numpy(
        task.agent.robot.get_qpos()
    ).astype(np.float32)

    gripper = as_numpy(
        (
            task.agent.finger1_link.pose.p
            + task.agent.finger2_link.pose.p
        ) / 2
    )

    source_ids = as_numpy(
        task.select_carrot_ids
    ).reshape(-1)

    target_ids = as_numpy(
        task.select_plate_ids
    ).reshape(-1)

    base = as_numpy(base_actions[:, 0, :]).astype(np.float32)

    rows = []

    for i in range(len(qpos)):
        obj_name = task.carrot_names[int(source_ids[i])]
        plate_name = task.plate_names[int(target_ids[i])]

        obj = as_numpy(
            task.objs_carrot[obj_name].pose.p
        )[i]

        plate = as_numpy(
            task.objs_plate[plate_name].pose.p
        )[i]

        rows.append(np.concatenate([
            obj - gripper[i],
            plate - obj,
            qpos[i],
            base[i],
        ]))

    states = np.asarray(rows, dtype=np.float32)

    if states.shape != (len(qpos), STATE_DIM):
        raise RuntimeError(
            f"Invalid state shape: {states.shape}"
        )

    if not np.isfinite(states).all():
        raise RuntimeError("Non-finite TD3 state")

    return states


def make_agent(mode):
    from td3 import TD3, ReplayBuffer

    agent = TD3(
        state_dim=STATE_DIM,
        action_dim=3,
        max_action=MAX_RESIDUAL,
        chunk_size=5,
        device="cpu",
    )

    replay = None

    if mode == "train":
        # Start from an exactly zero residual policy.
        last = agent.actor.net[-1]
        torch.nn.init.zeros_(last.weight)
        torch.nn.init.zeros_(last.bias)

        agent.actor_target.load_state_dict(
            agent.actor.state_dict()
        )

        replay = ReplayBuffer(
            state_dim=STATE_DIM,
            action_dim=3,
            capacity=100_000,
        )

    elif mode == "eval":
        checkpoint = os.environ["TD3_V2_CHECKPOINT"]
        agent.load(checkpoint)
        agent.actor.eval()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return agent, replay


def add_batch(replay, s, a, r, ns, done):
    if len(s) == 0:
        return

    replay.add(
        s.astype(np.float32),
        a.astype(np.float32),
        r.astype(np.float32),
        ns.astype(np.float32),
        done.astype(np.float32),
    )


def install():
    module = sys.modules.get(MODULE)

    if module is None:
        return

    cls = getattr(module, "EnvWorker", None)

    if cls is None or getattr(
        cls, "_td3_v2_installed", False
    ):
        return

    original = cls.env_evaluate_step

    def wrapped(self, raw_actions, stage_id):
        mode = os.environ["TD3_V2_MODE"]

        if not hasattr(self, "_td3_v2"):
            torch.set_num_threads(2)

            self._td3_v2, self._td3_v2_replay = (
                make_agent(mode)
            )

            self._td3_v2_pending = {}
            self._td3_v2_chunks = 0

            self._td3_v2_dir = Path(
                os.environ["TD3_V2_DIR"]
            )
            self._td3_v2_dir.mkdir(
                parents=True, exist_ok=True
            )

            self._td3_v2_log = (
                self._td3_v2_dir
                / f"worker_{os.getpid()}.csv"
            )

            if mode == "train":
                with self._td3_v2_log.open(
                    "w", newline=""
                ) as f:
                    csv.writer(f).writerow([
                        "chunk",
                        "replay_size",
                        "mean_reward",
                        "mean_abs_residual",
                        "critic_loss",
                        "actor_loss",
                    ])

            print(
                f"[TD3-V2] Initialized mode={mode}",
                flush=True,
            )

        agent = self._td3_v2
        replay = self._td3_v2_replay

        env = self.eval_env_list[stage_id]
        task = find_task(env)

        original_chunk_step = env.chunk_step

        def controlled_chunk_step(actions):
            if actions.ndim != 3 or actions.shape[2] != 7:
                raise RuntimeError(
                    f"Unexpected actions: {actions.shape}"
                )

            chunk_len = actions.shape[1]

            if chunk_len != 5:
                raise RuntimeError(
                    f"Expected 5 actions, got {chunk_len}"
                )

            state = get_state(task, actions)
            n = len(state)

            # The next VLA action is available only now.
            # Complete the previous nonterminal transition.
            if mode == "train":
                pending = self._td3_v2_pending.pop(
                    stage_id, None
                )

                if pending is not None:
                    idx = pending["idx"]

                    add_batch(
                        replay,
                        pending["state"],
                        pending["action"],
                        pending["reward"],
                        state[idx],
                        np.zeros(len(idx), np.float32),
                    )

            # GT scripted residual: evaluation only.
            if mode != "eval":
                raise RuntimeError(
                    "GT scripted hook supports evaluation only"
                )

            max_mm = float(
                os.environ.get("GT_MAX_MM", "2")
            )
            gain = float(
                os.environ.get("GT_GAIN", "0.25")
            )
            frame = os.environ.get(
                "GT_TRANSLATION_FRAME", "world"
            )

            if max_mm < 0 or gain < 0:
                raise ValueError("Invalid GT correction settings")

            # State[0:3] = object position - gripper position.
            # These coordinates are assumed to be world-aligned.
            error_world = state[:, :3].astype(
                np.float32
            ).copy()

            distance = np.linalg.norm(
                error_world, axis=1
            )

            # Approximate approach-stage gate.
            # No correction when already very close to object.
            active = (
                (distance > 0.04) &
                (distance < 0.25)
            )

            if frame == "world":
                error_action = error_world

            elif frame == "root":
                # Convert world displacement to robot-root axes.
                quat = as_numpy(
                    task.agent.robot.pose.q
                ).astype(np.float32)

                if quat.ndim == 1:
                    quat = quat[None, :]

                if len(quat) == 1 and n > 1:
                    quat = np.repeat(quat, n, axis=0)

                if quat.shape != (n, 4):
                    raise RuntimeError(
                        f"Unexpected root quaternion: {quat.shape}"
                    )

                # ManiSkill quaternion convention: w, x, y, z.
                qvec = -quat[:, 1:4]
                qw = quat[:, 0:1]

                t = 2.0 * np.cross(
                    qvec, error_world
                )
                error_action = (
                    error_world
                    + qw * t
                    + np.cross(qvec, t)
                )

            else:
                raise ValueError(
                    f"Unknown GT_TRANSLATION_FRAME: {frame}"
                )

            limit = max_mm / 1000.0

            residual = np.clip(
                gain * error_action,
                -limit,
                limit,
            ).astype(np.float32)

            residual[~active] = 0.0

            if not hasattr(self, "_gt_script_chunks"):
                self._gt_script_chunks = 0

            self._gt_script_chunks += 1

            if self._gt_script_chunks in (1, 8, 16):
                print(
                    f"[GT-SCRIPT] "
                    f"chunk={self._gt_script_chunks} "
                    f"max_mm={max_mm} "
                    f"gain={gain} "
                    f"frame={frame} "
                    f"active={int(active.sum())}/{n} "
                    f"mean_distance={distance.mean():.4f} "
                    f"mean_residual_mm="
                    f"{np.abs(residual).mean() * 1000:.4f} "
                    f"first_error={error_world[0]} "
                    f"first_residual={residual[0]}",
                    flush=True,
                )

            # Scale the residual during evaluation.
            scale = (
                float(os.environ.get("TD3_RESIDUAL_SCALE", "1.0"))
                if mode == "eval" else 1.0
            )
            if not 0.0 <= scale <= 1.0:
                raise ValueError(f"Invalid residual scale: {scale}")

            modified = actions.clone()

            delta = torch.as_tensor(
                residual * scale / chunk_len,
                device=modified.device,
                dtype=modified.dtype,
            )

            modified[:, :, :3] += delta[:, None, :]

            result = original_chunk_step(modified)

            if mode == "train":
                (
                    _obs,
                    chunk_rewards,
                    terminations,
                    truncations,
                    _infos,
                ) = result

                rewards = as_numpy(chunk_rewards)

                terminated = as_numpy(
                    terminations.any(dim=1)
                ).astype(bool)

                truncated = as_numpy(
                    truncations.any(dim=1)
                ).astype(bool)

                done = terminated | truncated

                weights = GAMMA ** np.arange(
                    chunk_len, dtype=np.float32
                )

                returns = (
                    rewards * weights[None, :]
                ).sum(axis=1).astype(np.float32)

                # Terminal rows do not bootstrap.
                # This pilot also treats truncation as terminal.
                if done.any():
                    add_batch(
                        replay,
                        state[done],
                        residual[done],
                        returns[done],
                        np.zeros(
                            (int(done.sum()), STATE_DIM),
                            dtype=np.float32,
                        ),
                        np.ones(
                            int(done.sum()), np.float32
                        ),
                    )

                # Nonterminal rows wait for the next VLA action.
                active = np.flatnonzero(~done)

                if len(active):
                    self._td3_v2_pending[stage_id] = {
                        "idx": active,
                        "state": state[active].copy(),
                        "action": residual[active].copy(),
                        "reward": returns[active].copy(),
                    }

                metrics = None

                if replay.size >= BATCH_SIZE:
                    for _ in range(n):
                        metrics = agent.train_step(
                            replay,
                            batch_size=BATCH_SIZE,
                        )

                self._td3_v2_chunks += 1
                count = self._td3_v2_chunks

                critic = (
                    metrics["critic_loss"]
                    if metrics else None
                )
                actor = (
                    metrics["actor_loss"]
                    if metrics else None
                )

                with self._td3_v2_log.open(
                    "a", newline=""
                ) as f:
                    csv.writer(f).writerow([
                        count,
                        replay.size,
                        float(returns.mean()),
                        float(np.abs(residual).mean()),
                        critic,
                        actor,
                    ])

                if count % 4 == 0:
                    agent.save(
                        self._td3_v2_dir
                        / "latest_td3_v2.pt"
                    )

                    print(
                        f"[TD3-V2] chunk={count} "
                        f"replay={replay.size} "
                        f"critic={critic} "
                        f"actor={actor} "
                        f"sat_rate={np.mean(np.abs(residual) >= MAX_RESIDUAL - 1e-6):.1%}",
                        flush=True,
                    )

            else:
                self._td3_v2_chunks += 1

                if self._td3_v2_chunks == 1 or self._td3_v2_chunks % 16 == 0:
                    print(
                        f"[TD3-V2] Eval chunk={self._td3_v2_chunks} "
                        f"mean_abs={np.abs(residual).mean():.6f} "
                        f"sat_rate={np.mean(np.abs(residual) >= MAX_RESIDUAL - 1e-6):.1%} "
                        f"first={residual[0]}",
                        flush=True,
                    )

            return result

        env.chunk_step = controlled_chunk_step

        try:
            return original(self, raw_actions, stage_id)
        finally:
            env.chunk_step = original_chunk_step

    cls.env_evaluate_step = wrapped
    cls._td3_v2_installed = True

    print("[TD3-V2] Hook installed", flush=True)


def hooked_import(
    name, globals=None, locals=None, fromlist=(), level=0
):
    result = original_import(
        name, globals, locals, fromlist, level
    )

    if MODULE in sys.modules:
        install()

    return result


if os.environ.get("TD3_V2_MODE"):
    builtins.__import__ = hooked_import
