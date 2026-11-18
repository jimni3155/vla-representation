import builtins
import os
import sys

import numpy as np
import torch

MODULE = "rlinf.workers.env.env_worker"
original_import = builtins.__import__


def as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_gt_state(task):
    qpos = as_numpy(task.agent.robot.get_qpos()).astype(np.float32)

    gripper = as_numpy(
        (
            task.agent.finger1_link.pose.p
            + task.agent.finger2_link.pose.p
        ) / 2
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

    return np.asarray(rows, dtype=np.float32)


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
                f"Cannot locate ManiSkill task: {type(task)}"
            )

        next_task = getattr(inner, "unwrapped", inner)

        if next_task is task:
            raise RuntimeError("Environment unwrapping stalled")

        task = next_task

    raise RuntimeError("ManiSkill task not found")


def install():
    module = sys.modules.get(MODULE)
    if module is None:
        return

    cls = getattr(module, "EnvWorker", None)
    if cls is None or getattr(cls, "_td3_eval_installed", False):
        return

    original = cls.env_evaluate_step

    def wrapped(self, raw_actions, stage_id):
        from td3 import TD3

        if not hasattr(self, "_td3_eval_actor"):
            torch.set_num_threads(2)

            agent = TD3(
                state_dim=14,
                action_dim=3,
                max_action=0.005,
                chunk_size=5,
                device="cpu",
            )

            agent.load(os.environ["TD3_EVAL_CHECKPOINT"])
            agent.actor.eval()

            self._td3_eval_actor = agent
            self._td3_eval_count = 0

            print("[TD3-EVAL] Checkpoint loaded", flush=True)

        env = self.eval_env_list[stage_id]
        task = find_task(env)
        original_chunk_step = env.chunk_step

        def eval_chunk_step(actions):
            state = get_gt_state(task)

            if state.shape[1] != 14:
                raise RuntimeError(
                    f"Invalid state shape: {state.shape}"
                )

            residual = self._td3_eval_actor.select_action(
                state, noise_std=0.0
            )

            modified = actions.clone()

            modified[:, 0, :3] += torch.as_tensor(
                residual,
                device=modified.device,
                dtype=modified.dtype,
            )

            self._td3_eval_count += 1

            if self._td3_eval_count == 1:
                print(
                    "[TD3-EVAL] Residual xyz:",
                    residual[0],
                    flush=True,
                )

            return original_chunk_step(modified)

        env.chunk_step = eval_chunk_step

        try:
            return original(self, raw_actions, stage_id)
        finally:
            env.chunk_step = original_chunk_step

    cls.env_evaluate_step = wrapped
    cls._td3_eval_installed = True

    print("[TD3-EVAL] Hook installed", flush=True)


def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
    result = original_import(
        name, globals, locals, fromlist, level
    )

    if MODULE in sys.modules:
        install()

    return result


if os.environ.get("TD3_EVAL_CHECKPOINT"):
    builtins.__import__ = hooked_import
