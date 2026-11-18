import builtins
import os
import sys

import numpy as np
import torch


MODULE = "rlinf.workers.env.env_worker"
original_import = builtins.__import__


def install():
    module = sys.modules.get(MODULE)
    if module is None:
        return

    worker_cls = getattr(module, "EnvWorker", None)
    if worker_cls is None or getattr(worker_cls, "_residual_installed", False):
        return

    original = worker_cls.env_evaluate_step

    def evaluate_with_residual(self, raw_actions, stage_id):
        mode = os.environ.get("RESIDUAL_MODE", "zero")
        env = self.eval_env_list[stage_id]
        original_chunk_step = env.chunk_step

        if mode not in ("zero", "test_x"):
            raise ValueError(f"Unknown residual mode: {mode}")

        def chunk_step_with_residual(actions):
            modified = (
                actions.clone()
                if isinstance(actions, torch.Tensor)
                else np.array(actions, copy=True)
            )

            if mode == "test_x" and not getattr(self, "_residual_checked", False):
                modified[:, 0, 0] += 0.001

            if not getattr(self, "_residual_checked", False):
                before = (
                    actions.detach().cpu().numpy()
                    if isinstance(actions, torch.Tensor)
                    else np.asarray(actions)
                )
                after = (
                    modified.detach().cpu().numpy()
                    if isinstance(modified, torch.Tensor)
                    else np.asarray(modified)
                )

                difference = after - before

                print(f"[residual] mode={mode}", flush=True)
                print(f"[residual] action shape={after.shape}", flush=True)
                print(
                    f"[residual] original xyz={before[0, 0, :3]}",
                    flush=True,
                )
                print(
                    f"[residual] modified xyz={after[0, 0, :3]}",
                    flush=True,
                )
                print(
                    f"[residual] maximum difference="
                    f"{np.max(np.abs(difference)):.7f}",
                    flush=True,
                )

                self._residual_checked = True

            return original_chunk_step(modified)

        env.chunk_step = chunk_step_with_residual
        try:
            return original(self, raw_actions, stage_id)
        finally:
            env.chunk_step = original_chunk_step

    worker_cls.env_evaluate_step = evaluate_with_residual
    worker_cls._residual_installed = True
    print("[residual] Hook installed", flush=True)


def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
    result = original_import(name, globals, locals, fromlist, level)
    if MODULE in sys.modules:
        install()
    return result


if os.environ.get("RESIDUAL_MODE"):
    builtins.__import__ = hooked_import
