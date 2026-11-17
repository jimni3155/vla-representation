import builtins
import os
import sys
from pathlib import Path

import numpy as np


MODULE = "rlinf.envs.sim.maniskill.maniskill_env"
original_import = builtins.__import__


def as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def save_observation(env, obs):
    output = os.environ.get("VLA_PROBE_DIR")
    if not output or "main_images" not in obs:
        return

    task = env.env.unwrapped
    images = as_numpy(obs["main_images"])
    states = as_numpy(obs["states"])
    instructions = obs["task_descriptions"]

    count = getattr(env, "_probe_count", 0)
    env._probe_count = count + 1

    interval = int(os.environ.get("VLA_PROBE_EVERY", "5"))
    if count % interval:
        return

    root = Path(output) / f"worker_{os.getpid()}_{id(env)}"
    root.mkdir(parents=True, exist_ok=True)

    gripper = (
        task.agent.finger1_link.pose.p
        + task.agent.finger2_link.pose.p
    ) / 2
    gripper = as_numpy(gripper)

    source_ids = as_numpy(task.select_carrot_ids).reshape(-1)
    target_ids = as_numpy(task.select_plate_ids).reshape(-1)

    for i in range(len(images)):
        source_name = task.carrot_names[int(source_ids[i])]
        target_name = task.plate_names[int(target_ids[i])]

        object_pos = as_numpy(task.objs_carrot[source_name].pose.p)[i]
        plate_pos = as_numpy(task.objs_plate[target_name].pose.p)[i]
        gripper_pos = gripper[i]

        path = root / f"env_{i:03d}_step_{count:03d}.npz"

        np.savez_compressed(
            path,
            image=images[i],
            state=states[i],
            instruction=str(instructions[i]),
            object_name=str(source_name),
            plate_name=str(target_name),
            object_pos=object_pos,
            plate_pos=plate_pos,
            gripper_pos=gripper_pos,
            gripper_to_object=object_pos - gripper_pos,
            object_to_plate=plate_pos - object_pos,
        )


def install_hook():
    module = sys.modules.get(MODULE)
    if module is None:
        return

    for cls in vars(module).values():
        if not isinstance(cls, type):
            continue
        if cls.__module__ != MODULE:
            continue
        if "_wrap_obs" not in cls.__dict__:
            continue
        if getattr(cls, "_probe_installed", False):
            continue

        original = cls._wrap_obs

        def wrapped(self, *args, _original=original, **kwargs):
            obs = _original(self, *args, **kwargs)
            try:
                save_observation(self, obs)
            except Exception as exc:
                print(f"[probe] Collection failed: {exc}", flush=True)
            return obs

        cls._wrap_obs = wrapped
        cls._probe_installed = True
        print(f"[probe] Hook installed: {cls.__name__}", flush=True)


def hooked_import(name, globals=None, locals=None, fromlist=(), level=0):
    result = original_import(name, globals, locals, fromlist, level)
    if MODULE in sys.modules:
        install_hook()
    return result


if os.environ.get("VLA_PROBE_DIR"):
    builtins.__import__ = hooked_import
