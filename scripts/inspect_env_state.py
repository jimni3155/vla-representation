import gymnasium as gym
import mani_skill.envs
import rlinf.envs.sim.maniskill
import torch


def describe(name, value):
    if isinstance(value, torch.Tensor):
        print(f"{name}: shape={tuple(value.shape)}")
        if value.numel() <= 20:
            print(f"  {value.detach().cpu().tolist()}")
    else:
        print(f"{name}: {value}")


def main():
    env = gym.make(
        "PutOnPlateInScene25Main-v3",
        num_envs=1,
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        sim_backend="gpu",
        render_mode="rgb_array",
        obj_set="train",
        max_episode_steps=80,
    )

    try:
        obs, _ = env.reset(seed=42)
        task = env.unwrapped

        print("\n=== Observation ===")
        print("Keys:", list(obs.keys()))

        print("\n=== Scene actors ===")
        for name, actor in task.scene.state_dict_registry.actors.items():
            print(name)
            describe("  position", actor.pose.p)

        print("\n=== Task attributes ===")
        keywords = ("obj", "plate", "target", "goal", "source")
        for name in dir(task):
            if name.startswith("_"):
                continue
            if any(word in name.lower() for word in keywords):
                print(name)

        print("\n=== Robot ===")
        print("Robot type:", type(task.agent).__name__)
        print("Robot attributes:")
        for name in dir(task.agent):
            if "tcp" in name or "ee_" in name or "grasp" in name:
                print(" ", name)

        print("\n=== State dict ===")
        state = task.get_state_dict()
        print("Keys:", list(state.keys()))
        for name, value in state.items():
            if isinstance(value, dict):
                print(f"  {name}: {list(value.keys())[:20]}")
            else:
                describe(f"  {name}", value)

    finally:
        env.close()


if __name__ == "__main__":
    main()
