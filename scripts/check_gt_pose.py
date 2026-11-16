import gymnasium as gym
import mani_skill.envs
import rlinf.envs.sim.maniskill
import torch


ENV_ID = "PutOnPlateInScene25Main-v3"
CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


def to_list(tensor):
    return tensor.detach().cpu().squeeze(0).tolist()


def main():
    env = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb+segmentation",
        control_mode=CONTROL_MODE,
        sim_backend="gpu",
        obj_set="train",
        max_episode_steps=80,
    )

    try:
        env.reset(seed=42)
        task = env.unwrapped

        source = task.objs[task.source_obj_name]
        target = task.objs[task.target_obj_name]

        object_pos = source.pose.p
        target_pos = target.pose.p

        finger_left = task.agent.finger1_link.pose.p
        finger_right = task.agent.finger2_link.pose.p
        gripper_pos = (finger_left + finger_right) / 2

        gripper_to_object = object_pos - gripper_pos
        object_to_target = target_pos - object_pos

        print("Instruction:", task.get_language_instruction()[0])
        print("Object:", task.source_obj_name)
        print("Target:", task.target_obj_name)

        print("\nWorld coordinates (m)")
        print("Object:", to_list(object_pos))
        print("Target:", to_list(target_pos))
        print("Gripper:", to_list(gripper_pos))

        print("\nRelative positions (m)")
        print("Gripper -> Object:", to_list(gripper_to_object))
        print("Object -> Target:", to_list(object_to_target))

        print("\nDistances (m)")
        print("Gripper -> Object:",
              torch.linalg.norm(gripper_to_object, dim=-1).item())
        print("Object -> Target:",
              torch.linalg.norm(object_to_target, dim=-1).item())

    finally:
        env.close()


if __name__ == "__main__":
    main()
