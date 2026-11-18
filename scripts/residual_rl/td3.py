import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, state_dim, action_dim=3, capacity=100_000):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), np.float32)
        self.actions = np.zeros((capacity, action_dim), np.float32)
        self.rewards = np.zeros((capacity, 1), np.float32)
        self.next_states = np.zeros((capacity, state_dim), np.float32)
        self.dones = np.zeros((capacity, 1), np.float32)

    def add(self, state, action, reward, next_state, done):
        n = len(state)

        for i in range(n):
            j = self.ptr
            self.states[j] = state[i]
            self.actions[j] = action[i]
            self.rewards[j] = reward[i]
            self.next_states[j] = next_state[i]
            self.dones[j] = done[i]

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, batch_size)

        arrays = (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
        return tuple(
            torch.as_tensor(x, device=device)
            for x in arrays
        )


def mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = mlp(state_dim, action_dim)
        self.max_action = max_action

    def forward(self, state):
        return self.net(state).tanh() * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = mlp(state_dim + action_dim, 1)
        self.q2 = mlp(state_dim + action_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


class TD3:
    def __init__(
        self,
        state_dim,
        action_dim=3,
        max_action=0.005,
        discount=0.99,
        chunk_size=5,
        tau=0.005,
        policy_noise=0.001,
        noise_clip=0.002,
        policy_freq=2,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount ** chunk_size
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

        self.actor = Actor(
            state_dim, action_dim, max_action
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(
            state_dim, action_dim
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=3e-4
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=3e-4
        )

    @torch.no_grad()
    def select_action(self, state, noise_std=0.0):
        x = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            device=self.device,
        )

        action = self.actor(x).cpu().numpy()

        if noise_std > 0:
            action += np.random.normal(
                0, noise_std, size=action.shape
            )

        return np.clip(
            action,
            -self.max_action,
            self.max_action,
        ).astype(np.float32)

    def train_step(self, replay, batch_size=128):
        if replay.size < batch_size:
            return None

        self.total_it += 1

        state, action, reward, next_state, done = (
            replay.sample(batch_size, self.device)
        )

        with torch.no_grad():
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            target_q1, target_q2 = self.critic_target(
                next_state, next_action
            )

            target_q = torch.min(target_q1, target_q2)
            target_q = (
                reward
                + (1.0 - done) * self.discount * target_q
            )

        current_q1, current_q2 = self.critic(state, action)

        critic_loss = (
            F.mse_loss(current_q1, target_q)
            + F.mse_loss(current_q2, target_q)
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        metrics = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": None,
        }

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic(
                state, self.actor(state)
            )[0].mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for network, target in (
                (self.actor, self.actor_target),
                (self.critic, self.critic_target),
            ):
                for p, tp in zip(
                    network.parameters(),
                    target.parameters(),
                ):
                    tp.data.mul_(1.0 - self.tau)
                    tp.data.add_(self.tau * p.data)

            metrics["actor_loss"] = float(
                actor_loss.item()
            )

        return metrics

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "max_action": self.max_action,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "total_it": self.total_it,
        }, path)

    def load(self, path):
        data = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        assert data["state_dim"] == self.state_dim
        assert data["action_dim"] == self.action_dim
        assert data["max_action"] == self.max_action

        for key in (
            "actor",
            "actor_target",
            "critic",
            "critic_target",
        ):
            getattr(self, key).load_state_dict(data[key])

        self.actor_optimizer.load_state_dict(
            data["actor_optimizer"]
        )
        self.critic_optimizer.load_state_dict(
            data["critic_optimizer"]
        )
        self.total_it = data["total_it"]


if __name__ == "__main__":
    torch.set_num_threads(2)

    np.random.seed(42)
    torch.manual_seed(42)

    state_dim = 14
    batch_size = 128

    td3 = TD3(state_dim=state_dim)
    replay = ReplayBuffer(state_dim=state_dim)

    for _ in range(256):
        s = np.random.randn(4, state_dim).astype(np.float32)
        a = td3.select_action(s, noise_std=0.001)
        ns = s + np.random.randn(4, state_dim).astype(np.float32) * 0.01
        r = np.random.randn(4).astype(np.float32)
        done = np.zeros(4, np.float32)
        replay.add(s, a, r, ns, done)

    for _ in range(100):
        metrics = td3.train_step(replay, batch_size)

    path = "results/residual/td3/smoke_test.pt"
    td3.save(path)

    restored = TD3(state_dim=state_dim)
    restored.load(path)

    state = np.random.randn(3, state_dim).astype(np.float32)
    a1 = td3.select_action(state)
    a2 = restored.select_action(state)

    assert np.allclose(a1, a2)
    assert np.isfinite(a1).all()

    print("TD3 smoke test: PASSED")
    print("Replay size:", replay.size)
    print("Training updates:", td3.total_it)
    print("Critic loss:", metrics["critic_loss"])
    print("Actor loss:", metrics["actor_loss"])
    print("Action range:", a1.min(), a1.max())
    print("Checkpoint:", path)
