import argparse
import time
import torch
from torch import nn
from torch.distributions import Categorical
from tensordict import TensorDict

import sys
import os
# Join src to the file

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from envs.pheromone_env import PheromoneEnv
from agents.models_v9 import ActorNet, CriticNet

PHEROMONE_CHARS = " .:-=+*#%@"


def render_global(grid: torch.Tensor) -> str:
    """Render the full grid (channels-first) to ASCII."""
    walls = grid[0] > 0.5
    agents = grid[1] > 0.5

    num_candy_types = (grid.shape[0] - 2) // 2
    candy_ch = 2
    pher_ch = 3

    lines = []
    h, w = walls.shape
    for r in range(h):
        line = []
        for c in range(w):
            if agents[r, c]:
                line.append("A")
                continue
            if walls[r, c]:
                line.append("#")
                continue
            candy = 0
            for t in range(num_candy_types):
                if grid[candy_ch + 2 * t, r, c] > 0.5:
                    candy = t + 1
                    break
            if candy:
                line.append(str(candy))
                continue
            pher_vals = [grid[pher_ch + 2 * t, r, c].item() for t in range(num_candy_types)]
            v = max(pher_vals) if pher_vals else 0.0
            idx = min(int(v * (len(PHEROMONE_CHARS) - 1)), len(PHEROMONE_CHARS) - 1)
            line.append(PHEROMONE_CHARS[idx])
        lines.append(" ".join(line))
    return "\n".join(lines)


class ActorNet(nn.Module):
    def __init__(self, in_channels: int, view: int, n_actions: int = 5):
        super().__init__()
        self.view = view
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(64 * view * view + 1, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, obs: torch.Tensor, carrying: torch.Tensor):
        # Support batched (B, A, C, H, W) and unbatched (A, C, H, W)
        if obs.dim() == 5:
            b, a, c, h, w = obs.shape
            obs_f = obs.view(b * a, c, h, w)
            carry_f = carrying.view(b * a, 1)
            logits = self.mlp(torch.cat([self.conv(obs_f), carry_f], dim=-1))
            return logits.view(b, a, -1)
        elif obs.dim() == 4:
            a, c, h, w = obs.shape
            obs_f = obs
            carry_f = carrying.view(a, 1)
            logits = self.mlp(torch.cat([self.conv(obs_f), carry_f], dim=-1))
            return logits
        else:
            raise ValueError(f"Unexpected obs shape {obs.shape}")


def load_actor(env: PheromoneEnv, path: str, device: str) -> ActorNet:
    action_high = int(env.action_spec["agents", "action"].high.max().item())
    net = ActorNet(in_channels=env.channels, view=env.agent_view, n_actions=action_high + 1).to(device)
    state = torch.load(path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    return net


def main(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    env = PheromoneEnv(num_agents=args.num_agents, grid_size=args.grid_size, agent_view=args.agent_view,
                       candy_types=args.candy_types, device=device, max_steps=args.max_env_steps)
    actor = load_actor(env, args.model, device)

    td = env.reset()
    total_reward = 0.0
    torch.set_grad_enabled(False)

    for step in range(args.steps):
        obs = td["agents", "observation"].to(device)
        carrying = td["agents", "carrying"].to(device).float()
        logits = actor(obs, carrying)
        actions = Categorical(logits=logits).sample().unsqueeze(-1)
        actions_cpu = actions.squeeze(-1).detach().cpu().tolist()
        carrying_cpu = carrying.squeeze(-1).long().cpu().tolist()

        action_td = TensorDict({
            "agents": TensorDict({"action": actions}, batch_size=[env.num_agents])
        }, batch_size=[]).to(device)

        td = env.step(action_td)
        step_td = td.get("next", td)

        reward_agents = step_td.get(("agents", "reward"), torch.zeros((env.num_agents, 1), device=device))
        reward = reward_agents.sum().item()
        total_reward += reward

        if args.render:
            grid = step_td["global"].cpu()
            print("\n" + "=" * (2 * env.grid_size))
            print(f"Step {step+1} | Reward {reward:.3f} | Total {total_reward:.3f}")
            print(f"Actions: {actions_cpu} | Carrying: {carrying_cpu}")
            print(render_global(grid))
            time.sleep(args.sleep)

        if step_td.get("done", torch.tensor([False], device=device)).any():
            break

        td = step_td

    print(f"Episode finished. Steps: {step+1}, Total reward: {total_reward:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="src/version_9/ppo_ctde_actor_pheromone.pth", help="Path to actor weights")
    parser.add_argument("--steps", type=int, default=100, help="Max steps to rollout")
    parser.add_argument("--sleep", type=float, default=0.2, help="Sleep between rendered steps")
    parser.add_argument("--render", action="store_true", help="Print ASCII render each step")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu/cuda)")
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--grid_size", type=int, default=10)
    parser.add_argument("--agent_view", type=int, default=5)
    parser.add_argument("--candy_types", type=int, default=3)
    parser.add_argument("--max_env_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None, help="Optional torch.manual_seed for reproducibility")
    args = parser.parse_args()
    main(args)
