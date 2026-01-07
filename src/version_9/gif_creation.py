import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch import nn
from torch.distributions import Categorical
from tensordict.nn import TensorDictModule
import sys
import os

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.envs.ant_env import AntSortingEnv
# from src.agents.models import LocalActorNet  # <--- Must match training!
from envs.pheromone_env import PheromoneEnv  # Using the same ActorNet for simplicity

torch.manual_seed(0)
# ==========================================
# 1. CONFIGURATION
# ==========================================
#num_agents=2, grid_size=10, agent_view=5
class Config:
    num_agents = 2
    grid_size = 10
    agent_view = 5
    max_steps = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
print(f"Using device: {cfg.device}")

# ==========================================
# 2. ENVIRONMENT & MODEL
# ==========================================
env = PheromoneEnv(
    num_agents=cfg.num_agents,
    grid_size=cfg.grid_size,
    agent_view=cfg.agent_view,
    device=cfg.device,
    max_steps=cfg.max_steps,
)
class ActorNet(nn.Module):
    def __init__(self, in_channels: int, n_actions: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # agent_view assumed square; two convs keep spatial dims
        self.mlp = nn.Sequential(
            nn.Linear(64 * env.agent_view * env.agent_view + 1, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, obs: torch.Tensor, carrying: torch.Tensor):
        # Support both batched (B, A, C, H, W) and unbatched (A, C, H, W)
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
            return logits  # shape (A, actions) to match unbatched agent dimension
        else:
            raise ValueError(f"Unexpected obs shape {obs.shape}")
# Action spec
action_spec = env.action_spec["agents", "action"]

# --- A. Load the CNN Actor ---
# We use the class from src/models.py so it matches exactly
action_high = int(env.action_spec["agents", "action"].high.max().item())
actor_net = ActorNet(in_channels=env.channels, n_actions=action_high + 1).to(cfg.device)

actor_module = TensorDictModule(
    actor_net,
    in_keys=[("agents", "observation"), ("agents", "carrying")],
    out_keys=[("agents", "logits")],
)

# --- B. Load Checkpoint ---
checkpoint_candidates = [
    "src/version_9/ppo_ctde_actor_pheromone_best.pth",
    "src/version_9/ppo_ctde_actor_pheromone.pth",
]
checkpoint_path = next((p for p in checkpoint_candidates if os.path.exists(p)), None)
if checkpoint_path is None:
    print("No checkpoint found (looked for ppo_ctde_actor_pheromone_best.pth and ppo_ctde_actor_pheromone.pth).")
    sys.exit(1)

try:
    state_dict = torch.load(checkpoint_path, map_location=cfg.device)
    actor_net.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {checkpoint_path}.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    sys.exit(1)

# ==========================================
# 3. RUN EPISODE
# ==========================================
print("Running evaluation episode...")
td = env.reset().to(cfg.device)
frames = []
max_steps = env.max_steps

# Disable autograd for rollout
torch.set_grad_enabled(False)


def build_render_grid(global_tensor: torch.Tensor, candy_types: int) -> np.ndarray:
    global_cpu = global_tensor.cpu()
    walls = global_cpu[0]
    grid = np.zeros_like(walls.numpy(), dtype=np.int64)
    grid[walls > 0.5] = -1
    for t in range(candy_types):
        candy_layer = global_cpu[2 + 2 * t]
        grid[candy_layer > 0.5] = t + 1
    return grid


for i in range(max_steps):
    with torch.no_grad():
        obs = td["agents", "observation"].to(cfg.device)
        carrying = td["agents", "carrying"].to(cfg.device).float()
        logits = actor_net(obs, carrying)
        actions = Categorical(logits=logits).sample().unsqueeze(-1)
        td.set(("agents", "action"), actions)

    actions_np = actions.flatten().cpu().tolist()
    print(f"Step {i} actions: {actions_np}")

    render_grid = build_render_grid(td["global"], env.candy_types)
    agents = td["agents", "self_pos"].cpu().numpy()
    frames.append((render_grid, agents))

    td_step = env.step(td)
    td = td_step.get("next", td_step)

    done = td.get("done")
    if done is not None and done.any():
        break

print(f"Collected {len(frames)} frames. Generating GIF...")

# ==========================================
# 4. RENDER
# ==========================================
fig, ax = plt.subplots(figsize=(6, 6))

# Colors: -1=Black (Wall), 0=White (Empty), 1=Red, 2=Green, 3=Blue
cmap = ListedColormap(['black', 'white', '#ff4d4d', '#4dff4d', '#4d4dff'])
bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)

def update(frame_idx):
    ax.clear()
    grid, agents = frames[frame_idx]
    
    # Plot Grid
    ax.imshow(grid, cmap=cmap, norm=norm, origin='upper')
    
    # Plot Agents
    # Note: agents[:, 1] is x-axis (columns), agents[:, 0] is y-axis (rows)
    ax.scatter(agents[:, 1], agents[:, 0], c='gold', s=120, label='Agents', edgecolors='black', zorder=10)
    
    ax.set_title(f"Step {frame_idx}")
    ax.set_xticks([])
    ax.set_yticks([])

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100)
ani.save("simulation_pheromone.gif", writer='pillow')
print("Saved simulation_pheromone.gif")