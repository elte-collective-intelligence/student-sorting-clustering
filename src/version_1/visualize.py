import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from torch import nn
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from torch.distributions import Categorical
import sys
import os

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.envs.ant_env import AntSortingEnv
from src.agents.models import LocalActorNet  # <--- Must match training!

# ==========================================
# 1. CONFIGURATION
# ==========================================
class Config:
    num_agents = 1
    grid_size = 10
    device = "cpu"

cfg = Config()
print(f"Using device: {cfg.device}")

# ==========================================
# 2. ENVIRONMENT & MODEL
# ==========================================
env = AntSortingEnv(num_agents=cfg.num_agents, grid_size=cfg.grid_size, device=cfg.device)

# New Action Spec
action_spec = env.full_action_spec["agents", "action"]
n_actions = 5 

# --- A. Load the CNN Actor ---
# We use the class from src/models.py so it matches exactly
actor_net = LocalActorNet(action_dim=n_actions).to(cfg.device)

actor_module = TensorDictModule(
    actor_net,
    in_keys=[("agents", "observation")],
    out_keys=[("agents", "logits")],
)

actor = ProbabilisticActor(
    module=actor_module,
    spec=action_spec,
    in_keys=[("agents", "logits")],
    out_keys=[("agents", "action")],
    distribution_class=Categorical,
    return_log_prob=False, 
)

# --- B. Load Checkpoint ---
checkpoint_path = "actor_checkpoint.pth"
if not os.path.exists(checkpoint_path):
    print(f"Checkpoint '{checkpoint_path}' not found.")
    sys.exit(1)

try:
    actor.load_state_dict(torch.load(checkpoint_path, map_location=cfg.device))
    print("Loaded checkpoint successfully.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    sys.exit(1)

# ==========================================
# 3. RUN EPISODE
# ==========================================
print("Running evaluation episode...")
td = env.reset()
frames = []
max_steps = 200 # Longer episode to see clustering

for i in range(max_steps):
    with torch.no_grad():
        # Pick best action (Deterministic)
        actor(td)
    
    actions = td["agents", "action"]
    print(f"Step {i} Actions: {actions.flatten().tolist()}")
    
    # --- CRITICAL UPDATES FOR NEW ENV STRUCTURE ---
    # 1. Grid is now in 'global_state_hidden'
    grid = td["global_state_hidden"].clone().numpy()
    
    # 2. Positions are now in 'agents_pos_hidden'
    agents = td["agents_pos_hidden"].clone().numpy() # Shape (N, 2)
    
    frames.append((grid, agents))
    
    # Step
    td = env.step(td)
    td = td["next"]

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
ani.save("simulation.gif", writer='pillow')
print("Saved simulation.gif")