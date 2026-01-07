import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
from torch.distributions import Categorical
import sys
import os
from torch import nn

# --- PATH SETUP ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.envs.ant_env import AntSortingEnv
from src.agents.models import LocalActorNet

# ==========================================
# 2. SETUP
# ==========================================
class Config:
    num_agents = 1
    grid_size = 10
    device = "cpu"

cfg = Config()
env = AntSortingEnv(num_agents=cfg.num_agents, grid_size=cfg.grid_size, device=cfg.device)

# Load Network
action_spec = env.full_action_spec["agents", "action"]
n_actions = 5

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

# Load Checkpoint
try:
    actor.load_state_dict(torch.load("actor_checkpoint.pth", map_location=cfg.device))
    print("Checkpoint loaded.")
except Exception as e:
    print(f"Warning: Could not load checkpoint ({e}). Running with random weights.")

# ==========================================
# 3. RUN EPISODE
# ==========================================
print("Generating frames...")
td = env.reset()
frames = []
max_steps = 100

for i in range(max_steps):
    with torch.no_grad():
        actor(td)
    
    # Snapshot data
    # Global grid (Hidden state)
    grid = td["global_state_hidden"].clone().numpy()
    
    # Agent 0 Position (Hidden state)
    # We need this to draw the "FOV Box" on the main map
    agent_pos = td["agents_pos_hidden"][0].clone().numpy() # [x, y]
    
    # Agent 0 Perception (The 5x5 Grid)
    # Obs is (Num_Agents, 26). We take Agent 0.
    # First 25 are the grid, last 1 is carrying.
    obs = td["agents", "observation"][0].clone().numpy()
    local_grid_flat = obs[:25] * 3.0 # Denormalize roughly
    local_grid = local_grid_flat.reshape(5, 5)
    
    frames.append((grid, local_grid, agent_pos))
    
    td = env.step(td)
    td = td["next"]

# ==========================================
# 4. VISUALIZE
# ==========================================
fig, (ax_main, ax_local) = plt.subplots(1, 2, figsize=(10, 5))

# Colors: -1=Wall(Black), 0=Empty(White), 1=Red, 2=Green, 3=Blue
cmap = ListedColormap(['black', 'white', '#ff4d4d', '#4dff4d', '#4d4dff'])
bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)

def update(frame_idx):
    ax_main.clear()
    ax_local.clear()
    
    global_grid, local_view, pos = frames[frame_idx]
    
    # --- LEFT: GLOBAL MAP ---
    ax_main.imshow(global_grid, cmap=cmap, norm=norm, origin='upper')
    ax_main.set_title("Global Map (Agent 0 highlighted)")
    
    # Draw Agent 0
    # Note: Matrix indexing [x, y] -> Plotting (y, x)
    ax_main.scatter(pos[1], pos[0], c='gold', s=150, edgecolors='black', label='Agent 0', zorder=10)
    
    # Draw FOV Box (5x5) centered on agent
    # Agent is at center of 5x5. So box starts at x-2, y-2
    rect = Rectangle((pos[1]-2.5, pos[0]-2.5), 5, 5, 
                     linewidth=2, edgecolor='gold', facecolor='none', linestyle='--')
    ax_main.add_patch(rect)
    
    # --- RIGHT: LOCAL VIEW ---
    # This is exactly what the CNN sees
    ax_local.imshow(local_view, cmap=cmap, norm=norm, origin='upper')
    ax_local.set_title("Agent 0 Internal View (5x5)")
    
    # Add grid lines to emphasize pixels
    ax_local.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax_local.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax_local.grid(which='minor', color='gray', linestyle='-', linewidth=1)

ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=200)
ani.save("debug_vision.gif", writer='pillow')
print("Saved debug_vision.gif")