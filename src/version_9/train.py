import math
import torch
from torch import nn
from torch.distributions import Categorical
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import TransformedEnv
from torchrl.modules import ProbabilisticActor, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tensordict.nn import TensorDictModule
from tensordict import TensorDict

import sys
import os
# Join src to the file

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from envs.pheromone_env import PheromoneEnv
from agents.models_v9 import ActorNet, CriticNet

# -----------------
# Hyperparameters
# -----------------
LR = 3e-4
GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 1.0
FRAMES_PER_BATCH = 4096
TOTAL_FRAMES = 200_000
MINIBATCH_SIZE = 256
PPO_EPOCHS = 4
MAX_STEPS = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float32

BEST_MODEL_PATH = "ppo_ctde_actor_pheromone_best.pth"
FINAL_MODEL_PATH = "ppo_ctde_actor_pheromone.pth"

torch.manual_seed(0)

# -----------------
# Environment (CTDE)
# -----------------
base_env = PheromoneEnv(num_agents=2, grid_size=10, agent_view=5, device=DEVICE, max_steps=MAX_STEPS)
# StepCounter removed; env internally handles step_count and termination
env = TransformedEnv(base_env)


action_high = int(base_env.action_spec["agents", "action"].high.max().item())
actor_net = ActorNet(in_channels=base_env.channels, n_actions=action_high + 1, base_env=base_env).to(DEVICE)
critic_net = CriticNet(in_channels=base_env.channels, base_env=base_env).to(DEVICE)

actor_module = TensorDictModule(
    actor_net,
    in_keys=[("agents", "observation"), ("agents", "carrying")],
    out_keys=[("agents", "logits")],
)
actor = ProbabilisticActor(
    module=actor_module,
    in_keys=[("agents", "logits")],
    out_keys=[("agents", "action")],
    spec=base_env.action_spec,
    distribution_class=Categorical,
    return_log_prob=True,
)

critic_module = ValueOperator(
    module=TensorDictModule(
        critic_net,
        in_keys=["global"],
        out_keys=["state_value"],
    ),
    in_keys=["global"],
)

# -----------------
# Collector & Buffer
# -----------------
collector = SyncDataCollector(
    env,
    policy=actor,
    frames_per_batch=FRAMES_PER_BATCH,
    total_frames=TOTAL_FRAMES,
    device=DEVICE,
    storing_device=DEVICE,
)

buffer = TensorDictReplayBuffer(
    storage=LazyTensorStorage(max_size=FRAMES_PER_BATCH, device=DEVICE),
    batch_size=MINIBATCH_SIZE,
)

# -----------------
# Losses
# -----------------
loss_module = ClipPPOLoss(
    actor_network=actor,
    critic_network=critic_module,
    clip_epsilon=CLIP_EPS,
    entropy_coef=ENTROPY_COEF,
    normalize_advantage=True,
)
loss_module.set_keys(
    reward="reward",
    done="done",
    terminated="terminated",
    action=("agents", "action"),
    sample_log_prob=("agents", "action_log_prob"),
    value="state_value",
    advantage=("agents", "advantage"),
)

advantage_module = GAE(
    gamma=GAMMA,
    lmbda=LAMBDA,
    value_network=critic_module,
    average_gae=True,
)

optim = torch.optim.Adam(loss_module.parameters(), lr=LR)

# -----------------
# Training Loop
# -----------------
print(f"Device: {DEVICE}")
print("Starting PPO-CTDE training on pheromone env...")

frame_count = 0
batch_idx = 0
best_mean_reward = float("-inf")

try:
    for data in collector:
        frame_count += data.numel()

        # CTDE: sum agent rewards -> root reward
        agent_rewards = data.get(("next", "agents", "reward"))  # (B, A, 1)
        root_reward = agent_rewards.sum(dim=1)  # (B, 1)
        data.set(("next", "reward"), root_reward)
        data.set("reward", root_reward)

        with torch.no_grad():
            advantage_module(data)

        # Broadcast advantage to agents
        adv_root = data.get("advantage")  # (B, 1)
        adv_agents = adv_root.unsqueeze(1).expand_as(agent_rewards)
        data.set(("agents", "advantage"), adv_agents)

        # Flatten time/batch dims if present
        flat = data.reshape(-1)
        buffer.extend(flat)

        # PPO updates
        for _ in range(PPO_EPOCHS):
            for _ in range(math.ceil(FRAMES_PER_BATCH / MINIBATCH_SIZE)):
                batch = buffer.sample()
                loss_dict = loss_module(batch)
                loss = loss_dict["loss_objective"] + loss_dict["loss_critic"] + loss_dict["loss_entropy"]
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), MAX_GRAD_NORM)
                optim.step()

        batch_idx += 1
        mean_rew = root_reward.mean().item()
        print(f"Batch {batch_idx} | Frames {frame_count} | MeanReward {mean_rew:.4f}")

        if mean_rew > best_mean_reward:
            best_mean_reward = mean_rew
            torch.save(actor_net.state_dict(), BEST_MODEL_PATH)
            print(f"New best mean reward {best_mean_reward:.4f} saved to {BEST_MODEL_PATH}")

        if frame_count >= TOTAL_FRAMES:
            break

except KeyboardInterrupt:
    print("Training interrupted by user.")

print("Training complete.")
# save final model
torch.save(actor_net.state_dict(), FINAL_MODEL_PATH)
print(f"Final model saved to {FINAL_MODEL_PATH}")
