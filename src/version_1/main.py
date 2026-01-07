import torch
import numpy as np
from torch import nn
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.modules import ProbabilisticActor, ValueOperator
from tensordict.nn import TensorDictModule
from tqdm import tqdm
from torch.distributions import Categorical
import sys
import os

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.envs.ant_env import AntSortingEnv
from src.agents.models import LocalActorNet, LocalCriticNet


class Config:
    # Env
    num_agents = 1
    grid_size = 10

    # Training
    total_frames = 50_000
    frames_per_batch = 2000
    mini_batch_size = 64
    ppo_epochs = 10
    lr = 3e-4
    clip_epsilon = 0.2
    gamma = 0.99
    lmbda = 0.95
    max_grad_norm = 1.0
    entropy_coef = 0.2
    device = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    cfg = Config()
    print(f"Using device: {cfg.device}")

    env = AntSortingEnv(num_agents=cfg.num_agents, grid_size=cfg.grid_size, device=cfg.device)

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
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )

    critic_net = LocalCriticNet().to(cfg.device)
    critic = ValueOperator(
        module=TensorDictModule(
            critic_net,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        ),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )

    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.total_frames,
        split_trajs=False,
        device=cfg.device,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(max_size=10_000),
        batch_size=cfg.mini_batch_size,
    )

    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.clip_epsilon,
        entropy_bonus=True,
        entropy_coef=cfg.entropy_coef,
        loss_critic_type="l2",
        normalize_advantage=True,
        normalize_advantage_exclude_dims=[1],
    )

    loss_module.set_keys(
        reward=("agents", "reward"),
        action=("agents", "action"),
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    advantage_module = GAE(
        gamma=cfg.gamma,
        lmbda=cfg.lmbda,
        value_network=critic,
        average_gae=True,
    )

    advantage_module.set_keys(
        reward=("agents", "reward"),
        value=("agents", "state_value"),
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

    print("Starting training loop...")
    pbar = tqdm(total=cfg.total_frames)

    for i, tensordict_data in enumerate(collector):
        with torch.no_grad():
            advantage_module(tensordict_data)

        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view)

        for _ in range(cfg.ppo_epochs):
            sub_batch = replay_buffer.sample().to(cfg.device)

            loss_vals = loss_module(sub_batch)
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            optimizer.zero_grad()
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
            optimizer.step()

        avg_reward = tensordict_data["next", "agents", "reward"].mean().item()
        pbar.update(tensordict_data.numel())
        pbar.set_description(f"Reward: {avg_reward:.4f} | Loss: {loss_value.item():.4f}")

        if i % 10 == 0:
            torch.save(actor.state_dict(), "actor_checkpoint.pth")

    torch.save(actor.state_dict(), "actor_checkpoint.pth")
    print("Training Complete.")


if __name__ == "__main__":
    main()
