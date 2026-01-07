import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Composite, Bounded, Unbounded

class PheromoneEnv(EnvBase):
    def __init__(self, batch_size = None, device = "cpu", grid_size = 10, 
                 num_agents = 3, agent_view = 5, candy_types = 3,
                 max_steps = 100):
        batch_size = batch_size if batch_size is not None else torch.Size([])
        self.device = device
        super().__init__(batch_size=batch_size, device=self.device)
        
        self.grid_size = grid_size
        self.agent_view = agent_view
        self.pad = agent_view // 2
        
        self.candy_types = candy_types
        
        self.num_agents = num_agents
        self.candies_per_type = grid_size // 2
        self.wall_count = grid_size
        
        self.max_steps = max_steps
        self.current_step = 0
        
        # [agents][walls][candy type 1][candy pheromone type 1]...
        self.channels = 2 + 2 * self.candy_types
        
        self.diffusion_kernel = self._build_diffusion_kernel()
        self.diffusion_iters = 3
        self.decay_rate = 0.95

        # Optional agent emission (only when moving)
        self.agent_emit_strength = 0.0

        # Potential-based reward parameters
        self.gradient_gamma = 0.99
        self.time_penalty = -0.01
        self.drop_base_reward = 1.0
        self.drop_neighbor_multiplier = 0.5
        self.pickup_reward = 0.1
        
        self.metric_ema = 0.0
        
        self._make_spec()
        
    def _make_spec(self):
        # 1. Observation Spec
        self.observation_spec = Composite({
            "global": Bounded(
                low = -1,
                high = 1,
                shape = (self.channels, self.grid_size, self.grid_size),
                device = self.device,
                dtype = torch.float
            ),
            "step_count" : Unbounded(
                shape = (1,),
                device = self.device,
                dtype = torch.long
            ),
            "agents": Composite({
                "observation" : Bounded(
                low = -1,
                high = 1,
                shape = (self.num_agents, self.channels, self.agent_view, self.agent_view),
                device = self.device,
                dtype = torch.float
                ),
                "self_pos" : Bounded(
                    low = 0,
                    high = self.grid_size - 1,
                    shape = (self.num_agents, 2),
                    device = self.device,
                    dtype = torch.long
                ),
                "carrying" : Bounded(
                    low = 0,
                    high = self.candy_types,
                    shape = (self.num_agents, 1),
                    device = self.device,
                    dtype = torch.long
                )
            }, shape = (self.num_agents,))    
        })
        
        # 2. Action Spec (4 moves + 1 pickup/drop)
        self.action_spec = Composite({
            "agents": Composite({
                "action": Bounded(
                    low=0, high=4,
                    shape=(self.num_agents, 1),
                    dtype=torch.long, device=self.device
                )
            }, shape=(self.num_agents,))
        })

        # 3. Reward Spec
        self.reward_spec = Composite({
            "agents": Composite({
                "reward": Unbounded(
                    shape=(self.num_agents, 1),
                    dtype=torch.float, device=self.device
                )
            }, shape=(self.num_agents,))
        })

        # 4. Done Spec
        self.done_spec = Composite({
            "done": Bounded(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
            "terminated": Bounded(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
            "truncated": Bounded(low=0, high=1, shape=(1,), dtype=torch.bool, device=self.device),
            "agents": Composite({
                "done": Bounded(low=0, high=1, shape=(self.num_agents, 1), dtype=torch.bool, device=self.device),
                "terminated": Bounded(low=0, high=1, shape=(self.num_agents, 1), dtype=torch.bool, device=self.device),
            }, shape=(self.num_agents,))
        })
    
    def _build_diffusion_kernel(self):
        # calculate the pheromone density based on placed candies
        diffusion_kernel = torch.tensor([[1.0,2.0,1.0],
                                         [2.0,4.0,2.0],
                                         [1.0,2.0,1.0]], device=self.device)
        diffusion_kernel /= diffusion_kernel.sum()
        return diffusion_kernel.view(1,1,3,3)
    
    def _pack_output(self, rewards=None, done=None, terminated=None, truncated=None) -> TensorDict:
        global_grid = torch.zeros((self.channels, self.grid_size, self.grid_size), device=self.device)
        global_grid[0] = self.walls

        agent_map = torch.zeros_like(self.walls)
        agent_map[self.agent_positions[:, 0], self.agent_positions[:, 1]] = 1
        global_grid[1] = agent_map

        channel = 2
        for t in range(1, self.candy_types + 1):
            mask = (self.items == t).float()
            global_grid[channel] = mask
            global_grid[channel + 1] = self.pheromone[t - 1]
            channel += 2

        padded_grid = F.pad(global_grid, (self.pad, self.pad + 1, self.pad, self.pad + 1), value=-1)
        obs_list = []
        for i in range(self.num_agents):
            x, y = self.agent_positions[i]
            obs_list.append(padded_grid[:, x:x + self.agent_view, y:y + self.agent_view])

        obs_batch = torch.stack(obs_list)

        agents_td = TensorDict({
            "observation": obs_batch,
            "self_pos": self.agent_positions.clone(),
            "carrying": self.carrying.clone()
        }, batch_size=[self.num_agents])

        if done is not None:
            agents_td["done"] = done
            agents_td["terminated"] = terminated

        if rewards is not None:
            agents_td["reward"] = rewards

        out = TensorDict({
            "global": global_grid,
            "agents": agents_td,
            "step_count": torch.tensor([self.current_step], device=self.device)
        }, batch_size=self.batch_size)

        if done is not None:
            out_done = done.any().view(1)
            out_terminated = terminated.any().view(1) if terminated is not None else torch.tensor([False], device=self.device)
            out_truncated = truncated.any().view(1) if truncated is not None else torch.tensor([False], device=self.device)
            out["done"] = out_done
            out["terminated"] = out_terminated
            out["truncated"] = out_truncated

        return out
    
    def _update_pheromones(self, agent_emit: torch.Tensor = None):
        wall_mask = (1.0 - self.walls).float().unsqueeze(0).unsqueeze(0)
        decay_rate = self.decay_rate
        for candy_type in range(self.candy_types):
            field = self.pheromone[candy_type].unsqueeze(0).unsqueeze(0)
            emit = (self.items == (candy_type + 1)).float().unsqueeze(0).unsqueeze(0)
            if agent_emit is not None:
                emit = emit + agent_emit[candy_type].unsqueeze(0).unsqueeze(0)

            # Fast propagation: run multiple diffusion steps per environment step.
            for _ in range(self.diffusion_iters):
                field = F.conv2d(field, self.diffusion_kernel, padding=1)
                field = field * decay_rate
                field = field * wall_mask
                field = field + emit

            self.pheromone[candy_type] = field.squeeze(0).squeeze(0)
        
        self.pheromone.clamp_(0.0,1.0)
        
    def _cluster_metric(self):
        # Average local same-type density for all candies
        totals = []
        for t in range(1, self.candy_types + 1):
            mask = (self.items == t).float()
            if mask.sum() == 0:
                continue
            density = F.conv2d(mask.unsqueeze(0).unsqueeze(0), self.diffusion_kernel, padding=1).squeeze(0).squeeze(0)
            totals.append(density[mask == 1].mean() / self.diffusion_kernel.sum())
        if not totals:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(totals).mean()

    def _sample_potential(self, positions: torch.Tensor, carrying: torch.Tensor) -> torch.Tensor:
        """Sample pheromone per agent, type-specific when carrying, otherwise max-field."""
        max_field = self.pheromone.max(dim=0).values
        vals = []
        for i in range(self.num_agents):
            carry = carrying[i].item()
            field = self.pheromone[carry - 1] if carry > 0 else max_field
            vals.append(field[positions[i, 0], positions[i, 1]])
        return torch.stack(vals).unsqueeze(-1)

    def _count_neighbors(self, mask: torch.Tensor, x: int, y: int) -> float:
        """Count neighbors in a 3x3 window around (x, y) in `mask`."""
        x_min = max(0, x - 1)
        x_max = min(self.grid_size, x + 2)
        y_min = max(0, y - 1)
        y_max = min(self.grid_size, y + 2)
        return mask[x_min:x_max, y_min:y_max].sum().item()
    
    def _reset(self, tensordict = None) -> TensorDict:
        total_cells = self.grid_size * self.grid_size
        total_needed = self.num_agents + self.wall_count + self.candy_types * self.candies_per_type
        if total_needed > total_cells:
            raise ValueError("Grid size too small for the number of agents, walls, and candies.")
        
        # Creating the random starting indexes
        self.current_step = 0
        
        indices = torch.randperm(total_cells, device=self.device)
        split = 0
        self.agent_positions = indices[split:split + self.num_agents]
        split += self.num_agents
        wall_indexes = indices[split:split + self.wall_count]
        split += self.wall_count
        candy_indexes = indices[split:split + self.candy_types * self.candies_per_type]
        
        # Casting to 2d
        to_2d = lambda idx: torch.stack((idx // self.grid_size, idx % self.grid_size), dim=1)
        self.agent_positions = to_2d(self.agent_positions)
        self.wall_positions = to_2d(wall_indexes)
        
        self.walls = torch.zeros((self.grid_size, self.grid_size), device=self.device)
        self.walls[self.wall_positions[:, 0], self.wall_positions[:, 1]] = 1
        
        self.items = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long, device=self.device)
        candy_2d = to_2d(candy_indexes)
        type_assignments = torch.arange(self.candy_types, device=self.device).repeat_interleave(self.candies_per_type) + 1
        self.items[candy_2d[:, 0], candy_2d[:, 1]] = type_assignments

        self.carrying = torch.zeros((self.num_agents, 1), dtype=torch.long, device=self.device)
        self.carry_baseline_density = torch.zeros((self.num_agents, 1), dtype=torch.float, device=self.device)
    
        # Persistent pheromone state per candy type
        self.pheromone = torch.zeros((self.candy_types, self.grid_size, self.grid_size), device=self.device)
        self._update_pheromones()
        
        self.metric_ema = self._cluster_metric()
        self.prev_potential = self._sample_potential(self.agent_positions, self.carrying)
        
        done = torch.zeros((self.num_agents, 1), dtype=torch.bool, device=self.device)
        terminated = torch.zeros((self.num_agents, 1), dtype=torch.bool, device=self.device)
        truncated = torch.zeros((self.num_agents, 1), dtype=torch.bool, device=self.device)
        
        return self._pack_output(done=done, terminated=terminated, truncated=truncated)
    
    def _step(self, tensordict: TensorDict) -> TensorDict:
        actions = tensordict["agents", "action"].squeeze(-1)
        rewards = torch.zeros((self.num_agents, 1), dtype=torch.float, device=self.device)
        moves = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], device=self.device)
        
        prev_potential = self.prev_potential.clone()
        drop_rewards = torch.zeros_like(rewards)
        pickup_rewards = torch.zeros_like(rewards)
        prev_positions = self.agent_positions.clone()
        moved_mask = torch.zeros((self.num_agents, 1), dtype=torch.bool, device=self.device)
        
        for i in range(self.num_agents):
            action = actions[i].item()
            x,y = self.agent_positions[i]
            carrying_before = self.carrying[i].item()
            
            if action < 4:
                dx, dy = moves[action]
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and self.walls[nx, ny] == 0:
                    self.agent_positions[i] = torch.tensor([nx, ny], device=self.device)
                    moved_mask[i] = True
                    x, y = nx, ny
            
            if action == 4:
                if self.carrying[i] == 0 and self.items[x, y] > 0:
                    self.carrying[i] = self.items[x, y]
                    self.items[x, y] = 0
                    pickup_rewards[i] = self.pickup_reward
                elif self.carrying[i] > 0 and self.items[x, y] == 0:
                    candy_type = self.carrying[i].item()
                    type_mask = (self.items == candy_type).float()
                    neighbors = self._count_neighbors(type_mask, x, y)
                    drop_rewards[i] = self.drop_base_reward + neighbors * self.drop_neighbor_multiplier
                    self.items[x, y] = candy_type
                    self.carrying[i] = 0
                    
        # Agent-dependent emissions: only when moving and carrying a candy (refractory when stationary)
        agent_emit = None
        if self.agent_emit_strength > 0.0:
            agent_emit = torch.zeros_like(self.pheromone)
            for i in range(self.num_agents):
                if moved_mask[i].item() and self.carrying[i].item() > 0:
                    t_idx = self.carrying[i].item() - 1
                    pos = self.agent_positions[i]
                    agent_emit[t_idx, pos[0], pos[1]] += self.agent_emit_strength

        self._update_pheromones(agent_emit)

        # Potential-based gradient reward with per-step penalty
        current_potential = self._sample_potential(self.agent_positions, self.carrying)
        gradient_reward = self.gradient_gamma * current_potential - prev_potential
        rewards += gradient_reward
        rewards += self.time_penalty
        rewards += drop_rewards
        rewards += pickup_rewards
        self.prev_potential = current_potential.detach()

        done = torch.zeros((self.num_agents, 1), dtype=torch.bool, device=self.device)
        terminated = torch.zeros_like(done)
        truncated = torch.zeros_like(done)
        self.current_step += 1
        # Hard stop on max_steps to avoid relying on StepCounter transform
        if self.current_step >= self.max_steps:
            done.fill_(True)
            truncated.fill_(True)
        return self._pack_output(rewards=rewards, done=done, terminated=terminated, truncated=truncated)
        
    
    def _set_seed(self, seed: int) -> None:
        pass
    
    
# test the environment
if __name__ == "__main__":
    env = PheromoneEnv(candy_types=1, grid_size=5, agent_view = 2, num_agents = 1)
    print(env.observation_spec)
    td = env._reset()
    # print(env.walls)
    # print(env.agent_positions)
    # print(env.items)
    # print(env.pheromone[0])
    print(td["global"])
    print(env.metric_ema)