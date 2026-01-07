import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Composite, Bounded, Unbounded
from src.envs.mechanics import AntWorld

class AntSortingEnv(EnvBase):
    def __init__(self, num_agents=1, grid_size=20, device="cpu"):
        super().__init__(device=device, batch_size=[])
        
        self.world = AntWorld(agent_num=num_agents, grid_size=grid_size, device=device)
        self.num_agents = num_agents
        self.grid_size = grid_size
        
        # Specs
        self.action_spec = Composite({
            "agents": Composite({
                "action": Bounded(low=0, high=4, shape=(self.num_agents, 1), dtype=torch.long, device=device)
            }, shape=(self.num_agents,))
        })

        # OBS: 5x5 grid (25 values) + 1 carrying state = 26 inputs
        self.observation_spec = Composite({
            "agents": Composite({
                "observation": Bounded(low=-1, high=10, shape=(self.num_agents, 26), dtype=torch.float32, device=device),
            }, shape=(self.num_agents,))
        })

        self.reward_spec = Composite({
            "agents": Composite({
                "reward": Unbounded(shape=(self.num_agents, 1), device=device)
            }, shape=(self.num_agents,))
        })
        
        self.done_spec = Composite({
            "agents": Composite({
                "done": Bounded(low=0, high=1, shape=(self.num_agents, 1), dtype=torch.bool, device=device),
                "terminated": Bounded(low=0, high=1, shape=(self.num_agents, 1), dtype=torch.bool, device=device),
            }, shape=(self.num_agents,)),
            "done": Bounded(low=0, high=1, shape=(1,), dtype=torch.bool, device=device),
            "terminated": Bounded(low=0, high=1, shape=(1,), dtype=torch.bool, device=device),
        })

    def _reset(self, tensordict):
        grid, agents, carrying = self.world.reset()
        out = self._encode_obs(grid, agents, carrying)
        return out

    def _step(self, tensordict):
        actions = tensordict["agents", "action"]
        grid = tensordict["global_state_hidden"] # Retain full grid for physics
        prev_obs = tensordict["agents", "observation"]
        
        # Extract Agent Pos from World (we need to track it manually or store it)
        # Note: In previous version, we stored pos in obs. Now obs is just 5x5 view.
        # We need to pass the real agent positions through the tensordict or keep them in the class.
        # BETTER APPROACH: Store absolute agent positions in the tensordict hidden state.
        agents_pos = tensordict["agents_pos_hidden"]
        carrying = tensordict["carrying_hidden"]
        
        new_grid, new_agents, new_carrying = self.world.step(
            grid, agents_pos, carrying, actions
        )
        
        out = self._encode_obs(new_grid, new_agents, new_carrying)
        
        reward = self._get_reward(grid, new_grid, carrying, new_carrying, agents_pos, new_agents)
        
        done = torch.zeros((1,), dtype=torch.bool, device=self.device)
        done_agents = done.expand(self.num_agents, 1)

        out["agents"].set("reward", reward)
        out["agents"].set("done", done_agents)
        out["agents"].set("terminated", done_agents)
        out.set("done", done)
        out.set("terminated", done)
        
        return out

    def _encode_obs(self, grid, agents, carrying):
        # 1. Pad the grid with Walls (-1) so we can slice 5x5 edges
        # Pad format: (Left, Right, Top, Bottom)
        # We need 2 cells padding on all sides for a 5x5 window centered on agent
        padded_grid = F.pad(grid, (2, 2, 2, 2), value=-1)
        
        local_views = []
        
        for i in range(self.num_agents):
            ax, ay = agents[i, 0].long(), agents[i, 1].long()
            # Because we padded by 2, the agent's (0,0) is now at (2,2)
            # We want x-2 to x+3
            # Remember: in tensor, x is row, y is col
            window = padded_grid[ax:ax+5, ay:ay+5]
            local_views.append(window.flatten())
            
        # Shape: (Num_Agents, 25)
        visual_obs = torch.stack(local_views).float() / 3.0 # Normalize grid
        
        # Carrying State (Num_Agents, 1)
        state_obs = carrying.float() / 3.0 
        
        # Combine: (Num_Agents, 26)
        obs = torch.cat([visual_obs, state_obs], dim=1)

        return TensorDict({
            "agents": TensorDict({
                "observation": obs
            }, batch_size=[self.num_agents], device=self.device),
            "global_state_hidden": grid,   # Keep true state hidden from network
            "agents_pos_hidden": agents,   # Keep pos hidden
            "carrying_hidden": carrying
        }, batch_size=[], device=self.device)
        
    def _get_reward(self, old_grid, new_grid, old_carry, new_carry, old_pos, new_pos):
        reward = torch.zeros((self.num_agents, 1), device=self.device)
        
        # 1. MOVEMENT PENALTY (Efficiency Bias)
        # Small penalty for every step to encourage efficiency.
        # Strict penalty for hitting walls/staying still.
        dists = (old_pos - new_pos).abs().sum(dim=1, keepdim=True)
        reward[dists > 0] -= 0.005  
        reward[dists == 0] -= 0.02 

        # 2. PICK UP LOGIC (The Cleaner)
        # We detected a pickup if old_carry was 0 and new_carry > 0
        picked_up = (old_carry == 0) & (new_carry > 0)
        
        if picked_up.any():
            indices = torch.nonzero(picked_up.squeeze()).flatten()
            for i in indices:
                # Get the type of the item that was just picked up
                # It is now in new_carry[i], but it WAS at old_grid[ax, ay]
                ax, ay = old_pos[i, 0].long(), old_pos[i, 1].long()
                item_type = new_carry[i].item()
                
                # Analyze the spot we just took it FROM (using old_grid)
                n_same, n_diff = self._get_neighbor_stats(old_grid, ax, ay, item_type)
                
                # REWARD FORMULA:
                # Base reward (0.5) for working.
                # Bonus (+0.5 per neighbor) if we removed it from "Enemies" (cleaning).
                # Penalty (-1.0 per neighbor) if we broke a cluster (bad robot!).
                r_val = 0.5 + (n_diff * 0.5) - (n_same * 1.0)
                reward[i] += r_val

        # 3. DROP LOGIC (The Builder)
        # We detected a drop if old_carry > 0 and new_carry == 0
        dropped = (old_carry > 0) & (new_carry == 0)
        
        if dropped.any():
            indices = torch.nonzero(dropped.squeeze()).flatten()
            for i in indices:
                ax, ay = old_pos[i, 0].long(), old_pos[i, 1].long()
                item_type = old_carry[i].item()
                
                # Analyze the spot we just dropped INTO (using new_grid)
                n_same, n_diff = self._get_neighbor_stats(new_grid, ax, ay, item_type)
                
                # REWARD FORMULA:
                # Small base (0.2) so they don't hold items forever.
                # Massive Bonus (+1.5) per friend. This drives clustering.
                # Massive Penalty (-2.0) per enemy. This prevents mixing.
                r_val = 0.2 + (n_same * 1.5) - (n_diff * 2.0)
                reward[i] += r_val

        return reward

    def _get_neighbor_stats(self, grid, x, y, item_type):
        """
        Counts neighbors of the same type vs different type.
        """
        # Define 3x3 window limits
        x_min, x_max = max(0, x-1), min(self.grid_size, x+2)
        y_min, y_max = max(0, y-1), min(self.grid_size, y+2)
        
        patch = grid[x_min:x_max, y_min:y_max]
        
        # Count Same (subtract 1 to exclude the agent's current cell if it counts itself)
        # However, for Pickup, the item IS in the grid. For Drop, it IS in the grid.
        # So we count total occurrences in patch and subtract 1 (the item itself).
        n_same = (patch == item_type).sum().item() - 1
        
        # Count Diff (Items 1, 2, 3 that are NOT item_type)
        # We ignore 0 (empty) and -1 (walls) for "difference" penalties usually,
        # unless you want them to cluster against walls. Let's strictly count items.
        is_item = (patch > 0) # 1, 2, or 3
        is_diff = is_item & (patch != item_type)
        n_diff = is_diff.sum().item()
        
        # Safety clip (in case of boundary weirdness)
        n_same = max(0, n_same)
        
        return n_same, n_diff

    # def _get_reward(self, old_grid, new_grid, old_carry, new_carry, old_pos, new_pos):
    #     reward = torch.zeros((self.num_agents, 1), device=self.device)
        
    #     # 1. MOVEMENT & WALLS
    #     dists = (old_pos - new_pos).abs().sum(dim=1, keepdim=True) # Shape (N, 1)
    #     reward[dists > 0] += 0.01      
    #     reward[dists == 0] -= 0.02     
        
    #     # 2. THE "SCENT" REWARD (Exploit Fixed)
    #     # We assume 'dists' aligns with agent index i
    #     for i in range(self.num_agents):
    #         # CONDITION ADDED: 'and dists[i] > 0'
    #         # You only get credit for smelling candy if you are actively searching (moving).
    #         if old_carry[i] == 0 and dists[i] > 0: 
    #             ax, ay = old_pos[i, 0].long(), old_pos[i, 1].long()
                
    #             x_min = max(0, ax - 2)
    #             x_max = min(self.grid_size, ax + 3)
    #             y_min = max(0, ay - 2)
    #             y_max = min(self.grid_size, ay + 3)
                
    #             visible_patch = old_grid[x_min:x_max, y_min:y_max]
                
    #             if (visible_patch > 0).any():
    #                 reward[i] += 0.05 # Reward for 'Finding' while moving

    #     # 3. PICK UP 
    #     picked_up = (old_carry == 0) & (new_carry > 0)
    #     if picked_up.any():
    #         # This is 2.0.
    #         # Scent+Move is 0.06. 
    #         # Picking up is 33x more valuable than just walking past it.
    #         # They will learn to pick it up.
    #         reward[picked_up] += 2.0 

    #     # 4. DROP 
    #     dropped = (old_carry > 0) & (new_carry == 0)
    #     if dropped.any():
    #         reward[dropped] += 0.5 

    #     return reward

    

    def _count_neighbors(self, grid, x, y, type_to_match):
        # Check 3x3 surrounding
        x_min, x_max = max(0, x-1), min(self.grid_size, x+2)
        y_min, y_max = max(0, y-1), min(self.grid_size, y+2)
        
        patch = grid[x_min:x_max, y_min:y_max]
        return (patch == type_to_match).sum().item()
    
    def _set_seed(self, seed):
        torch.manual_seed(seed)