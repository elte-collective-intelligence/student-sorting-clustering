import torch

class AntWorld:
    def __init__(self, agent_num = 1, grid_size=10, device="cpu"):
        self.agent_num = agent_num
        self.grid_size = grid_size
        self.device = device
        
    def generate_tile(self, type, amount, grid) -> None:
        # Randomly generate a tile with walls and items
        for _ in range(amount):
            x, y = torch.randint(0, self.grid_size, (2,))
            grid[x, y] = type
        
    def reset(self):
        # 1. The Map: 0=Empty, 1=ItemA, 2=ItemB, 3=ItemC -1=Wall
        grid = torch.zeros((self.grid_size, self.grid_size), dtype=torch.long, device=self.device)
        
        # Randomly create walls -> 10% of the grid
        self.generate_tile(-1, self.grid_size, grid)
        
        # Randomly create item As Bs and Cs -> 5% of the grid
        self.generate_tile(1, self.grid_size // 2, grid)
        self.generate_tile(2, self.grid_size // 2, grid)
        self.generate_tile(3, self.grid_size // 2, grid)
    
        # 2. The Agent: Position (x, y)
        # agent number determined by self.agent_num
        agents = torch.randint(0, self.grid_size, (self.agent_num, 2), dtype=torch.long, device=self.device)
        # 3. Carrying Status: 0=Empty, 1=ItemA, 2=ItemB
        carrying = torch.zeros((self.agent_num,1), dtype=torch.long, device=self.device)
        
        return grid, agents, carrying
    
    def step(self, grid, agents, carrying, actions):
        # actions: Tensor of shape (agent_num, 1)
        
        # Clone data so we don't mess up the history
        new_grid = grid.clone()
        new_agents = agents.clone()
        new_carrying = carrying.clone()
        
        # Define moves: Up, Down, Left, Right, Stay
        moves = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0], [0, 0]], device=self.device)

        # Loop through each agent
        for i in range(self.agent_num):
            action_code = actions[i].item() # Get action for Agent i
            x, y = new_agents[i, 0].item(), new_agents[i, 1].item()
            
            # --- MOVEMENT (0-3) ---
            if action_code < 4:
                # Calculate simple target coordinates manually to avoid tensor shape issues
                target_x, target_y = x, y
                if action_code == 0: target_y -= 1   # Up
                elif action_code == 1: target_y += 1 # Down
                elif action_code == 2: target_x -= 1 # Left
                elif action_code == 3: target_x += 1 # Right

                # Check Bounds
                if 0 <= target_x < self.grid_size and 0 <= target_y < self.grid_size:
                    # Check Walls
                    if new_grid[target_x, target_y] != -1:
                        # OPTIONAL: Check collision with other agents
                        # For now, we allow them to overlap (Ghost Mode).
                        # To block overlap, check if [target_x, target_y] is in new_agents
                        
                        new_agents[i, 0] = target_x
                        new_agents[i, 1] = target_y

            # --- INTERACTION (4) ---
            elif action_code == 4:
                cell_value = new_grid[x, y].item()
                is_carrying = new_carrying[i].item() # Check Agent i's hands
                
                # Pick Up
                if is_carrying == 0 and cell_value > 0:
                    new_carrying[i] = cell_value
                    new_grid[x, y] = 0
                
                # Drop
                elif is_carrying > 0 and cell_value == 0:
                    new_grid[x, y] = is_carrying
                    new_carrying[i] = 0
                    
        return new_grid, new_agents, new_carrying