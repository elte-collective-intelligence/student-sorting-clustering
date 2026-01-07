import pygame
import time
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Now your imports will work ---
from src.envs.mechanics import AntWorld



class AntRenderer:
    def __init__(self, grid_size=15, cell_size=60):
        pygame.init()
        self.width = grid_size * cell_size
        self.height = grid_size * cell_size
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.grid_size = grid_size
        pygame.display.set_caption("Ant Sorting")

    def render(self, grid, agents, carrying): 
        self.screen.fill((255, 255, 255))
        
        # FIX: Get dynamic size from the tensor itself
        rows, cols = grid.shape 
        
        # 1. Draw Grid Lines
        # We need to ensure lines match the actual grid size
        current_width = rows * 60 # 60 is cell_size
        current_height = cols * 60
        
        for x in range(0, current_width, 60):
            pygame.draw.line(self.screen, (200, 200, 200), (x, 0), (x, current_height))
        for y in range(0, current_height, 60):
            pygame.draw.line(self.screen, (200, 200, 200), (0, y), (current_width, y))

        # 2. Draw Items and Walls
        # FIX: Loop using the tensor dimensions, not self.grid_size
        for x in range(rows): 
            for y in range(cols):
                value = grid[x, y].item()
                
                rect = pygame.Rect(x*60 + 10, y*60 + 10, 40, 40)
                
                if value == -1: # Wall
                    pygame.draw.rect(self.screen, (0, 0, 0), rect) # Black
                elif value == 1: # Item A
                    pygame.draw.rect(self.screen, (200, 50, 50), rect) # Red
                elif value == 2: # Item B
                    pygame.draw.rect(self.screen, (50, 50, 200), rect) # Blue
                elif value == 3: # Item C (New!)
                    pygame.draw.rect(self.screen, (50, 200, 50), rect) # Greenish

        # 3. Draw Agents
        # We iterate over the number of agents in the tensor
        for i in range(len(agents)):
            ax, ay = int(agents[i, 0].item()), int(agents[i, 1].item())
            center_x, center_y = ax * 60 + 30, ay * 60 + 30
            
            # Agent Body (Green)
            pygame.draw.circle(self.screen, (50, 200, 50), (center_x, center_y), 20)
            
            # Carrying Indicator
            if carrying[i].item() > 0:
                pygame.draw.circle(self.screen, (0, 0, 0), (center_x, center_y), 10)

        pygame.display.flip()
        
def play():
    # 1. Initialize with 2 Agents
    renderer = AntRenderer(grid_size=10) # <--- Change to 10
    world = AntWorld(agent_num=2, grid_size=10) # <--- Change to 2
    
    grid, agents, carrying = world.reset()
    
    running = True
    while running:
        action_code = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP: action_code = 0
                elif event.key == pygame.K_DOWN: action_code = 1
                elif event.key == pygame.K_LEFT: action_code = 2
                elif event.key == pygame.K_RIGHT: action_code = 3
                elif event.key == pygame.K_SPACE: action_code = 4

        if action_code is not None:
            # Create actions for BOTH agents
            # Agent 1 obeys Keyboard. Agent 2 moves randomly.
            
            # Shape: (2, 1)
            actions = torch.zeros((2, 1), dtype=torch.long)
            actions[0] = action_code
            actions[1] = torch.randint(0, 5, (1,)).item() # Random friend
            
            grid, agents, carrying = world.step(grid, agents, carrying, actions)

        renderer.render(grid, agents, carrying)

    renderer.close()

if __name__ == "__main__":
    play()