import torch
import torch.nn as nn

class LocalActorNet(nn.Module):
    def __init__(self, action_dim=5):
        super().__init__()
        
        # 1. The Eye (Visual Encoder)
        # Processes the 5x5 grid
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # Retain 5x5
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=0), # Down to 3x3
            nn.ReLU(),
            nn.Flatten() # 64 * 3 * 3 = 576
        )
        
        # 2. The Hand (State Encoder)
        # We assume input is a normalized scalar (0, 0.33, 0.66, 1.0)
        # We expand it to give it weight in the network
        self.state_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 3. The Brain (Fusion)
        # Concatenate Vision (576) + State (64)
        self.head = nn.Sequential(
            nn.Linear(576 + 64, 256),
            nn.LayerNorm(256), # Normalize for stability
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, obs):
        # 1. Handle TensorDict Shape (Batch, Agents, Obs)
        input_shape = obs.shape[:-1] 
        obs_flat = obs.view(-1, 26) 
        
        # 2. Split Vision and State
        grid_part = obs_flat[:, :25].view(-1, 1, 5, 5)
        state_part = obs_flat[:, 25:] # Shape (Batch, 1)
        
        # 3. Process
        visual_embed = self.cnn(grid_part)  # (Batch, 576)
        state_embed = self.state_net(state_part) # (Batch, 64)
        
        # 4. Fuse
        combined = torch.cat([visual_embed, state_embed], dim=1) # (Batch, 640)
        logits = self.head(combined)
        
        # 5. Restore Shape
        return logits.view(*input_shape, -1)

class LocalCriticNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. The Eye (Identical to Actor)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Flatten() # 576
        )
        
        # 2. The Hand (Identical to Actor)
        self.state_net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # 3. The Value Head (Converges to 1 value)
        self.head = nn.Sequential(
            nn.Linear(576 + 64, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1) # Output is Value (Scalar)
        )

    def forward(self, obs):
        input_shape = obs.shape[:-1]
        obs_flat = obs.view(-1, 26)
        
        grid_part = obs_flat[:, :25].view(-1, 1, 5, 5)
        state_part = obs_flat[:, 25:]
        
        visual_embed = self.cnn(grid_part)
        state_embed = self.state_net(state_part)
        
        combined = torch.cat([visual_embed, state_embed], dim=1)
        value = self.head(combined)
        
        return value.view(*input_shape, 1)