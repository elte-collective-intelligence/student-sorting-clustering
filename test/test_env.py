import sys
import os
import torch
import src.envs
assert src.envs is not None
# Add the project root (marl-task) to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --- Now your imports will work ---
from src.envs.ant_env import AntSortingEnv
from torchrl.envs.utils import check_env_specs

def test_specs():
    print("1. Initializing Env...")
    env = AntSortingEnv(num_agents=3, grid_size=20, device=device)
    print("   [SUCCESS] Env initialized.")
    
    print("2. Checking TorchRL Specs Compliance...")
    # This built-in function checks if your specs match your data output
    check_env_specs(env)
    print("   [SUCCESS] Specs are valid.")

    print("3. Testing Rollout (Simulation)...")
    # This runs the environment for 5 steps with random actions
    rollout = env.rollout(5)
    
    print(f"   Rollout shape: {rollout.shape}")
    print(f"   Final Agent Position: {rollout['next', 'agents', 'observation'][-1, :, :2]}")
    print("   [SUCCESS] Rollout completed.")

   
if __name__ == "__main__":
    test_specs()