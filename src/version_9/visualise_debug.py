import curses
import torch
from tensordict import TensorDict
import sys
import os
# Join src to the file

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from envs.pheromone_env import PheromoneEnv


def get_action(key: int):
    if key == curses.KEY_UP:
        return 0
    if key == curses.KEY_DOWN:
        return 1
    if key == curses.KEY_LEFT:
        return 2
    if key == curses.KEY_RIGHT:
        return 3
    if key == ord(" "):
        return 4
    if key == ord("q"):
        return -1
    return None


def enforce_single_candy(env: PheromoneEnv) -> TensorDict:
    """Reset to a single candy and refresh pheromones for easier debugging."""
    env.reset()
    env.items.zero_()

    free_mask = env.walls == 0
    free_mask[env.agent_positions[:, 0], env.agent_positions[:, 1]] = False
    free_cells = free_mask.nonzero(as_tuple=False)
    if free_cells.numel() == 0:
        raise RuntimeError("No free cells to place a candy.")

    idx = free_cells[torch.randint(0, free_cells.size(0), (1,), device=env.device)].squeeze(0)
    env.items[idx[0], idx[1]] = 1

    env.carrying.zero_()
    env.pheromone.zero_()
    env._update_pheromones()
    env.metric_ema = env._cluster_metric()

    done = torch.zeros((env.num_agents, 1), dtype=torch.bool, device=env.device)
    terminated = torch.zeros_like(done)
    truncated = torch.zeros_like(done)
    return env._pack_output(done=done, terminated=terminated, truncated=truncated)


def render_debug(stdscr, td: TensorDict, last_reward: float, metric_value: float):
    obs = td["agents", "observation"][0]
    carrying = int(td["agents", "carrying"][0].item())
    global_grid = td["global"]
    step_count = int(td["step_count"].item())

    h = obs.shape[1]
    w = obs.shape[2]
    num_candy_types = (obs.shape[0] - 2) // 2

    stdscr.clear()
    stdscr.addstr(0, 0, f"Step: {step_count} | Carrying: {carrying} | Reward: {last_reward:.4f} | metric_ema: {metric_value:.4f}")
    stdscr.addstr(1, 0, "Agent view (no pheromone):")

    for r in range(h):
        line = ""
        for c in range(w):
            wall = obs[0, r, c] > 0.5
            agent = obs[1, r, c] > 0.5
            candy_type = 0
            if not agent and not wall:
                for t in range(num_candy_types):
                    if obs[2 + t * 2, r, c] > 0.5:
                        candy_type = t + 1
                        break
            if agent:
                ch = "A"
            elif wall:
                ch = "#"
            elif candy_type:
                ch = str(candy_type)
            else:
                ch = "."
            line += ch + " "
        stdscr.addstr(r + 2, 0, line)

    col_offset = w * 2 + 6
    stdscr.addstr(1, col_offset, "Pheromone field (numeric):")
    pher_grid = global_grid[3]
    for r in range(pher_grid.shape[0]):
        line = " ".join(f"{pher_grid[r, c]:.2f}" for c in range(pher_grid.shape[1]))
        stdscr.addstr(r + 2, col_offset, line)

    done_flag = bool(td.get("done", torch.tensor([False])).any().item())
    truncated_flag = bool(td.get("truncated", torch.tensor([False])).any().item())
    status_line = "done" if done_flag else "running"
    if truncated_flag:
        status_line += " (truncated)"

    footer_row = max(h, pher_grid.shape[0]) + 3
    stdscr.addstr(footer_row, 0, f"Status: {status_line}")
    stdscr.addstr(footer_row + 1, 0, "Controls: arrows=move, space=pick/drop, q=quit")
    stdscr.refresh()


def main(stdscr):
    device = "cpu"
    env = PheromoneEnv(num_agents=1, grid_size=10, agent_view=5, candy_types=1, max_steps=1e100, device=device)
    td = enforce_single_candy(env)
    last_reward = 0.0

    curses.curs_set(0)
    stdscr.nodelay(False)

    while True:
        metric_value = float(env.metric_ema)
        render_debug(stdscr, td, last_reward, metric_value)

        key = stdscr.getch()
        action_idx = get_action(key)
        if action_idx == -1:
            break
        if action_idx is None:
            continue

        action_td = TensorDict({
            "agents": TensorDict({"action": torch.tensor([[action_idx]], device=device)}, batch_size=[1])
        }, batch_size=[])

        step_td = env.step(action_td)
        env.metric_ema = env._cluster_metric()
        last_reward = float(step_td["next", "agents", "reward"].item())
        td = step_td["next"]

        if td.get("done", torch.tensor([False], device=device)).any():
            render_debug(stdscr, td, last_reward, float(env.metric_ema))
            stdscr.getch()
            break


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except Exception as e:
        print(f"Crash: {e}")
        import os
        if os.name == "nt":
            print("On Windows, install 'windows-curses' (pip install windows-curses) to enable curses UI.")
