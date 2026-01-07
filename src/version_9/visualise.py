import curses
import torch
from tensordict import TensorDict
import sys
import os
# Join src to the file

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from envs.pheromone_env import PheromoneEnv


PHEROMONE_CHARS = " .:-=+*#%@"

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


def render_view(stdscr, obs: torch.Tensor, carrying: torch.Tensor, reward: float, step_count: int):
    """Render a single-agent local view.

    Channels layout: [walls][agents][candy_t][pheromone_t] for each candy type.
    We draw agent > wall > candy; pheromone is shown as intensity when empty.
    """
    stdscr.clear()
    view = obs.squeeze(0)  # (C, H, W)
    _, h, w = view.shape

    num_candy_types = (view.shape[0] - 2) // 2
    candy_ch_start = 2
    pher_ch_start = 3

    stdscr.addstr(0, 0, f"Step: {step_count} | Carrying: {int(carrying.item())} | Last reward: {reward:.4f}")
    stdscr.addstr(1, 0, "-" * (w * 2 + 3))

    for r in range(h):
        line = " "
        for c in range(w):
            char = " "
            wall = view[0, r, c] > 0.5
            agent = view[1, r, c] > 0.5
            if agent:
                char = "A"
            elif wall:
                char = "-1"
            else:
                candy_type = 0
                for t in range(num_candy_types):
                    if view[candy_ch_start + t * 2, r, c] > 0.5:
                        candy_type = t + 1
                        break
                if candy_type:
                    char = str(candy_type)
                else:
                    pher_vals = []
                    for t in range(num_candy_types):
                        pher_vals.append(view[pher_ch_start + t * 2, r, c].item())
                    if pher_vals:
                        v = max(pher_vals)
                        idx = min(int(v * (len(PHEROMONE_CHARS) - 1)), len(PHEROMONE_CHARS) - 1)
                        char = PHEROMONE_CHARS[idx]
                    else:
                        char = "."
            line += char + " "
        stdscr.addstr(r + 2, 0, line)

    stdscr.addstr(h + 3, 0, "Controls: arrows=move, space=pick/drop, q=quit")
    stdscr.refresh()


def main(stdscr):
    device = "cpu"
    env = PheromoneEnv(num_agents=1, grid_size=10, agent_view=5, candy_types=2, max_steps=1e100, device=device)
    td = env.reset()

    step_count = 0
    last_reward = 0.0

    curses.curs_set(0)
    stdscr.nodelay(False)

    while True:
        obs = td["agents", "observation"]
        carrying = td["agents", "carrying"]
        render_view(stdscr, obs, carrying, last_reward, step_count)

        key = stdscr.getch()
        action_idx = get_action(key)
        if action_idx == -1:
            break
        if action_idx is None:
            continue

        action_td = TensorDict({
            "agents": TensorDict({"action": torch.tensor([[action_idx]], device=device)}, batch_size=[1])
        }, batch_size=[])

        td = env.step(action_td)
        last_reward = td["next", "agents", "reward"].item()
        step_count += 1
        td = td["next"]


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except Exception as e:
        print(f"Crash: {e}")
        import os
        if os.name == "nt":
            print("On Windows, install 'windows-curses' (pip install windows-curses) to enable curses UI.")
