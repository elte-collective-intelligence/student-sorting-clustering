from src.utils.metrics import episode_return

def test_episode_return_correctness():
    assert abs(episode_return([1, 0.5, -0.25]) - 1.25) < 1e-9
