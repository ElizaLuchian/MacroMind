import pandas as pd

from src.agents import MomentumAgent, RandomAgent
from src.simulator import simulate_market


def test_simulator_runs_with_basic_agents():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5),
            "close": [100, 101, 102, 101, 99],
            "cluster_id": [0, 1, 0, 1, 0],
            "volume": [1000] * 5,
        }
    )
    agents = [RandomAgent(name="random", seed=1), MomentumAgent(name="momentum")]
    result = simulate_market(data, agents, alpha=0.05, noise_std=0.0)
    assert len(result.prices) == len(data)
    assert result.to_frame().shape[0] == len(data)


