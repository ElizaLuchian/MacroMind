"""Agent-based market simulator."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

from .agents import Action, BaseAgent

logger = logging.getLogger(__name__)

ACTION_IMPACT = {"buy": 1.0, "sell": -1.0, "hold": 0.0}


@dataclass
class SimulationResult:
    dates: List[pd.Timestamp]
    prices: List[float]
    action_log: Dict[str, List[Action]]
    order_flow: List[float]

    def to_frame(self) -> pd.DataFrame:
        data = {
            "date": self.dates,
            "simulated_close": self.prices,
            "order_flow": self.order_flow,
        }
        df = pd.DataFrame(data)
        for agent_name, actions in self.action_log.items():
            df[f"action_{agent_name}"] = actions
        return df


def simulate_market(
    merged_df: pd.DataFrame,
    agents: Iterable[BaseAgent],
    alpha: float = 0.01,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> SimulationResult:
    """Simulate the market reaction to latent news events."""
    if merged_df.empty:
        raise ValueError("Merged dataframe must not be empty.")
    df = merged_df.copy().reset_index(drop=True)
    if "close" not in df.columns:
        raise ValueError("Merged dataframe must contain a 'close' column.")

    rng = np.random.default_rng(seed)
    price_series = [float(df.at[0, "close"])]
    dates = [pd.to_datetime(df.at[0, "date"])]
    action_log: Dict[str, List[Action]] = {agent.name: [] for agent in agents}
    order_flow: List[float] = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        current_price = float(price_series[-1])
        prev_price = float(price_series[-2]) if len(price_series) > 1 else None
        market_state = {
            "price": current_price,
            "prev_price": prev_price,
            "volume": float(row.get("volume", 0.0)),
        }
        cluster_id = int(row["cluster_id"]) if "cluster_id" in row and not pd.isna(row["cluster_id"]) else None
        news_text = str(row.get("combined_text", "")) if "combined_text" in row else None

        net_flow = 0.0
        for agent in agents:
            # Try passing news_text for AI agents, fall back to old signature
            try:
                decision = agent.decide(market_state, cluster_id, news_text)
            except TypeError:
                decision = agent.decide(market_state, cluster_id)
            action_log[agent.name].append(decision)
            net_flow += ACTION_IMPACT.get(decision, 0.0)

        noise = rng.normal(0, noise_std) if noise_std > 0 else 0.0
        next_price = current_price + alpha * net_flow + noise
        price_series.append(max(next_price, 0.0))
        order_flow.append(net_flow)
        if idx + 1 < len(df):
            dates.append(pd.to_datetime(df.at[idx + 1, "date"]))

    return SimulationResult(dates=dates, prices=price_series[1:], action_log=action_log, order_flow=order_flow)


