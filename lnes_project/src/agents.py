"""Agent definitions for the MacroMind latent news simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

import numpy as np

Action = str
MarketState = Dict[str, float]


def _decision_from_sentiment(score: float, threshold: float = 0.15) -> Action:
    if score > threshold:
        return "buy"
    if score < -threshold:
        return "sell"
    return "hold"


@dataclass
class BaseAgent(ABC):
    name: str
    seed: Optional[int] = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    @abstractmethod
    def decide(self, market_state: MarketState, cluster_id: Optional[int]) -> Action:
        """Return 'buy', 'sell', or 'hold'."""


class RandomAgent(BaseAgent):
    def decide(self, market_state: MarketState, cluster_id: Optional[int]) -> Action:  # noqa: D401
        return self.rng.choice(["buy", "sell", "hold"])


class MomentumAgent(BaseAgent):
    def decide(self, market_state: MarketState, cluster_id: Optional[int]) -> Action:
        price = market_state.get("price")
        prev_price = market_state.get("prev_price")
        if price is None or prev_price is None:
            return "hold"
        return "buy" if price > prev_price else "sell"


class ContrarianAgent(BaseAgent):
    def decide(self, market_state: MarketState, cluster_id: Optional[int]) -> Action:
        price = market_state.get("price")
        prev_price = market_state.get("prev_price")
        if price is None or prev_price is None:
            return "hold"
        return "sell" if price > prev_price else "buy"


@dataclass
class NewsReactiveAgent(BaseAgent):
    cluster_sentiment: Mapping[int, float] = field(default_factory=dict)
    neutral_action: Action = "hold"

    def decide(self, market_state: MarketState, cluster_id: Optional[int]) -> Action:
        if cluster_id is None:
            return self.neutral_action
        score = self.cluster_sentiment.get(cluster_id, 0.0)
        return _decision_from_sentiment(score)

