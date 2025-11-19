"""Agent definitions for the MacroMind latent news simulation."""

from __future__ import annotations

import os
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


@dataclass
class FinBERTAgent(BaseAgent):
    """Agent using FinBERT sentiment analysis (100% FREE, runs locally)."""

    confidence_threshold: float = 0.7
    _pipeline: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "sentiment-analysis", model="ProsusAI/finbert", device=-1  # -1 = CPU
            )
        except ImportError:
            raise ImportError("Install transformers: pip install transformers torch")

    def decide(
        self, market_state: MarketState, cluster_id: Optional[int], news_text: Optional[str] = None
    ) -> Action:
        """Decide based on FinBERT sentiment of news text."""
        if news_text is None or not news_text.strip():
            return "hold"

        try:
            # FinBERT returns: positive, negative, or neutral
            result = self._pipeline(news_text[:512])[0]  # Truncate to max length
            label = result["label"].lower()
            score = result["score"]

            if score >= self.confidence_threshold:
                if label == "positive":
                    return "buy"
                elif label == "negative":
                    return "sell"
            return "hold"
        except Exception:
            return "hold"


@dataclass
class GroqAgent(BaseAgent):
    """Agent using Groq API with Llama models (FREE TIER available)."""

    api_key: Optional[str] = None
    model: str = "llama-3.1-8b-instant"
    temperature: float = 0.1
    _client: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        api_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Get free key at: https://console.groq.com/keys"
            )

        try:
            from groq import Groq

            self._client = Groq(api_key=api_key)
        except ImportError:
            raise ImportError("Install groq: pip install groq")

    def decide(
        self, market_state: MarketState, cluster_id: Optional[int], news_text: Optional[str] = None
    ) -> Action:
        """Decide using Groq LLM reasoning."""
        if news_text is None or not news_text.strip():
            return "hold"

        try:
            prompt = f"""You are a trading agent. Based on this news, should you buy, sell, or hold?

News: {news_text[:500]}

Respond with ONLY one word: buy, sell, or hold"""

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=self.temperature,
            )

            decision = response.choices[0].message.content.strip().lower()
            if decision in ["buy", "sell", "hold"]:
                return decision
            return "hold"
        except Exception:
            return "hold"

