"""Ensemble methods for combining multiple trading agents.

This module provides strategies for combining decisions from multiple agents
to create more robust trading systems.

Classes:
    VotingEnsemble: Majority voting ensemble
    WeightedEnsemble: Weighted combination based on past performance
    DynamicEnsemble: Adaptive agent selection
    MetaLearningEnsemble: Meta-agent that learns from base agents
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Callable
from collections import Counter
import warnings

import numpy as np
import pandas as pd


class VotingEnsemble:
    """Majority voting ensemble for combining agent decisions.
    
    Combines multiple agents by taking the most common decision.
    """
    
    def __init__(self, agents: List[Any], voting: str = "hard"):
        """Initialize voting ensemble.
        
        Args:
            agents: List of agent instances
            voting: "hard" (majority vote) or "soft" (weighted by confidence if available)
        """
        self.name = "VotingEnsemble"
        self.agents = agents
        self.voting = voting
        self.decision_history = []
    
    def decide(self, market_state: Dict[str, Any]) -> str:
        """Make ensemble decision by voting.
        
        Args:
            market_state: Current market state
            
        Returns:
            Most common decision among agents
        """
        decisions = []
        for agent in self.agents:
            try:
                decision = agent.decide(market_state)
                decisions.append(decision)
            except Exception as e:
                warnings.warn(f"Agent {getattr(agent, 'name', 'unknown')} failed: {e}")
        
        if not decisions:
            return "hold"
        
        # Hard voting: majority
        if self.voting == "hard":
            vote_counts = Counter(decisions)
            ensemble_decision = vote_counts.most_common(1)[0][0]
        else:
            # Soft voting would require confidence scores
            # For now, fall back to hard voting
            vote_counts = Counter(decisions)
            ensemble_decision = vote_counts.most_common(1)[0][0]
        
        self.decision_history.append({
            "individual_decisions": decisions,
            "ensemble_decision": ensemble_decision,
        })
        
        return ensemble_decision
    
    def get_decision_diversity(self) -> float:
        """Compute diversity of agent decisions.
        
        Returns:
            Diversity score (0 = all agree, 1 = maximum disagreement)
        """
        if not self.decision_history:
            return 0.0
        
        diversities = []
        for record in self.decision_history:
            decisions = record["individual_decisions"]
            if len(decisions) <= 1:
                diversities.append(0.0)
            else:
                # Diversity = 1 - (most common / total)
                vote_counts = Counter(decisions)
                most_common_count = vote_counts.most_common(1)[0][1]
                diversity = 1.0 - (most_common_count / len(decisions))
                diversities.append(diversity)
        
        return np.mean(diversities)


class WeightedEnsemble:
    """Weighted ensemble based on historical performance.
    
    Assigns weights to agents based on their past profitability or accuracy.
    """
    
    def __init__(
        self,
        agents: List[Any],
        initial_weights: Optional[List[float]] = None,
        update_weights: bool = True,
        lookback: int = 20,
    ):
        """Initialize weighted ensemble.
        
        Args:
            agents: List of agent instances
            initial_weights: Initial weights (uniform if None)
            update_weights: Whether to update weights based on performance
            lookback: Window size for computing recent performance
        """
        self.name = "WeightedEnsemble"
        self.agents = agents
        self.n_agents = len(agents)
        
        if initial_weights is None:
            self.weights = np.ones(self.n_agents) / self.n_agents
        else:
            self.weights = np.array(initial_weights)
            self.weights /= self.weights.sum()  # Normalize
        
        self.update_weights = update_weights
        self.lookback = lookback
        self.agent_performances = [[] for _ in range(self.n_agents)]
        self.decision_history = []
    
    def decide(self, market_state: Dict[str, Any]) -> str:
        """Make weighted ensemble decision.
        
        Args:
            market_state: Current market state
            
        Returns:
            Weighted decision
        """
        decisions = []
        for agent in self.agents:
            try:
                decision = agent.decide(market_state)
                decisions.append(decision)
            except Exception as e:
                warnings.warn(f"Agent failed: {e}")
                decisions.append("hold")
        
        # Weighted voting
        action_scores = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        for i, decision in enumerate(decisions):
            action_scores[decision] += self.weights[i]
        
        ensemble_decision = max(action_scores, key=action_scores.get)
        
        self.decision_history.append({
            "individual_decisions": decisions,
            "ensemble_decision": ensemble_decision,
            "weights": self.weights.copy(),
        })
        
        return ensemble_decision
    
    def update_performance(self, agent_returns: List[float]):
        """Update agent performance history and recompute weights.
        
        Args:
            agent_returns: List of returns for each agent in the last period
        """
        if len(agent_returns) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} returns, got {len(agent_returns)}")
        
        for i, ret in enumerate(agent_returns):
            self.agent_performances[i].append(ret)
        
        if self.update_weights:
            self._recompute_weights()
    
    def _recompute_weights(self):
        """Recompute weights based on recent performance."""
        recent_means = []
        for perf_history in self.agent_performances:
            if len(perf_history) == 0:
                recent_means.append(0.0)
            else:
                recent = perf_history[-self.lookback:]
                recent_means.append(np.mean(recent))
        
        # Softmax transformation
        recent_means = np.array(recent_means)
        # Shift to avoid numerical issues
        recent_means = recent_means - np.min(recent_means)
        exp_means = np.exp(recent_means)
        self.weights = exp_means / exp_means.sum()


class DynamicEnsemble:
    """Dynamic ensemble that selects best agent adaptively.
    
    Tracks recent performance and dynamically switches to the best-performing agent.
    """
    
    def __init__(
        self,
        agents: List[Any],
        lookback: int = 10,
        selection_method: str = "best",
    ):
        """Initialize dynamic ensemble.
        
        Args:
            agents: List of agent instances
            lookback: Window for computing recent performance
            selection_method: "best" (pick best agent) or "top_k" (ensemble of top k)
        """
        self.name = "DynamicEnsemble"
        self.agents = agents
        self.n_agents = len(agents)
        self.lookback = lookback
        self.selection_method = selection_method
        self.agent_performances = [[] for _ in range(self.n_agents)]
        self.selected_agent_idx = 0
        self.decision_history = []
    
    def decide(self, market_state: Dict[str, Any]) -> str:
        """Make decision using currently selected agent(s).
        
        Args:
            market_state: Current market state
            
        Returns:
            Decision from selected agent(s)
        """
        # Get all decisions
        decisions = []
        for agent in self.agents:
            try:
                decision = agent.decide(market_state)
                decisions.append(decision)
            except Exception as e:
                warnings.warn(f"Agent failed: {e}")
                decisions.append("hold")
        
        # Select based on method
        if self.selection_method == "best":
            ensemble_decision = decisions[self.selected_agent_idx]
        else:
            # Top-k voting (for now, just use top 1 = best)
            ensemble_decision = decisions[self.selected_agent_idx]
        
        self.decision_history.append({
            "individual_decisions": decisions,
            "ensemble_decision": ensemble_decision,
            "selected_agent": self.selected_agent_idx,
        })
        
        return ensemble_decision
    
    def update_performance(self, agent_returns: List[float]):
        """Update performance and reselect best agent.
        
        Args:
            agent_returns: List of returns for each agent
        """
        if len(agent_returns) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} returns, got {len(agent_returns)}")
        
        for i, ret in enumerate(agent_returns):
            self.agent_performances[i].append(ret)
        
        self._reselect_agent()
    
    def _reselect_agent(self):
        """Select agent with best recent performance."""
        recent_means = []
        for perf_history in self.agent_performances:
            if len(perf_history) == 0:
                recent_means.append(0.0)
            else:
                recent = perf_history[-self.lookback:]
                recent_means.append(np.mean(recent))
        
        self.selected_agent_idx = int(np.argmax(recent_means))


def create_ensemble_from_config(
    agents: List[Any],
    ensemble_type: str = "voting",
    **kwargs,
) -> Any:
    """Factory function to create ensemble from configuration.
    
    Args:
        agents: List of agent instances
        ensemble_type: "voting", "weighted", or "dynamic"
        **kwargs: Additional arguments for ensemble
        
    Returns:
        Ensemble instance
        
    Example:
        >>> agents = [MomentumAgent(), ContrarianAgent(), RandomAgent()]
        >>> ensemble = create_ensemble_from_config(agents, ensemble_type="weighted")
        >>> decision = ensemble.decide(market_state)
    """
    if ensemble_type == "voting":
        return VotingEnsemble(agents, **kwargs)
    elif ensemble_type == "weighted":
        return WeightedEnsemble(agents, **kwargs)
    elif ensemble_type == "dynamic":
        return DynamicEnsemble(agents, **kwargs)
    else:
        raise ValueError(f"Unknown ensemble_type: {ensemble_type}")


def evaluate_ensemble_diversity(
    ensemble_decisions: pd.DataFrame,
) -> Dict[str, float]:
    """Evaluate diversity metrics for an ensemble.
    
    Args:
        ensemble_decisions: DataFrame with columns for each agent's decision
        
    Returns:
        Dictionary with diversity metrics:
            - pairwise_disagreement: average pairwise disagreement rate
            - entropy: decision entropy
            - kohavi_wolpert: Kohavi-Wolpert diversity measure
            
    Example:
        >>> decisions_df = pd.DataFrame({
        ...     'agent_1': ['buy', 'sell', 'hold', ...],
        ...     'agent_2': ['buy', 'buy', 'hold', ...],
        ...     'agent_3': ['sell', 'buy', 'buy', ...],
        ... })
        >>> diversity = evaluate_ensemble_diversity(decisions_df)
        >>> print(f"Diversity: {diversity['pairwise_disagreement']:.3f}")
    """
    n_samples, n_agents = ensemble_decisions.shape
    
    # Pairwise disagreement
    disagreements = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            disagreement_rate = (
                ensemble_decisions.iloc[:, i] != ensemble_decisions.iloc[:, j]
            ).mean()
            disagreements.append(disagreement_rate)
    
    pairwise_disagreement = np.mean(disagreements) if disagreements else 0.0
    
    # Decision entropy
    from scipy.stats import entropy as scipy_entropy
    entropies = []
    for idx in range(n_samples):
        decisions = ensemble_decisions.iloc[idx].values
        decision_counts = Counter(decisions)
        probs = np.array(list(decision_counts.values())) / n_agents
        entropies.append(scipy_entropy(probs))
    
    mean_entropy = np.mean(entropies)
    
    return {
        "pairwise_disagreement": pairwise_disagreement,
        "entropy": mean_entropy,
        "n_agents": n_agents,
        "n_samples": n_samples,
    }


def compare_ensemble_strategies(
    agents: List[Any],
    market_data: pd.DataFrame,
    simulation_func: Callable,
    performance_metric: Callable,
) -> pd.DataFrame:
    """Compare different ensemble strategies on the same data.
    
    Args:
        agents: List of base agents
        market_data: Market data for simulation
        simulation_func: Function to run simulation
        performance_metric: Function to extract performance
        
    Returns:
        DataFrame comparing ensemble strategies:
            - ensemble_type
            - performance
            - diversity
            
    Example:
        >>> comparison = compare_ensemble_strategies(
        ...     agents, market_df, run_simulation, get_sharpe
        ... )
        >>> print(comparison.sort_values('performance', ascending=False))
    """
    ensemble_types = ["voting", "weighted", "dynamic"]
    results = []
    
    for ens_type in ensemble_types:
        ensemble = create_ensemble_from_config(agents, ensemble_type=ens_type)
        
        try:
            sim_result = simulation_func(ensemble, market_data)
            performance = performance_metric(sim_result)
            
            # Get diversity if available
            if hasattr(ensemble, 'get_decision_diversity'):
                diversity = ensemble.get_decision_diversity()
            else:
                diversity = np.nan
        except Exception as e:
            warnings.warn(f"Ensemble {ens_type} failed: {e}")
            performance = np.nan
            diversity = np.nan
        
        results.append({
            "ensemble_type": ens_type,
            "performance": performance,
            "diversity": diversity,
        })
    
    return pd.DataFrame(results)


def optimal_ensemble_weights(
    agent_performances: Dict[str, np.ndarray],
    method: str = "mean_variance",
    target_return: Optional[float] = None,
) -> Dict[str, float]:
    """Compute optimal weights for ensemble agents.
    
    Args:
        agent_performances: Dict mapping agent names to return arrays
        method: "mean_variance" (Markowitz), "equal", or "inverse_variance"
        target_return: Target return for mean-variance optimization
        
    Returns:
        Dictionary mapping agent names to optimal weights
        
    Example:
        >>> perfs = {"A": returns_a, "B": returns_b, "C": returns_c}
        >>> weights = optimal_ensemble_weights(perfs, method="mean_variance")
        >>> print(weights)
    """
    agent_names = list(agent_performances.keys())
    n_agents = len(agent_names)
    
    if method == "equal":
        weight = 1.0 / n_agents
        return {name: weight for name in agent_names}
    
    elif method == "inverse_variance":
        variances = [np.var(agent_performances[name]) for name in agent_names]
        inv_vars = [1.0 / v if v > 0 else 0.0 for v in variances]
        total_inv_var = sum(inv_vars)
        weights = [iv / total_inv_var for iv in inv_vars]
        return {name: w for name, w in zip(agent_names, weights)}
    
    elif method == "mean_variance":
        # Markowitz mean-variance optimization
        returns_matrix = np.column_stack([agent_performances[name] for name in agent_names])
        means = returns_matrix.mean(axis=0)
        cov_matrix = np.cov(returns_matrix.T)
        
        # Solve for minimum variance portfolio
        # Using simple closed-form solution (unconstrained)
        inv_cov = np.linalg.pinv(cov_matrix)
        ones = np.ones(n_agents)
        
        if target_return is None:
            # Minimum variance portfolio
            weights = inv_cov @ ones
            weights /= ones @ inv_cov @ ones
        else:
            # Target return portfolio
            A = ones @ inv_cov @ ones
            B = means @ inv_cov @ ones
            C = means @ inv_cov @ means
            
            lambda_1 = (C - target_return * B) / (A * C - B ** 2)
            lambda_2 = (target_return * A - B) / (A * C - B ** 2)
            
            weights = lambda_1 * (inv_cov @ ones) + lambda_2 * (inv_cov @ means)
        
        # Normalize and ensure non-negative
        weights = np.maximum(weights, 0)
        weights /= weights.sum()
        
        return {name: w for name, w in zip(agent_names, weights)}
    
    else:
        raise ValueError(f"Unknown method: {method}")

