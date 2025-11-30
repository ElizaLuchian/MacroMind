"""Cross-validation framework for time-series data.

This module provides cross-validation methods adapted for time-series trading data,
respecting temporal ordering and avoiding look-ahead bias.

Functions:
    walk_forward_cv: Walk-forward (rolling window) cross-validation
    expanding_window_cv: Expanding window cross-validation
    blocked_cv: Blocked time-series cross-validation
    
References:
    - Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for
      time series predictor evaluation
"""

from __future__ import annotations

from typing import Dict, List, Callable, Any, Optional, Tuple, Iterator
import warnings

import numpy as np
import pandas as pd


def walk_forward_cv(
    data_indices: np.ndarray,
    train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Walk-forward cross-validation splits for time-series.
    
    Uses a rolling window that moves forward in time. Training and test
    windows have fixed sizes.
    
    Args:
        data_indices: Array of indices (typically dates)
        train_size: Number of samples in training window
        test_size: Number of samples in test window
        step_size: How many samples to move forward (default: test_size)
        
    Returns:
        List of (train_indices, test_indices) tuples
        
    Example:
        >>> indices = np.arange(100)
        >>> splits = walk_forward_cv(indices, train_size=50, test_size=10)
        >>> for train_idx, test_idx in splits:
        ...     # Train on train_idx, test on test_idx
        ...     ...
    """
    if step_size is None:
        step_size = test_size
    
    n_samples = len(data_indices)
    splits = []
    
    start = 0
    while start + train_size + test_size <= n_samples:
        train_end = start + train_size
        test_end = train_end + test_size
        
        train_indices = data_indices[start:train_end]
        test_indices = data_indices[train_end:test_end]
        
        splits.append((train_indices, test_indices))
        
        start += step_size
    
    return splits


def expanding_window_cv(
    data_indices: np.ndarray,
    min_train_size: int,
    test_size: int,
    step_size: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Expanding window cross-validation for time-series.
    
    Training window starts small and grows over time. Test window has fixed size.
    
    Args:
        data_indices: Array of indices
        min_train_size: Minimum training samples
        test_size: Number of samples in test window
        step_size: How many samples to move forward (default: test_size)
        
    Returns:
        List of (train_indices, test_indices) tuples
        
    Example:
        >>> splits = expanding_window_cv(indices, min_train_size=50, test_size=10)
        >>> # First split uses 50 training samples, next uses 60, then 70, etc.
    """
    if step_size is None:
        step_size = test_size
    
    n_samples = len(data_indices)
    splits = []
    
    train_end = min_train_size
    while train_end + test_size <= n_samples:
        test_end = train_end + test_size
        
        train_indices = data_indices[:train_end]
        test_indices = data_indices[train_end:test_end]
        
        splits.append((train_indices, test_indices))
        
        train_end += step_size
    
    return splits


def blocked_cv(
    data_indices: np.ndarray,
    n_splits: int,
    train_ratio: float = 0.8,
    gap_size: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Blocked cross-validation for time-series.
    
    Divides time series into n_splits consecutive blocks. Each block is used
    as test set once, with all preceding data as training set.
    
    Args:
        data_indices: Array of indices
        n_splits: Number of blocks/folds
        train_ratio: Ratio of data to use for training in each split
        gap_size: Number of samples to skip between train and test (avoid leakage)
        
    Returns:
        List of (train_indices, test_indices) tuples
        
    Example:
        >>> splits = blocked_cv(indices, n_splits=5, gap_size=5)
        >>> # 5 splits with 5-sample gap between training and testing
    """
    n_samples = len(data_indices)
    block_size = n_samples // n_splits
    
    splits = []
    
    for i in range(n_splits):
        test_start = i * block_size
        test_end = min((i + 1) * block_size, n_samples)
        
        # Training data: all data before test set
        train_end = max(0, test_start - gap_size)
        train_start = max(0, train_end - int(train_end * train_ratio))
        
        if train_end > train_start and test_end > test_start:
            train_indices = data_indices[train_start:train_end]
            test_indices = data_indices[test_start:test_end]
            splits.append((train_indices, test_indices))
    
    return splits


def cross_validate_agent(
    agent_factory: Callable,
    run_simulation: Callable,
    performance_metric: Callable,
    data: pd.DataFrame,
    cv_method: str = "walk_forward",
    train_size: int = 100,
    test_size: int = 20,
    **cv_kwargs,
) -> Dict[str, Any]:
    """Cross-validate an agent using specified CV method.
    
    Args:
        agent_factory: Function that creates a fresh agent instance
        run_simulation: Function that runs simulation given (agent, data)
        performance_metric: Function that extracts performance from simulation result
        data: DataFrame with time-series data (must have datetime index or 'date' column)
        cv_method: "walk_forward", "expanding", or "blocked"
        train_size: Training window size
        test_size: Test window size
        **cv_kwargs: Additional kwargs for CV method
        
    Returns:
        Dictionary with:
            - cv_scores: list of test scores for each fold
            - mean_score: mean test score
            - std_score: standard deviation of test scores
            - n_splits: number of folds
            - train_scores: list of training scores (optional)
            
    Example:
        >>> def create_agent():
        ...     return MomentumAgent()
        >>> 
        >>> def run_sim(agent, data):
        ...     return simulate(agent, data)
        >>> 
        >>> def get_sharpe(result):
        ...     return result['sharpe_ratio']
        >>> 
        >>> cv_results = cross_validate_agent(
        ...     create_agent, run_sim, get_sharpe, df,
        ...     cv_method="walk_forward", train_size=100, test_size=20
        ... )
        >>> print(f"Mean CV score: {cv_results['mean_score']:.3f}")
    """
    # Get indices (dates)
    if isinstance(data.index, pd.DatetimeIndex):
        indices = data.index.values
    elif 'date' in data.columns:
        indices = data['date'].values
    else:
        indices = np.arange(len(data))
    
    # Generate splits
    if cv_method == "walk_forward":
        splits = walk_forward_cv(
            indices,
            train_size=train_size,
            test_size=test_size,
            **cv_kwargs
        )
    elif cv_method == "expanding":
        splits = expanding_window_cv(
            indices,
            min_train_size=train_size,
            test_size=test_size,
            **cv_kwargs
        )
    elif cv_method == "blocked":
        n_splits = cv_kwargs.get("n_splits", 5)
        splits = blocked_cv(
            indices,
            n_splits=n_splits,
            **{k: v for k, v in cv_kwargs.items() if k != "n_splits"}
        )
    else:
        raise ValueError(f"Unknown cv_method: {cv_method}")
    
    # Run cross-validation
    test_scores = []
    train_scores = []
    
    for fold_idx, (train_indices, test_indices) in enumerate(splits):
        # Select data
        if isinstance(data.index, pd.DatetimeIndex) or 'date' in data.columns:
            if isinstance(data.index, pd.DatetimeIndex):
                train_data = data.loc[train_indices]
                test_data = data.loc[test_indices]
            else:
                train_data = data[data['date'].isin(train_indices)]
                test_data = data[data['date'].isin(test_indices)]
        else:
            train_data = data.iloc[train_indices]
            test_data = data.iloc[test_indices]
        
        if len(train_data) == 0 or len(test_data) == 0:
            warnings.warn(f"Empty data in fold {fold_idx}, skipping")
            continue
        
        # Create fresh agent
        agent = agent_factory()
        
        # Train (if applicable - some agents don't need training)
        # For now, we just run simulation on train data
        try:
            train_result = run_simulation(agent, train_data)
            train_score = performance_metric(train_result)
            train_scores.append(train_score)
        except Exception as e:
            warnings.warn(f"Training failed in fold {fold_idx}: {e}")
            train_scores.append(np.nan)
        
        # Test
        try:
            # Create fresh agent for testing
            test_agent = agent_factory()
            test_result = run_simulation(test_agent, test_data)
            test_score = performance_metric(test_result)
            test_scores.append(test_score)
        except Exception as e:
            warnings.warn(f"Testing failed in fold {fold_idx}: {e}")
            test_scores.append(np.nan)
    
    # Remove NaN scores
    test_scores = [s for s in test_scores if not np.isnan(s)]
    train_scores = [s for s in train_scores if not np.isnan(s)]
    
    if len(test_scores) == 0:
        return {
            "cv_scores": [],
            "mean_score": np.nan,
            "std_score": np.nan,
            "n_splits": 0,
            "train_scores": train_scores,
        }
    
    return {
        "cv_scores": test_scores,
        "mean_score": np.mean(test_scores),
        "std_score": np.std(test_scores, ddof=1) if len(test_scores) > 1 else 0.0,
        "n_splits": len(test_scores),
        "train_scores": train_scores,
        "train_mean": np.mean(train_scores) if len(train_scores) > 0 else np.nan,
        "train_std": np.std(train_scores, ddof=1) if len(train_scores) > 1 else 0.0,
    }


def compare_agents_cv(
    agent_factories: Dict[str, Callable],
    run_simulation: Callable,
    performance_metric: Callable,
    data: pd.DataFrame,
    cv_method: str = "walk_forward",
    **cv_kwargs,
) -> pd.DataFrame:
    """Compare multiple agents using cross-validation.
    
    Args:
        agent_factories: Dict mapping agent names to factory functions
        run_simulation: Simulation function
        performance_metric: Performance extraction function
        data: Time-series data
        cv_method: Cross-validation method
        **cv_kwargs: CV parameters
        
    Returns:
        DataFrame with CV results for each agent:
            - agent: agent name
            - mean_score: mean CV test score
            - std_score: standard deviation
            - train_mean: mean training score
            - generalization_gap: train_mean - mean_score
            
    Example:
        >>> factories = {
        ...     "Momentum": lambda: MomentumAgent(),
        ...     "Contrarian": lambda: ContrarianAgent(),
        ... }
        >>> comparison = compare_agents_cv(factories, run_sim, get_sharpe, df)
        >>> print(comparison.sort_values('mean_score', ascending=False))
    """
    results = []
    
    for agent_name, agent_factory in agent_factories.items():
        cv_result = cross_validate_agent(
            agent_factory,
            run_simulation,
            performance_metric,
            data,
            cv_method=cv_method,
            **cv_kwargs,
        )
        
        generalization_gap = cv_result["train_mean"] - cv_result["mean_score"]
        
        results.append({
            "agent": agent_name,
            "mean_score": cv_result["mean_score"],
            "std_score": cv_result["std_score"],
            "train_mean": cv_result["train_mean"],
            "train_std": cv_result["train_std"],
            "generalization_gap": generalization_gap,
            "n_splits": cv_result["n_splits"],
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("mean_score", ascending=False).reset_index(drop=True)
    
    return df


def detect_overfitting(
    cv_results: Dict[str, Any],
    threshold: float = 0.1,
) -> Dict[str, Any]:
    """Detect potential overfitting from CV results.
    
    Args:
        cv_results: Output from cross_validate_agent()
        threshold: Threshold for generalization gap to flag overfitting
        
    Returns:
        Dictionary with overfitting diagnostics:
            - is_overfitting: bool
            - generalization_gap: train - test
            - cv_stability: inverse of CV coefficient of variation
            - recommendation: text recommendation
            
    Example:
        >>> diagnosis = detect_overfitting(cv_results, threshold=0.15)
        >>> if diagnosis['is_overfitting']:
        ...     print(diagnosis['recommendation'])
    """
    train_mean = cv_results.get("train_mean", np.nan)
    test_mean = cv_results.get("mean_score", np.nan)
    test_std = cv_results.get("std_score", np.nan)
    
    # Generalization gap
    if not np.isnan(train_mean) and not np.isnan(test_mean):
        gap = train_mean - test_mean
        gap_relative = gap / abs(train_mean) if abs(train_mean) > 1e-10 else 0.0
    else:
        gap = gap_relative = np.nan
    
    # CV stability
    if not np.isnan(test_mean) and not np.isnan(test_std) and abs(test_mean) > 1e-10:
        cv_coefficient = test_std / abs(test_mean)
        stability = 1.0 / (1.0 + cv_coefficient)
    else:
        stability = np.nan
    
    # Diagnosis
    is_overfitting = False
    recommendation = "Performance appears stable."
    
    if not np.isnan(gap_relative) and gap_relative > threshold:
        is_overfitting = True
        recommendation = (
            f"Potential overfitting detected (gap={gap:.4f}, {gap_relative*100:.1f}%). "
            "Consider: (1) regularization, (2) simpler model, (3) more training data."
        )
    elif not np.isnan(stability) and stability < 0.5:
        recommendation = (
            f"High CV variance detected (stability={stability:.3f}). "
            "Performance may be sensitive to data selection. "
            "Consider: (1) more CV folds, (2) longer time windows, (3) ensemble methods."
        )
    
    return {
        "is_overfitting": is_overfitting,
        "generalization_gap": gap,
        "generalization_gap_relative": gap_relative,
        "cv_stability": stability,
        "train_mean": train_mean,
        "test_mean": test_mean,
        "test_std": test_std,
        "recommendation": recommendation,
    }


def learning_curve(
    agent_factory: Callable,
    run_simulation: Callable,
    performance_metric: Callable,
    data: pd.DataFrame,
    train_sizes: List[int],
    test_size: int = 20,
) -> pd.DataFrame:
    """Generate learning curve by varying training set size.
    
    Args:
        agent_factory: Function that creates agent
        run_simulation: Simulation function
        performance_metric: Performance extraction function
        data: Time-series data
        train_sizes: List of training sizes to test
        test_size: Fixed test size
        
    Returns:
        DataFrame with learning curve results:
            - train_size: training size
            - train_score: training performance
            - test_score: test performance
            - gap: train_score - test_score
            
    Example:
        >>> curve = learning_curve(
        ...     lambda: MomentumAgent(), run_sim, get_sharpe,
        ...     df, train_sizes=[50, 100, 150, 200]
        ... )
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(curve['train_size'], curve['train_score'], label='Train')
        >>> plt.plot(curve['train_size'], curve['test_score'], label='Test')
        >>> plt.legend(); plt.show()
    """
    results = []
    
    for train_size in train_sizes:
        # Use expanding window with this train size
        cv_result = cross_validate_agent(
            agent_factory,
            run_simulation,
            performance_metric,
            data,
            cv_method="expanding",
            min_train_size=train_size,
            test_size=test_size,
        )
        
        results.append({
            "train_size": train_size,
            "train_score": cv_result["train_mean"],
            "test_score": cv_result["mean_score"],
            "test_std": cv_result["std_score"],
            "gap": cv_result["train_mean"] - cv_result["mean_score"],
        })
    
    return pd.DataFrame(results)

