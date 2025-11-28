"""Evaluation metrics for the latent news simulation.

This module provides comprehensive financial evaluation metrics including:
- Basic metrics (directional accuracy, profitability)
- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis (maximum drawdown, recovery time)
- Trading performance metrics (win rate, profit factor)
- Risk metrics (VaR, CVaR, beta)
- Transaction cost modeling

References:
- Bailey, D. H., & Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier"
- Magdon-Ismail, M., et al. (2004). "On the Maximum Drawdown of a Brownian Motion"
- Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions"
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr

ActionLog = Mapping[str, Sequence[str]]

ACTION_TO_SIGN = {"buy": 1, "sell": -1, "hold": 0}

# Constants for financial calculations
TRADING_DAYS_PER_YEAR = 252
HOURS_PER_YEAR = 252 * 6.5  # Trading hours
MINUTES_PER_YEAR = HOURS_PER_YEAR * 60


def compute_directional_accuracy(reference: Sequence[float], simulated: Sequence[float]) -> float:
    """Share of days where simulated direction matches reference direction."""
    if len(reference) != len(simulated):
        raise ValueError("Reference and simulated series must have the same length.")
    if len(reference) < 2:
        return 0.0
    ref_diff = np.diff(reference)
    sim_diff = np.diff(simulated)
    matches = np.sign(ref_diff) == np.sign(sim_diff)
    return float(matches.sum() / len(matches))


def agent_profitability(action_log: ActionLog, price_series: Sequence[float]) -> pd.DataFrame:
    """Compute a naive profitability metric per agent."""
    if len(price_series) < 2:
        raise ValueError("Price series must contain at least two values.")
    returns = np.diff(price_series)
    results: List[Dict[str, float]] = []
    for name, actions in action_log.items():
        signals = np.array([ACTION_TO_SIGN.get(a, 0) for a in actions[: len(returns)]])
        pnl = float(np.dot(signals, returns))
        accuracy = float((signals == np.sign(returns)).mean())
        results.append({"agent": name, "pnl": pnl, "directional_accuracy": accuracy})
    return pd.DataFrame(results)


def volatility_clustering(prices: Sequence[float], window: int = 5) -> float:
    """Estimate autocorrelation of squared returns as a proxy for volatility clustering."""
    returns = np.diff(prices)
    if len(returns) <= window:
        return 0.0
    squared = returns**2
    series = pd.Series(squared)
    return float(series.autocorr(lag=window))


def cluster_price_correlation(clusters: Sequence[int], prices: Sequence[float]) -> float:
    """Compute Pearson correlation between clusters and price changes."""
    if len(clusters) != len(prices):
        raise ValueError("Clusters and prices must have the same length.")
    if len(prices) < 2:
        return 0.0
    price_changes = np.diff(prices)
    clipped_clusters = np.array(clusters[1:], dtype=float)
    corr, _ = pearsonr(clipped_clusters, price_changes)
    return float(corr)


def decision_correlation_matrix(action_log: ActionLog) -> pd.DataFrame:
    """Correlation of discrete agent decisions."""
    encoded = {agent: [ACTION_TO_SIGN.get(action, 0) for action in actions] for agent, actions in action_log.items()}
    df = pd.DataFrame(encoded)
    if df.empty:
        raise ValueError("Action log is empty.")
    return df.corr()


def summarize_metrics(
    reference_prices: Sequence[float],
    simulated_prices: Sequence[float],
    action_log: ActionLog,
    clusters: Sequence[int],
) -> Dict[str, object]:
    """Bundle the key summary statistics."""
    summary = {
        "directional_accuracy": compute_directional_accuracy(reference_prices, simulated_prices),
        "volatility_clustering": volatility_clustering(simulated_prices),
        "cluster_price_correlation": cluster_price_correlation(clusters, simulated_prices),
        "agent_profitability": agent_profitability(action_log, simulated_prices),
        "decision_correlation": decision_correlation_matrix(action_log),
    }
    return summary


# ============================================================================
# RISK-ADJUSTED RETURNS
# ============================================================================


def compute_returns(prices: Sequence[float], log_returns: bool = False) -> np.ndarray:
    """Compute returns from price series.
    
    Args:
        prices: Price series
        log_returns: If True, compute log returns; otherwise simple returns
        
    Returns:
        Array of returns
    """
    prices_arr = np.array(prices, dtype=float)
    if len(prices_arr) < 2:
        return np.array([])
    
    if log_returns:
        return np.diff(np.log(prices_arr))
    else:
        return np.diff(prices_arr) / prices_arr[:-1]


def sharpe_ratio(
    prices: Sequence[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    log_returns: bool = False,
) -> float:
    """Compute annualized Sharpe ratio.
    
    Sharpe Ratio = (Mean Return - Risk Free Rate) / Std Dev of Returns
    
    Args:
        prices: Price series
        risk_free_rate: Annual risk-free rate (e.g., 0.02 for 2%)
        periods_per_year: Number of periods in a year (252 for daily)
        log_returns: If True, use log returns
        
    Returns:
        Annualized Sharpe ratio
        
    References:
        Sharpe, W. F. (1966). "Mutual Fund Performance"
    """
    returns = compute_returns(prices, log_returns=log_returns)
    if len(returns) == 0:
        return 0.0
    
    # Convert annual risk-free rate to per-period rate
    rf_per_period = risk_free_rate / periods_per_year
    
    excess_returns = returns - rf_per_period
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return 0.0
    
    # Annualize
    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return float(sharpe)


def sortino_ratio(
    prices: Sequence[float],
    risk_free_rate: float = 0.0,
    target_return: Optional[float] = None,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    log_returns: bool = False,
) -> float:
    """Compute annualized Sortino ratio.
    
    Sortino Ratio = (Mean Return - Target) / Downside Deviation
    Downside deviation only considers returns below target.
    
    Args:
        prices: Price series
        risk_free_rate: Annual risk-free rate
        target_return: Target return threshold (defaults to risk_free_rate)
        periods_per_year: Number of periods in a year
        log_returns: If True, use log returns
        
    Returns:
        Annualized Sortino ratio
        
    References:
        Sortino, F. A., & Price, L. N. (1994). "Performance Measurement in a Downside Risk Framework"
    """
    returns = compute_returns(prices, log_returns=log_returns)
    if len(returns) == 0:
        return 0.0
    
    # Default target is risk-free rate
    if target_return is None:
        target_return = risk_free_rate
    
    target_per_period = target_return / periods_per_year
    
    excess_returns = returns - target_per_period
    mean_excess = np.mean(excess_returns)
    
    # Downside deviation: only negative excess returns
    downside_returns = excess_returns[excess_returns < 0]
    if len(downside_returns) == 0:
        return float('inf') if mean_excess > 0 else 0.0
    
    downside_std = np.sqrt(np.mean(downside_returns**2))
    
    if downside_std == 0:
        return 0.0
    
    # Annualize
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def calmar_ratio(
    prices: Sequence[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    log_returns: bool = False,
) -> float:
    """Compute Calmar ratio (CAGR / Max Drawdown).
    
    Args:
        prices: Price series
        periods_per_year: Number of periods in a year
        log_returns: If True, use log returns for CAGR
        
    Returns:
        Calmar ratio
        
    References:
        Young, T. W. (1991). "Calmar Ratio: A Smoother Tool"
    """
    if len(prices) < 2:
        return 0.0
    
    # Handle zero starting price
    if prices[0] == 0:
        return 0.0
    
    # Compute CAGR
    total_periods = len(prices) - 1
    years = total_periods / periods_per_year
    
    if years == 0:
        return 0.0
    
    if log_returns:
        total_return = np.log(prices[-1] / prices[0])
        cagr = total_return / years
    else:
        total_return = (prices[-1] / prices[0]) - 1
        cagr = (1 + total_return) ** (1 / years) - 1
    
    # Compute max drawdown
    max_dd = maximum_drawdown(prices)
    
    if max_dd == 0:
        return float('inf') if cagr > 0 else 0.0
    
    return float(cagr / abs(max_dd))


def information_ratio(
    portfolio_prices: Sequence[float],
    benchmark_prices: Sequence[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute information ratio (excess return / tracking error).
    
    Args:
        portfolio_prices: Portfolio price series
        benchmark_prices: Benchmark price series
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized information ratio
    """
    if len(portfolio_prices) != len(benchmark_prices):
        raise ValueError("Portfolio and benchmark must have same length")
    
    port_returns = compute_returns(portfolio_prices)
    bench_returns = compute_returns(benchmark_prices)
    
    if len(port_returns) == 0:
        return 0.0
    
    excess_returns = port_returns - bench_returns
    mean_excess = np.mean(excess_returns)
    tracking_error = np.std(excess_returns, ddof=1)
    
    if tracking_error == 0:
        return 0.0
    
    # Annualize
    ir = (mean_excess / tracking_error) * np.sqrt(periods_per_year)
    return float(ir)


# ============================================================================
# DRAWDOWN ANALYSIS
# ============================================================================


def maximum_drawdown(prices: Sequence[float]) -> float:
    """Compute maximum drawdown (largest peak-to-trough decline).
    
    Args:
        prices: Price series
        
    Returns:
        Maximum drawdown as a negative percentage (e.g., -0.25 for 25% drawdown)
    """
    prices_arr = np.array(prices, dtype=float)
    if len(prices_arr) < 2:
        return 0.0
    
    cummax = np.maximum.accumulate(prices_arr)
    drawdowns = (prices_arr - cummax) / cummax
    max_dd = np.min(drawdowns)
    
    return float(max_dd)


def drawdown_series(prices: Sequence[float]) -> np.ndarray:
    """Compute drawdown at each point in time.
    
    Args:
        prices: Price series
        
    Returns:
        Array of drawdowns (as negative percentages)
    """
    prices_arr = np.array(prices, dtype=float)
    if len(prices_arr) < 1:
        return np.array([])
    
    cummax = np.maximum.accumulate(prices_arr)
    drawdowns = (prices_arr - cummax) / cummax
    
    return drawdowns


def average_drawdown(prices: Sequence[float]) -> float:
    """Compute average of all drawdown values.
    
    Args:
        prices: Price series
        
    Returns:
        Average drawdown
    """
    dd = drawdown_series(prices)
    if len(dd) == 0:
        return 0.0
    
    # Only consider actual drawdown periods (negative values)
    dd_periods = dd[dd < 0]
    if len(dd_periods) == 0:
        return 0.0
    
    return float(np.mean(dd_periods))


def drawdown_details(prices: Sequence[float]) -> Dict[str, float]:
    """Compute comprehensive drawdown statistics.
    
    Args:
        prices: Price series
        
    Returns:
        Dictionary with drawdown metrics:
        - max_drawdown: Maximum drawdown
        - avg_drawdown: Average drawdown
        - max_drawdown_duration: Longest consecutive drawdown period
        - current_drawdown: Current drawdown level
    """
    prices_arr = np.array(prices, dtype=float)
    dd = drawdown_series(prices)
    
    if len(dd) == 0:
        return {
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "current_drawdown": 0.0,
        }
    
    # Maximum drawdown duration
    in_drawdown = dd < 0
    durations = []
    current_duration = 0
    
    for is_dd in in_drawdown:
        if is_dd:
            current_duration += 1
        else:
            if current_duration > 0:
                durations.append(current_duration)
            current_duration = 0
    
    if current_duration > 0:
        durations.append(current_duration)
    
    max_duration = max(durations) if durations else 0
    
    return {
        "max_drawdown": float(np.min(dd)),
        "avg_drawdown": float(np.mean(dd[dd < 0])) if np.any(dd < 0) else 0.0,
        "max_drawdown_duration": max_duration,
        "current_drawdown": float(dd[-1]),
    }


def underwater_periods(prices: Sequence[float]) -> List[Tuple[int, int, float]]:
    """Identify all underwater periods (time below previous peak).
    
    Args:
        prices: Price series
        
    Returns:
        List of tuples (start_idx, end_idx, max_drawdown_in_period)
    """
    dd = drawdown_series(prices)
    if len(dd) == 0:
        return []
    
    periods = []
    in_drawdown = False
    start_idx = 0
    period_max_dd = 0.0
    
    for i, d in enumerate(dd):
        if d < 0:
            if not in_drawdown:
                start_idx = i
                in_drawdown = True
                period_max_dd = d
            else:
                period_max_dd = min(period_max_dd, d)
        else:
            if in_drawdown:
                periods.append((start_idx, i - 1, period_max_dd))
                in_drawdown = False
    
    # Handle case where we end in drawdown
    if in_drawdown:
        periods.append((start_idx, len(dd) - 1, period_max_dd))
    
    return periods


# ============================================================================
# TRADING PERFORMANCE METRICS
# ============================================================================


def win_rate(action_log: ActionLog, price_series: Sequence[float]) -> pd.DataFrame:
    """Compute win rate (percentage of profitable trades) for each agent.
    
    Args:
        action_log: Dictionary mapping agent names to action sequences
        price_series: Price series
        
    Returns:
        DataFrame with agent win rates and trade counts
    """
    if len(price_series) < 2:
        raise ValueError("Price series must contain at least two values.")
    
    returns = np.diff(price_series)
    results = []
    
    for name, actions in action_log.items():
        signals = np.array([ACTION_TO_SIGN.get(a, 0) for a in actions[: len(returns)]])
        
        # Only consider actual trades (not holds)
        trades_mask = signals != 0
        if not np.any(trades_mask):
            results.append({
                "agent": name,
                "win_rate": 0.0,
                "trades": 0,
                "wins": 0,
                "losses": 0,
            })
            continue
        
        trade_returns = signals[trades_mask] * returns[trades_mask]
        wins = np.sum(trade_returns > 0)
        losses = np.sum(trade_returns < 0)
        total_trades = wins + losses
        
        win_rate_pct = wins / total_trades if total_trades > 0 else 0.0
        
        results.append({
            "agent": name,
            "win_rate": win_rate_pct,
            "trades": int(np.sum(trades_mask)),
            "wins": int(wins),
            "losses": int(losses),
        })
    
    return pd.DataFrame(results)


def profit_factor(action_log: ActionLog, price_series: Sequence[float]) -> pd.DataFrame:
    """Compute profit factor (gross profits / gross losses) for each agent.
    
    Args:
        action_log: Dictionary mapping agent names to action sequences
        price_series: Price series
        
    Returns:
        DataFrame with profit factors
    """
    if len(price_series) < 2:
        raise ValueError("Price series must contain at least two values.")
    
    returns = np.diff(price_series)
    results = []
    
    for name, actions in action_log.items():
        signals = np.array([ACTION_TO_SIGN.get(a, 0) for a in actions[: len(returns)]])
        trade_returns = signals * returns
        
        gross_profits = np.sum(trade_returns[trade_returns > 0])
        gross_losses = abs(np.sum(trade_returns[trade_returns < 0]))
        
        if gross_losses == 0:
            pf = float('inf') if gross_profits > 0 else 0.0
        else:
            pf = gross_profits / gross_losses
        
        results.append({
            "agent": name,
            "profit_factor": pf,
            "gross_profits": gross_profits,
            "gross_losses": gross_losses,
        })
    
    return pd.DataFrame(results)


def trade_expectancy(action_log: ActionLog, price_series: Sequence[float]) -> pd.DataFrame:
    """Compute average trade expectancy for each agent.
    
    Args:
        action_log: Dictionary mapping agent names to action sequences
        price_series: Price series
        
    Returns:
        DataFrame with expectancy metrics
    """
    if len(price_series) < 2:
        raise ValueError("Price series must contain at least two values.")
    
    returns = np.diff(price_series)
    results = []
    
    for name, actions in action_log.items():
        signals = np.array([ACTION_TO_SIGN.get(a, 0) for a in actions[: len(returns)]])
        
        # Only consider actual trades
        trades_mask = signals != 0
        if not np.any(trades_mask):
            results.append({
                "agent": name,
                "expectancy": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
            })
            continue
        
        trade_returns = signals[trades_mask] * returns[trades_mask]
        
        wins = trade_returns[trade_returns > 0]
        losses = trade_returns[trade_returns < 0]
        
        avg_win = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        expectancy = np.mean(trade_returns)
        
        results.append({
            "agent": name,
            "expectancy": expectancy,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        })
    
    return pd.DataFrame(results)


# ============================================================================
# RISK METRICS
# ============================================================================


def value_at_risk(
    prices: Sequence[float],
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Compute Value at Risk (VaR).
    
    Args:
        prices: Price series
        confidence: Confidence level (e.g., 0.95 for 95% VaR)
        method: "historical" or "parametric" (Gaussian assumption)
        
    Returns:
        VaR as a positive number (e.g., 0.02 means 2% loss at confidence level)
    """
    returns = compute_returns(prices)
    if len(returns) == 0:
        return 0.0
    
    if method == "historical":
        # Historical VaR: percentile of actual returns
        var = -np.percentile(returns, (1 - confidence) * 100)
    elif method == "parametric":
        # Parametric VaR: assumes normal distribution
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        z_score = stats.norm.ppf(1 - confidence)
        var = -(mean + z_score * std)
    else:
        raise ValueError(f"Unknown VaR method: {method}")
    
    return float(max(0, var))


def conditional_value_at_risk(
    prices: Sequence[float],
    confidence: float = 0.95,
) -> float:
    """Compute Conditional Value at Risk (CVaR / Expected Shortfall).
    
    CVaR is the expected loss given that loss exceeds VaR.
    
    Args:
        prices: Price series
        confidence: Confidence level
        
    Returns:
        CVaR as a positive number
    """
    returns = compute_returns(prices)
    if len(returns) == 0:
        return 0.0
    
    var_threshold = -value_at_risk(prices, confidence, method="historical")
    
    # Average of returns worse than VaR
    tail_losses = returns[returns <= var_threshold]
    
    if len(tail_losses) == 0:
        return 0.0
    
    cvar = -np.mean(tail_losses)
    return float(cvar)


def beta_to_market(
    portfolio_prices: Sequence[float],
    market_prices: Sequence[float],
) -> float:
    """Compute portfolio beta relative to market.
    
    Beta = Cov(portfolio_returns, market_returns) / Var(market_returns)
    
    Args:
        portfolio_prices: Portfolio price series
        market_prices: Market/benchmark price series
        
    Returns:
        Beta coefficient
    """
    if len(portfolio_prices) != len(market_prices):
        raise ValueError("Portfolio and market must have same length")
    
    port_returns = compute_returns(portfolio_prices)
    market_returns = compute_returns(market_prices)
    
    if len(port_returns) == 0:
        return 0.0
    
    market_var = np.var(market_returns, ddof=1)
    if market_var == 0:
        return 0.0
    
    covariance = np.cov(port_returns, market_returns)[0, 1]
    beta = covariance / market_var
    
    return float(beta)


def tracking_error(
    portfolio_prices: Sequence[float],
    benchmark_prices: Sequence[float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Compute annualized tracking error.
    
    Args:
        portfolio_prices: Portfolio price series
        benchmark_prices: Benchmark price series
        periods_per_year: Number of periods in a year
        
    Returns:
        Annualized tracking error
    """
    if len(portfolio_prices) != len(benchmark_prices):
        raise ValueError("Portfolio and benchmark must have same length")
    
    port_returns = compute_returns(portfolio_prices)
    bench_returns = compute_returns(benchmark_prices)
    
    if len(port_returns) == 0:
        return 0.0
    
    diff_returns = port_returns - bench_returns
    te = np.std(diff_returns, ddof=1) * np.sqrt(periods_per_year)
    
    return float(te)


# ============================================================================
# TRANSACTION COSTS
# ============================================================================


def apply_transaction_costs(
    action_log: ActionLog,
    price_series: Sequence[float],
    fixed_cost: float = 0.0,
    proportional_cost: float = 0.0,
    slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """Compute P&L after transaction costs.
    
    Args:
        action_log: Dictionary mapping agent names to action sequences
        price_series: Price series
        fixed_cost: Fixed cost per trade (e.g., 0.01)
        proportional_cost: Proportional cost (e.g., 0.001 for 10 bps)
        slippage_bps: Slippage in basis points (e.g., 5.0 for 5 bps)
        
    Returns:
        DataFrame with adjusted P&L
    """
    if len(price_series) < 2:
        raise ValueError("Price series must contain at least two values.")
    
    returns = np.diff(price_series)
    prices_arr = np.array(price_series[:-1])  # Prices at trade time
    
    results = []
    
    for name, actions in action_log.items():
        signals = np.array([ACTION_TO_SIGN.get(a, 0) for a in actions[: len(returns)]])
        
        # Gross P&L
        gross_pnl = float(np.dot(signals, returns))
        
        # Count trades (transitions from hold or direction changes)
        trades_mask = signals != 0
        num_trades = int(np.sum(trades_mask))
        
        # Calculate costs
        total_fixed_cost = num_trades * fixed_cost
        
        # Proportional cost based on trade value
        trade_values = abs(signals[trades_mask]) * prices_arr[trades_mask]
        total_proportional_cost = np.sum(trade_values) * proportional_cost
        
        # Slippage cost
        slippage_rate = slippage_bps / 10000  # Convert bps to decimal
        total_slippage = np.sum(trade_values) * slippage_rate
        
        total_costs = total_fixed_cost + total_proportional_cost + total_slippage
        net_pnl = gross_pnl - total_costs
        
        results.append({
            "agent": name,
            "gross_pnl": gross_pnl,
            "transaction_costs": total_costs,
            "net_pnl": net_pnl,
            "num_trades": num_trades,
        })
    
    return pd.DataFrame(results)


def market_impact_cost(
    trade_size: float,
    daily_volume: float,
    participation_rate: float = 0.1,
    impact_coefficient: float = 0.1,
) -> float:
    """Estimate market impact cost using square-root model.
    
    Market Impact = sigma * (trade_size / daily_volume)^0.5
    
    Args:
        trade_size: Size of trade (number of shares or dollar amount)
        daily_volume: Average daily volume
        participation_rate: Fraction of daily volume traded
        impact_coefficient: Impact parameter (typically 0.1-0.5)
        
    Returns:
        Market impact as percentage of trade
        
    References:
        Almgren, R., & Chriss, N. (2001). "Optimal execution of portfolio transactions"
    """
    if daily_volume <= 0:
        return 0.0
    
    volume_fraction = trade_size / daily_volume
    impact = impact_coefficient * np.sqrt(volume_fraction)
    
    return float(impact)


# ============================================================================
# COMPREHENSIVE PERFORMANCE REPORTING
# ============================================================================


def generate_performance_report(
    prices: Sequence[float],
    action_log: Optional[ActionLog] = None,
    reference_prices: Optional[Sequence[float]] = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    include_transaction_costs: bool = False,
    fixed_cost: float = 0.0,
    proportional_cost: float = 0.0,
) -> Dict[str, object]:
    """Generate comprehensive performance report.
    
    Args:
        prices: Price series (simulated or actual)
        action_log: Optional agent action log for agent-specific metrics
        reference_prices: Optional reference prices for comparison
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        include_transaction_costs: Whether to include transaction cost analysis
        fixed_cost: Fixed transaction cost per trade
        proportional_cost: Proportional transaction cost
        
    Returns:
        Dictionary containing all performance metrics organized by category
    """
    report = {}
    
    # Return Metrics
    returns = compute_returns(prices)
    if len(returns) > 0:
        report["returns"] = {
            "total_return": (prices[-1] / prices[0] - 1) if prices[0] != 0 else 0.0,
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns, ddof=1)),
            "min_return": float(np.min(returns)),
            "max_return": float(np.max(returns)),
            "skewness": float(stats.skew(returns)),
            "kurtosis": float(stats.kurtosis(returns)),
        }
    
    # Risk-Adjusted Returns
    report["risk_adjusted"] = {
        "sharpe_ratio": sharpe_ratio(prices, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(prices, risk_free_rate, None, periods_per_year),
        "calmar_ratio": calmar_ratio(prices, periods_per_year),
    }
    
    # Drawdown Analysis
    report["drawdown"] = drawdown_details(prices)
    
    # Risk Metrics
    report["risk"] = {
        "value_at_risk_95": value_at_risk(prices, 0.95, "historical"),
        "cvar_95": conditional_value_at_risk(prices, 0.95),
        "volatility_annualized": float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 0 else 0.0,
    }
    
    # Comparison to reference if provided
    if reference_prices is not None and len(reference_prices) == len(prices):
        report["comparison"] = {
            "directional_accuracy": compute_directional_accuracy(reference_prices, prices),
            "beta": beta_to_market(prices, reference_prices),
            "tracking_error": tracking_error(prices, reference_prices, periods_per_year),
            "information_ratio": information_ratio(prices, reference_prices, periods_per_year),
        }
    
    # Agent-specific metrics if action log provided
    if action_log is not None:
        report["agent_profitability"] = agent_profitability(action_log, prices)
        report["win_rates"] = win_rate(action_log, prices)
        report["profit_factors"] = profit_factor(action_log, prices)
        report["expectancy"] = trade_expectancy(action_log, prices)
        
        if include_transaction_costs:
            report["transaction_costs"] = apply_transaction_costs(
                action_log, prices, fixed_cost, proportional_cost
            )
    
    return report


def format_performance_report(report: Dict[str, object], title: str = "Performance Report") -> str:
    """Format performance report as human-readable string.
    
    Args:
        report: Report dictionary from generate_performance_report
        title: Title for the report
        
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"{title:^70}")
    lines.append("=" * 70)
    lines.append("")
    
    # Returns Section
    if "returns" in report:
        lines.append("RETURN METRICS")
        lines.append("-" * 70)
        r = report["returns"]
        lines.append(f"  Total Return:        {r['total_return']:>10.2%}")
        lines.append(f"  Mean Return:         {r['mean_return']:>10.4%}")
        lines.append(f"  Std Deviation:       {r['std_return']:>10.4%}")
        lines.append(f"  Min Return:          {r['min_return']:>10.4%}")
        lines.append(f"  Max Return:          {r['max_return']:>10.4%}")
        lines.append(f"  Skewness:            {r['skewness']:>10.4f}")
        lines.append(f"  Kurtosis:            {r['kurtosis']:>10.4f}")
        lines.append("")
    
    # Risk-Adjusted Returns
    if "risk_adjusted" in report:
        lines.append("RISK-ADJUSTED RETURNS")
        lines.append("-" * 70)
        ra = report["risk_adjusted"]
        lines.append(f"  Sharpe Ratio:        {ra['sharpe_ratio']:>10.4f}")
        lines.append(f"  Sortino Ratio:       {ra['sortino_ratio']:>10.4f}")
        lines.append(f"  Calmar Ratio:        {ra['calmar_ratio']:>10.4f}")
        lines.append("")
    
    # Drawdown Analysis
    if "drawdown" in report:
        lines.append("DRAWDOWN ANALYSIS")
        lines.append("-" * 70)
        dd = report["drawdown"]
        lines.append(f"  Max Drawdown:        {dd['max_drawdown']:>10.2%}")
        lines.append(f"  Avg Drawdown:        {dd['avg_drawdown']:>10.2%}")
        lines.append(f"  Max DD Duration:     {dd['max_drawdown_duration']:>10d} periods")
        lines.append(f"  Current Drawdown:    {dd['current_drawdown']:>10.2%}")
        lines.append("")
    
    # Risk Metrics
    if "risk" in report:
        lines.append("RISK METRICS")
        lines.append("-" * 70)
        risk = report["risk"]
        lines.append(f"  VaR (95%):           {risk['value_at_risk_95']:>10.4%}")
        lines.append(f"  CVaR (95%):          {risk['cvar_95']:>10.4%}")
        lines.append(f"  Volatility (Ann.):   {risk['volatility_annualized']:>10.4%}")
        lines.append("")
    
    # Comparison Metrics
    if "comparison" in report:
        lines.append("COMPARISON TO BENCHMARK")
        lines.append("-" * 70)
        comp = report["comparison"]
        lines.append(f"  Directional Acc:     {comp['directional_accuracy']:>10.2%}")
        lines.append(f"  Beta:                {comp['beta']:>10.4f}")
        lines.append(f"  Tracking Error:      {comp['tracking_error']:>10.4%}")
        lines.append(f"  Information Ratio:   {comp['information_ratio']:>10.4f}")
        lines.append("")
    
    # Agent Metrics (if DataFrames are present, show summary)
    if "agent_profitability" in report:
        lines.append("AGENT PERFORMANCE SUMMARY")
        lines.append("-" * 70)
        lines.append(report["agent_profitability"].to_string(index=False))
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)


def compare_agents_report(
    action_log: ActionLog,
    price_series: Sequence[float],
    risk_free_rate: float = 0.02,
) -> pd.DataFrame:
    """Generate comparative report across all agents.
    
    Args:
        action_log: Agent action log
        price_series: Price series
        risk_free_rate: Risk-free rate for Sharpe calculation
        
    Returns:
        DataFrame with comparative metrics for all agents
    """
    if len(price_series) < 2:
        raise ValueError("Price series must contain at least two values.")
    
    # Get basic profitability
    prof_df = agent_profitability(action_log, price_series)
    
    # Get win rates
    wr_df = win_rate(action_log, price_series)
    
    # Get profit factors
    pf_df = profit_factor(action_log, price_series)
    
    # Get expectancy
    exp_df = trade_expectancy(action_log, price_series)
    
    # Merge all metrics
    result = prof_df.merge(wr_df, on="agent")
    result = result.merge(pf_df[["agent", "profit_factor"]], on="agent")
    result = result.merge(exp_df[["agent", "expectancy", "avg_win", "avg_loss"]], on="agent")
    
    # Sort by PnL
    result = result.sort_values("pnl", ascending=False)
    
    return result


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================


def prepare_equity_curve_data(prices: Sequence[float]) -> Dict[str, np.ndarray]:
    """Prepare data for equity curve visualization.
    
    Args:
        prices: Price series
        
    Returns:
        Dictionary with arrays for plotting: prices, cummax, drawdowns
    """
    prices_arr = np.array(prices, dtype=float)
    cummax = np.maximum.accumulate(prices_arr)
    drawdowns = drawdown_series(prices)
    
    return {
        "prices": prices_arr,
        "cummax": cummax,
        "drawdowns": drawdowns,
        "indices": np.arange(len(prices_arr)),
    }


def prepare_returns_distribution_data(prices: Sequence[float], bins: int = 50) -> Dict[str, object]:
    """Prepare data for returns distribution visualization.
    
    Args:
        prices: Price series
        bins: Number of histogram bins
        
    Returns:
        Dictionary with histogram data and normal fit parameters
    """
    returns = compute_returns(prices)
    if len(returns) == 0:
        return {"returns": np.array([]), "bins": bins}
    
    hist, edges = np.histogram(returns, bins=bins, density=True)
    
    # Fit normal distribution
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    # Generate normal curve points
    x = np.linspace(returns.min(), returns.max(), 100)
    normal_curve = stats.norm.pdf(x, mu, sigma)
    
    return {
        "returns": returns,
        "hist": hist,
        "edges": edges,
        "bins": bins,
        "mu": mu,
        "sigma": sigma,
        "x_normal": x,
        "y_normal": normal_curve,
    }


def prepare_rolling_metric_data(
    prices: Sequence[float],
    window: int = 20,
    metric: str = "sharpe",
    risk_free_rate: float = 0.02,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, np.ndarray]:
    """Prepare data for rolling metric visualization.
    
    Args:
        prices: Price series
        window: Rolling window size
        metric: Metric to compute ("sharpe", "volatility", "return")
        risk_free_rate: Risk-free rate for Sharpe
        periods_per_year: Periods per year
        
    Returns:
        Dictionary with rolling metric values
    """
    if len(prices) < window + 1:
        return {"indices": np.array([]), "values": np.array([])}
    
    prices_arr = np.array(prices, dtype=float)
    rolling_values = []
    
    for i in range(window, len(prices_arr)):
        window_prices = prices_arr[i - window : i + 1]
        
        if metric == "sharpe":
            value = sharpe_ratio(window_prices, risk_free_rate, periods_per_year)
        elif metric == "volatility":
            returns = compute_returns(window_prices)
            value = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
        elif metric == "return":
            value = (window_prices[-1] / window_prices[0] - 1)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        rolling_values.append(value)
    
    indices = np.arange(window, len(prices_arr))
    
    return {
        "indices": indices,
        "values": np.array(rolling_values),
        "window": window,
        "metric": metric,
    }


def prepare_monthly_returns_data(
    prices: Sequence[float],
    dates: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Prepare monthly returns data for heatmap.
    
    Args:
        prices: Price series
        dates: Optional date sequence (if None, uses integer months)
        
    Returns:
        DataFrame with months as columns and years as rows
    """
    returns = compute_returns(prices)
    
    if dates is None:
        # Create synthetic monthly periods
        n_months = max(1, len(prices) // 20)  # Assume ~20 periods per month
        monthly_returns = []
        for i in range(n_months):
            start = i * 20
            end = min((i + 1) * 20, len(prices))
            if end > start:
                period_return = (prices[end - 1] / prices[start] - 1) if prices[start] != 0 else 0
                monthly_returns.append(period_return)
        
        # Create DataFrame
        n_years = max(1, n_months // 12 + 1)
        data = {}
        for month in range(1, 13):
            data[month] = [monthly_returns[y * 12 + month - 1] if y * 12 + month - 1 < len(monthly_returns) else np.nan 
                          for y in range(n_years)]
        
        df = pd.DataFrame(data)
        df.index = [f"Year {y+1}" for y in range(n_years)]
    else:
        # Use actual dates
        df = pd.DataFrame({"date": dates[1:], "return": returns})
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df = df.pivot_table(values="return", index="year", columns="month")
    
    return df




