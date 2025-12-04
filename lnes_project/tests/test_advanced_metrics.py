"""Tests for advanced financial metrics."""

import numpy as np
import pandas as pd
import pytest

from src import metrics


class TestRiskAdjustedReturns:
    """Tests for risk-adjusted return metrics."""
    
    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio with positive trending prices."""
        prices = [100, 102, 105, 103, 108, 110]
        sharpe = metrics.sharpe_ratio(prices, risk_free_rate=0.0)
        assert sharpe > 0, "Sharpe ratio should be positive for upward trend"
    
    def test_sharpe_ratio_negative_returns(self):
        """Test Sharpe ratio with negative trending prices."""
        prices = [100, 98, 95, 97, 92, 90]
        sharpe = metrics.sharpe_ratio(prices, risk_free_rate=0.0)
        assert sharpe < 0, "Sharpe ratio should be negative for downward trend"
    
    def test_sharpe_ratio_constant_prices(self):
        """Test Sharpe ratio with constant prices (no volatility)."""
        prices = [100] * 10
        sharpe = metrics.sharpe_ratio(prices, risk_free_rate=0.0)
        assert sharpe == 0.0, "Sharpe ratio should be zero for constant prices"
    
    def test_sharpe_ratio_single_price(self):
        """Test Sharpe ratio with insufficient data."""
        prices = [100]
        sharpe = metrics.sharpe_ratio(prices)
        assert sharpe == 0.0, "Should return 0 for single price"
    
    def test_sortino_ratio_positive(self):
        """Test Sortino ratio with positive returns."""
        prices = [100, 102, 101, 105, 104, 108]
        sortino = metrics.sortino_ratio(prices, risk_free_rate=0.0)
        assert sortino > 0, "Sortino ratio should be positive"
    
    def test_sortino_ratio_no_downside(self):
        """Test Sortino ratio when all returns are positive."""
        prices = [100, 101, 102, 103, 104, 105]
        sortino = metrics.sortino_ratio(prices, risk_free_rate=0.0)
        # Should be very high (theoretically infinite)
        assert sortino > 0, "Sortino ratio should be positive with no downside"
    
    def test_calmar_ratio(self):
        """Test Calmar ratio calculation."""
        prices = [100, 105, 103, 110, 108, 115]
        calmar = metrics.calmar_ratio(prices)
        assert isinstance(calmar, float), "Should return float"
        # With positive returns and some drawdown, should be positive
        assert calmar > 0, "Calmar ratio should be positive for upward trend"
    
    def test_calmar_ratio_no_drawdown(self):
        """Test Calmar ratio with no drawdown."""
        prices = [100, 101, 102, 103, 104, 105]
        calmar = metrics.calmar_ratio(prices)
        # Should be very high (infinite) with no drawdown
        assert calmar > 0, "Calmar ratio should be positive"
    
    def test_information_ratio(self):
        """Test information ratio vs benchmark."""
        portfolio = [100, 103, 105, 107, 110]
        benchmark = [100, 101, 102, 103, 104]
        ir = metrics.information_ratio(portfolio, benchmark)
        assert ir > 0, "IR should be positive when portfolio outperforms"
    
    def test_information_ratio_same_performance(self):
        """Test IR when portfolio matches benchmark."""
        prices = [100, 102, 105, 103, 108]
        ir = metrics.information_ratio(prices, prices)
        assert abs(ir) < 0.01, "IR should be near zero for identical series"


class TestDrawdownAnalysis:
    """Tests for drawdown metrics."""
    
    def test_maximum_drawdown_declining(self):
        """Test max drawdown with declining prices."""
        prices = [100, 90, 80, 70]
        max_dd = metrics.maximum_drawdown(prices)
        assert max_dd < 0, "Max drawdown should be negative"
        assert max_dd == pytest.approx(-0.3, rel=1e-5), "Should be 30% drawdown"
    
    def test_maximum_drawdown_recovery(self):
        """Test max drawdown with price recovery."""
        prices = [100, 120, 80, 110]
        max_dd = metrics.maximum_drawdown(prices)
        # Max drawdown should be from 120 to 80
        expected = (80 - 120) / 120
        assert max_dd == pytest.approx(expected, rel=1e-5)
    
    def test_maximum_drawdown_no_decline(self):
        """Test max drawdown with only increasing prices."""
        prices = [100, 105, 110, 115, 120]
        max_dd = metrics.maximum_drawdown(prices)
        assert max_dd == 0.0, "No drawdown with only increasing prices"
    
    def test_drawdown_series(self):
        """Test drawdown series calculation."""
        prices = [100, 110, 105, 115, 110]
        dd = metrics.drawdown_series(prices)
        assert len(dd) == len(prices), "Drawdown series should match price length"
        assert dd[0] == 0.0, "First drawdown should be zero"
        assert dd[1] == 0.0, "Peak has zero drawdown"
    
    def test_average_drawdown(self):
        """Test average drawdown calculation."""
        prices = [100, 90, 95, 85, 90]
        avg_dd = metrics.average_drawdown(prices)
        assert avg_dd < 0, "Average drawdown should be negative"
    
    def test_drawdown_details(self):
        """Test comprehensive drawdown details."""
        prices = [100, 110, 90, 95, 85, 100]
        details = metrics.drawdown_details(prices)
        
        assert "max_drawdown" in details
        assert "avg_drawdown" in details
        assert "max_drawdown_duration" in details
        assert "current_drawdown" in details
        
        assert details["max_drawdown"] < 0
        assert isinstance(details["max_drawdown_duration"], int)
    
    def test_underwater_periods(self):
        """Test underwater period identification."""
        prices = [100, 110, 105, 115, 110, 120]
        periods = metrics.underwater_periods(prices)
        assert isinstance(periods, list), "Should return list of periods"


class TestTradingPerformance:
    """Tests for trading performance metrics."""
    
    @pytest.fixture
    def sample_action_log(self):
        """Sample action log for testing."""
        return {
            "momentum": ["buy", "buy", "sell", "buy", "hold"],
            "contrarian": ["sell", "hold", "buy", "sell", "buy"],
        }
    
    @pytest.fixture
    def sample_prices(self):
        """Sample price series."""
        return [100, 102, 101, 105, 103, 108]
    
    def test_win_rate(self, sample_action_log, sample_prices):
        """Test win rate calculation."""
        df = metrics.win_rate(sample_action_log, sample_prices)
        
        assert "agent" in df.columns
        assert "win_rate" in df.columns
        assert "trades" in df.columns
        assert "wins" in df.columns
        assert "losses" in df.columns
        
        assert len(df) == 2, "Should have metrics for both agents"
        assert all(0 <= wr <= 1 for wr in df["win_rate"]), "Win rate should be 0-1"
    
    def test_profit_factor(self, sample_action_log, sample_prices):
        """Test profit factor calculation."""
        df = metrics.profit_factor(sample_action_log, sample_prices)
        
        assert "agent" in df.columns
        assert "profit_factor" in df.columns
        assert "gross_profits" in df.columns
        assert "gross_losses" in df.columns
        
        assert len(df) == 2
    
    def test_trade_expectancy(self, sample_action_log, sample_prices):
        """Test trade expectancy calculation."""
        df = metrics.trade_expectancy(sample_action_log, sample_prices)
        
        assert "agent" in df.columns
        assert "expectancy" in df.columns
        assert "avg_win" in df.columns
        assert "avg_loss" in df.columns
        
        assert len(df) == 2
    
    def test_all_hold_agent(self, sample_prices):
        """Test metrics with agent that never trades."""
        action_log = {"holder": ["hold"] * (len(sample_prices) - 1)}
        
        df = metrics.win_rate(action_log, sample_prices)
        assert df.loc[0, "trades"] == 0
        assert df.loc[0, "win_rate"] == 0.0


class TestRiskMetrics:
    """Tests for risk metrics."""
    
    def test_value_at_risk_historical(self):
        """Test historical VaR calculation."""
        # Create returns with known distribution
        np.random.seed(42)
        prices = [100]
        for _ in range(100):
            ret = np.random.normal(0.001, 0.02)
            prices.append(prices[-1] * (1 + ret))
        
        var = metrics.value_at_risk(prices, confidence=0.95, method="historical")
        assert var >= 0, "VaR should be non-negative"
        assert var < 1, "VaR should be less than 100%"
    
    def test_value_at_risk_parametric(self):
        """Test parametric VaR calculation."""
        np.random.seed(42)
        prices = [100]
        for _ in range(100):
            ret = np.random.normal(0.001, 0.02)
            prices.append(prices[-1] * (1 + ret))
        
        var = metrics.value_at_risk(prices, confidence=0.95, method="parametric")
        assert var >= 0, "VaR should be non-negative"
    
    def test_conditional_value_at_risk(self):
        """Test CVaR calculation."""
        np.random.seed(42)
        prices = [100]
        for _ in range(100):
            ret = np.random.normal(0.0, 0.02)
            prices.append(prices[-1] * (1 + ret))
        
        cvar = metrics.conditional_value_at_risk(prices, confidence=0.95)
        var = metrics.value_at_risk(prices, confidence=0.95, method="historical")
        
        assert cvar >= var, "CVaR should be >= VaR"
    
    def test_beta_to_market(self):
        """Test beta calculation."""
        # Create correlated price series
        np.random.seed(42)
        market = [100]
        for _ in range(50):
            ret = np.random.normal(0.001, 0.02)
            market.append(market[-1] * (1 + ret))
        
        # Portfolio with beta ~1.5
        portfolio = [100]
        for i in range(50):
            market_ret = (market[i+1] - market[i]) / market[i]
            port_ret = 1.5 * market_ret + np.random.normal(0, 0.01)
            portfolio.append(portfolio[-1] * (1 + port_ret))
        
        beta = metrics.beta_to_market(portfolio, market)
        # Beta should be close to 1.5 but with noise
        assert 0.5 < beta < 3.0, f"Beta {beta} seems unreasonable"
    
    def test_tracking_error(self):
        """Test tracking error calculation."""
        portfolio = [100, 102, 105, 103, 108, 110]
        benchmark = [100, 101, 104, 102, 107, 109]
        
        te = metrics.tracking_error(portfolio, benchmark)
        assert te >= 0, "Tracking error should be non-negative"
    
    def test_tracking_error_identical(self):
        """Test tracking error with identical series."""
        prices = [100, 102, 105, 103, 108]
        te = metrics.tracking_error(prices, prices)
        assert te == pytest.approx(0.0, abs=1e-10), "TE should be zero for identical series"


class TestTransactionCosts:
    """Tests for transaction cost modeling."""
    
    @pytest.fixture
    def sample_action_log(self):
        return {
            "agent1": ["buy", "hold", "sell", "buy", "hold"],
            "agent2": ["hold", "buy", "hold", "hold", "sell"],
        }
    
    @pytest.fixture
    def sample_prices(self):
        return [100, 102, 101, 105, 103, 108]
    
    def test_apply_transaction_costs_fixed(self, sample_action_log, sample_prices):
        """Test fixed transaction costs."""
        df = metrics.apply_transaction_costs(
            sample_action_log, sample_prices, fixed_cost=1.0
        )
        
        assert "agent" in df.columns
        assert "gross_pnl" in df.columns
        assert "transaction_costs" in df.columns
        assert "net_pnl" in df.columns
        assert "num_trades" in df.columns
        
        # Net PnL should be less than gross PnL
        for _, row in df.iterrows():
            if row["num_trades"] > 0:
                assert row["net_pnl"] < row["gross_pnl"]
    
    def test_apply_transaction_costs_proportional(self, sample_action_log, sample_prices):
        """Test proportional transaction costs."""
        df = metrics.apply_transaction_costs(
            sample_action_log, sample_prices, proportional_cost=0.001
        )
        
        assert len(df) == 2
        assert all(df["net_pnl"] <= df["gross_pnl"])
    
    def test_apply_transaction_costs_slippage(self, sample_action_log, sample_prices):
        """Test slippage costs."""
        df = metrics.apply_transaction_costs(
            sample_action_log, sample_prices, slippage_bps=5.0
        )
        
        assert len(df) == 2
        assert all(df["transaction_costs"] >= 0)
    
    def test_market_impact_cost(self):
        """Test market impact calculation."""
        impact = metrics.market_impact_cost(
            trade_size=1000, daily_volume=100000, impact_coefficient=0.1
        )
        assert impact > 0, "Market impact should be positive"
        assert impact < 1, "Impact should be reasonable"
    
    def test_market_impact_zero_volume(self):
        """Test market impact with zero volume."""
        impact = metrics.market_impact_cost(
            trade_size=1000, daily_volume=0, impact_coefficient=0.1
        )
        assert impact == 0.0, "Should return 0 for zero volume"


class TestPerformanceReporting:
    """Tests for performance reporting functions."""
    
    def test_generate_performance_report(self):
        """Test comprehensive performance report generation."""
        prices = [100, 105, 103, 110, 108, 115, 113, 120]
        
        report = metrics.generate_performance_report(prices)
        
        # Check all sections exist
        assert "returns" in report
        assert "risk_adjusted" in report
        assert "drawdown" in report
        assert "risk" in report
        
        # Check specific metrics
        assert "sharpe_ratio" in report["risk_adjusted"]
        assert "max_drawdown" in report["drawdown"]
        assert "value_at_risk_95" in report["risk"]
    
    def test_generate_performance_report_with_reference(self):
        """Test report with reference prices."""
        prices = [100, 105, 103, 110, 108]
        reference = [100, 104, 102, 109, 107]
        
        report = metrics.generate_performance_report(
            prices, reference_prices=reference
        )
        
        assert "comparison" in report
        assert "directional_accuracy" in report["comparison"]
        assert "beta" in report["comparison"]
    
    def test_generate_performance_report_with_agents(self):
        """Test report with agent data."""
        prices = [100, 105, 103, 110, 108, 115]
        action_log = {
            "agent1": ["buy", "hold", "sell", "buy", "hold"],
            "agent2": ["sell", "buy", "hold", "sell", "buy"],
        }
        
        report = metrics.generate_performance_report(
            prices, action_log=action_log
        )
        
        assert "agent_profitability" in report
        assert "win_rates" in report
        assert "profit_factors" in report
    
    def test_format_performance_report(self):
        """Test report formatting."""
        prices = [100, 105, 103, 110, 108, 115]
        report = metrics.generate_performance_report(prices)
        
        formatted = metrics.format_performance_report(report)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "Performance Report" in formatted
        assert "RETURN METRICS" in formatted
    
    def test_compare_agents_report(self):
        """Test agent comparison report."""
        prices = [100, 105, 103, 110, 108, 115]
        action_log = {
            "agent1": ["buy", "hold", "sell", "buy", "hold"],
            "agent2": ["sell", "buy", "hold", "sell", "buy"],
        }
        
        df = metrics.compare_agents_report(action_log, prices)
        
        assert len(df) == 2
        assert "agent" in df.columns
        assert "pnl" in df.columns
        assert "win_rate" in df.columns
        assert "profit_factor" in df.columns


class TestVisualizationDataPrep:
    """Tests for visualization data preparation functions."""
    
    def test_prepare_equity_curve_data(self):
        """Test equity curve data preparation."""
        prices = [100, 110, 105, 115, 110, 120]
        data = metrics.prepare_equity_curve_data(prices)
        
        assert "prices" in data
        assert "cummax" in data
        assert "drawdowns" in data
        assert "indices" in data
        
        assert len(data["prices"]) == len(prices)
        assert len(data["cummax"]) == len(prices)
    
    def test_prepare_returns_distribution_data(self):
        """Test returns distribution data preparation."""
        prices = [100, 105, 103, 110, 108, 115, 113, 120]
        data = metrics.prepare_returns_distribution_data(prices, bins=10)
        
        assert "returns" in data
        assert "mu" in data
        assert "sigma" in data
        assert "x_normal" in data
        assert "y_normal" in data
    
    def test_prepare_rolling_metric_data_sharpe(self):
        """Test rolling Sharpe ratio data preparation."""
        prices = [100 + i * 0.5 for i in range(50)]
        data = metrics.prepare_rolling_metric_data(prices, window=10, metric="sharpe")
        
        assert "indices" in data
        assert "values" in data
        assert "window" in data
        assert data["window"] == 10
        assert len(data["values"]) > 0
    
    def test_prepare_rolling_metric_data_volatility(self):
        """Test rolling volatility data preparation."""
        prices = [100 + i * 0.5 for i in range(50)]
        data = metrics.prepare_rolling_metric_data(prices, window=10, metric="volatility")
        
        assert len(data["values"]) > 0
        assert all(v >= 0 for v in data["values"]), "Volatility should be non-negative"
    
    def test_prepare_monthly_returns_data(self):
        """Test monthly returns data preparation."""
        # Create 2 years of daily data
        prices = [100 * (1.001 ** i) for i in range(500)]
        df = metrics.prepare_monthly_returns_data(prices)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0  # At least one year
        assert df.shape[1] > 0  # At least one month


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_prices(self):
        """Test metrics with empty price series."""
        prices = []
        
        sharpe = metrics.sharpe_ratio(prices)
        assert sharpe == 0.0
        
        max_dd = metrics.maximum_drawdown(prices)
        assert max_dd == 0.0
    
    def test_single_price(self):
        """Test metrics with single price point."""
        prices = [100]
        
        sharpe = metrics.sharpe_ratio(prices)
        assert sharpe == 0.0
        
        returns = metrics.compute_returns(prices)
        assert len(returns) == 0
    
    def test_constant_prices(self):
        """Test metrics with constant prices (no volatility)."""
        prices = [100] * 20
        
        sharpe = metrics.sharpe_ratio(prices)
        assert sharpe == 0.0
        
        max_dd = metrics.maximum_drawdown(prices)
        assert max_dd == 0.0
    
    def test_negative_prices(self):
        """Test that metrics handle negative prices gracefully."""
        # Most metrics should still work with negative prices
        prices = [100, 50, 25, 10, 5]
        
        sharpe = metrics.sharpe_ratio(prices)
        assert isinstance(sharpe, float)
        
        max_dd = metrics.maximum_drawdown(prices)
        assert max_dd < 0
    
    def test_mismatched_lengths(self):
        """Test error handling for mismatched input lengths."""
        portfolio = [100, 102, 105]
        benchmark = [100, 101]
        
        with pytest.raises(ValueError):
            metrics.beta_to_market(portfolio, benchmark)
    
    def test_zero_starting_price(self):
        """Test handling of zero starting price."""
        prices = [0, 1, 2, 3]
        report = metrics.generate_performance_report(prices)
        assert report["returns"]["total_return"] == 0.0

