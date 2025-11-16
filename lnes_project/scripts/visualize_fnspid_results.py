"""Create comprehensive visualizations of FNSPID simulation results."""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_comprehensive_view(actions_file: Path, prices_file: Path, ticker: str):
    """Create a comprehensive view of prices and agent decisions."""
    
    # Load data
    actions_df = pd.read_csv(actions_file)
    prices_df = pd.read_csv(prices_file)
    
    actions_df['date'] = pd.to_datetime(actions_df['date'])
    prices_df['date'] = pd.to_datetime(prices_df['date'])
    
    # Get action columns
    action_cols = [col for col in actions_df.columns if col.startswith("action_")]
    
    # Create figure with subplots
    n_agents = len(action_cols)
    fig, axes = plt.subplots(n_agents + 1, 1, figsize=(16, 3 * (n_agents + 1)), sharex=True)
    
    # Map actions to numeric
    action_map = {"buy": 1, "sell": -1, "hold": 0}
    colors = {"buy": "#55A868", "sell": "#DD8452", "hold": "#4C72B0"}
    
    # Plot 1: Price comparison at the top
    ax_price = axes[0]
    
    # Plot real prices
    ax_price.plot(prices_df['date'], prices_df['close'], 
                  label=f'Real {ticker} Price', color='#2E86AB', linewidth=2, alpha=0.8)
    
    # Plot simulated prices
    ax_price.plot(actions_df['date'], actions_df['simulated_close'],
                  label='Simulated Price', color='#A23B72', linewidth=2, linestyle='--', alpha=0.8)
    
    ax_price.set_ylabel('Price ($)', fontsize=12, fontweight='bold')
    ax_price.set_title(f'{ticker} Price Movements & Agent Decisions (FNSPID Dataset)', 
                       fontsize=16, fontweight='bold')
    ax_price.legend(loc='upper left', fontsize=11)
    ax_price.grid(True, alpha=0.3)
    
    # Plot order flow as background
    ax_flow = ax_price.twinx()
    ax_flow.fill_between(actions_df['date'], 0, actions_df['order_flow'],
                          where=(actions_df['order_flow'] > 0), color='green', alpha=0.1, label='Net Buy')
    ax_flow.fill_between(actions_df['date'], 0, actions_df['order_flow'],
                          where=(actions_df['order_flow'] < 0), color='red', alpha=0.1, label='Net Sell')
    ax_flow.set_ylabel('Order Flow', fontsize=10, alpha=0.7)
    ax_flow.tick_params(axis='y', labelsize=9, labelcolor='gray')
    
    # Plot each agent's decisions
    for idx, col in enumerate(action_cols):
        ax = axes[idx + 1]
        agent_name = col.replace("action_", "").replace("_", " ").title()
        
        # Get actions
        actions_numeric = actions_df[col].map(action_map)
        point_colors = [colors.get(action, "#808080") for action in actions_df[col]]
        
        # Scatter plot with colors
        for action, color in colors.items():
            mask = actions_df[col] == action
            if mask.any():
                ax.scatter(actions_df.loc[mask, 'date'], 
                          actions_numeric[mask],
                          c=color, s=40, alpha=0.7, 
                          edgecolors='black', linewidth=0.5,
                          label=action.upper())
        
        # Connect with line
        ax.plot(actions_df['date'], actions_numeric, 
                linewidth=1, alpha=0.3, color='gray', zorder=1)
        
        # Styling
        ax.set_ylabel(agent_name, rotation=0, ha="right", va="center", 
                     fontsize=11, fontweight='bold', labelpad=40)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["SELL", "HOLD", "BUY"], fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        
        # Shading
        ax.axhspan(-1.5, -0.5, alpha=0.03, color='red')
        ax.axhspan(0.5, 1.5, alpha=0.03, color='green')
        ax.set_ylim(-1.5, 1.5)
        
        # Legend only on first agent
        if idx == 0:
            ax.legend(loc='upper right', fontsize=9, ncol=3)
    
    # X-axis label
    axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
    
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    
    return fig


def plot_agent_comparison(actions_file: Path):
    """Plot side-by-side comparison of agent strategies."""
    
    actions_df = pd.read_csv(actions_file)
    actions_df['date'] = pd.to_datetime(actions_df['date'])
    
    action_cols = [col for col in actions_df.columns if col.startswith("action_")]
    
    # Count action types for each agent
    fig, axes = plt.subplots(1, len(action_cols), figsize=(4 * len(action_cols), 5))
    if len(action_cols) == 1:
        axes = [axes]
    
    colors = {"buy": "#55A868", "sell": "#DD8452", "hold": "#4C72B0"}
    
    for idx, col in enumerate(action_cols):
        ax = axes[idx]
        agent_name = col.replace("action_", "").replace("_", " ").title()
        
        # Count each action type
        action_counts = actions_df[col].value_counts()
        
        # Plot pie chart
        wedges, texts, autotexts = ax.pie(
            action_counts.values,
            labels=action_counts.index,
            colors=[colors.get(action, "#808080") for action in action_counts.index],
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'}
        )
        
        ax.set_title(agent_name, fontsize=13, fontweight='bold', pad=20)
    
    fig.suptitle('Agent Decision Distribution', fontsize=16, fontweight='bold', y=0.98)
    fig.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize FNSPID simulation results")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker")
    parser.add_argument("--actions-file", type=Path, default=None, help="Actions CSV file")
    parser.add_argument("--prices-file", type=Path, default=None, help="Prices CSV file")
    
    args = parser.parse_args()
    
    # Set defaults
    if args.actions_file is None:
        args.actions_file = Path(f"fnspid_{args.ticker}_actions.csv")
    
    if args.prices_file is None:
        args.prices_file = Path(__file__).resolve().parents[1] / "data" / f"fnspid_{args.ticker}_prices.csv"
    
    if not args.actions_file.exists():
        print(f"Error: Actions file not found: {args.actions_file}")
        print(f"Run: python scripts/run_fnspid_experiment.py --ticker {args.ticker}")
        return 1
    
    if not args.prices_file.exists():
        print(f"Error: Prices file not found: {args.prices_file}")
        return 1
    
    print("=" * 70)
    print(f"Creating Comprehensive Visualizations for {args.ticker}")
    print("=" * 70)
    
    # Create comprehensive view
    fig1 = plot_comprehensive_view(args.actions_file, args.prices_file, args.ticker)
    
    # Create agent comparison
    fig2 = plot_agent_comparison(args.actions_file)
    
    plt.show()
    
    print("\nVisualizations complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())

