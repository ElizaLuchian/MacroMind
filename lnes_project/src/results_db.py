"""Results database for storing and querying experimental results.

This module provides a SQLite-based database for storing experiment configurations,
performance metrics, and statistical test results.

Classes:
    ResultsDB: Main database interface
    Experiment: Dataclass for experiment records
"""

from __future__ import annotations

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import sqlite3
import json
import warnings

import pandas as pd


@dataclass
class ExperimentRecord:
    """Record for a single experiment run."""
    experiment_id: Optional[int] = None
    timestamp: Optional[str] = None
    experiment_type: str = "default"
    agent_name: str = "Unknown"
    dataset: str = "unknown"
    config: str = "{}"  # JSON string
    performance_metrics: str = "{}"  # JSON string
    notes: str = ""


class ResultsDB:
    """Database for storing experimental results.
    
    Usage:
        >>> db = ResultsDB("experiments.db")
        >>> experiment_id = db.add_experiment(
        ...     agent_name="Momentum",
        ...     dataset="FNSPID_AAPL",
        ...     config={"alpha": 0.01, "k_clusters": 3},
        ...     performance={"sharpe": 1.5, "return": 0.08}
        ... )
        >>> results = db.query_experiments(agent_name="Momentum")
        >>> db.close()
    """
    
    def __init__(self, db_path: str = "experiments.db"):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables if they don't exist."""
        # Experiments table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                experiment_type TEXT,
                agent_name TEXT NOT NULL,
                dataset TEXT NOT NULL,
                config TEXT,
                performance_metrics TEXT,
                notes TEXT
            )
        ''')
        
        # Comparison results table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                experiment_id_1 INTEGER,
                experiment_id_2 INTEGER,
                test_type TEXT,
                test_statistic REAL,
                p_value REAL,
                effect_size REAL,
                conclusion TEXT,
                FOREIGN KEY (experiment_id_1) REFERENCES experiments(experiment_id),
                FOREIGN KEY (experiment_id_2) REFERENCES experiments(experiment_id)
            )
        ''')
        
        # Meta-analysis table
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS meta_analyses (
                meta_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                analysis_name TEXT,
                pooled_effect REAL,
                ci_lower REAL,
                ci_upper REAL,
                p_value REAL,
                heterogeneity REAL,
                n_studies INTEGER,
                notes TEXT
            )
        ''')
        
        self.conn.commit()
    
    def add_experiment(
        self,
        agent_name: str,
        dataset: str,
        config: Dict[str, Any],
        performance: Dict[str, float],
        experiment_type: str = "default",
        notes: str = "",
    ) -> int:
        """Add experiment result to database.
        
        Args:
            agent_name: Name of the agent
            dataset: Dataset name
            config: Configuration dictionary
            performance: Performance metrics dictionary
            experiment_type: Type of experiment
            notes: Optional notes
            
        Returns:
            experiment_id of the inserted record
        """
        timestamp = datetime.now().isoformat()
        config_json = json.dumps(config)
        performance_json = json.dumps(performance)
        
        self.cursor.execute('''
            INSERT INTO experiments (timestamp, experiment_type, agent_name, dataset, 
                                    config, performance_metrics, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, experiment_type, agent_name, dataset, config_json, 
              performance_json, notes))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def add_comparison(
        self,
        experiment_id_1: int,
        experiment_id_2: int,
        test_type: str,
        test_statistic: float,
        p_value: float,
        effect_size: float,
        conclusion: str = "",
    ) -> int:
        """Add comparison result to database.
        
        Args:
            experiment_id_1: First experiment ID
            experiment_id_2: Second experiment ID
            test_type: Type of statistical test
            test_statistic: Test statistic value
            p_value: P-value
            effect_size: Effect size (e.g., Cohen's d)
            conclusion: Text conclusion
            
        Returns:
            comparison_id of the inserted record
        """
        timestamp = datetime.now().isoformat()
        
        self.cursor.execute('''
            INSERT INTO comparisons (timestamp, experiment_id_1, experiment_id_2,
                                   test_type, test_statistic, p_value, effect_size, conclusion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, experiment_id_1, experiment_id_2, test_type, 
              test_statistic, p_value, effect_size, conclusion))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def add_meta_analysis(
        self,
        analysis_name: str,
        pooled_effect: float,
        ci_lower: float,
        ci_upper: float,
        p_value: float,
        heterogeneity: float,
        n_studies: int,
        notes: str = "",
    ) -> int:
        """Add meta-analysis result to database.
        
        Args:
            analysis_name: Name of the meta-analysis
            pooled_effect: Pooled effect size
            ci_lower: Lower CI bound
            ci_upper: Upper CI bound
            p_value: P-value
            heterogeneity: I² statistic
            n_studies: Number of studies included
            notes: Optional notes
            
        Returns:
            meta_id of the inserted record
        """
        timestamp = datetime.now().isoformat()
        
        self.cursor.execute('''
            INSERT INTO meta_analyses (timestamp, analysis_name, pooled_effect,
                                      ci_lower, ci_upper, p_value, heterogeneity,
                                      n_studies, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, analysis_name, pooled_effect, ci_lower, ci_upper,
              p_value, heterogeneity, n_studies, notes))
        
        self.conn.commit()
        return self.cursor.lastrowid
    
    def query_experiments(
        self,
        agent_name: Optional[str] = None,
        dataset: Optional[str] = None,
        experiment_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Query experiments from database.
        
        Args:
            agent_name: Filter by agent name
            dataset: Filter by dataset
            experiment_type: Filter by experiment type
            limit: Maximum number of results
            
        Returns:
            DataFrame with matching experiments
        """
        query = "SELECT * FROM experiments WHERE 1=1"
        params = []
        
        if agent_name is not None:
            query += " AND agent_name = ?"
            params.append(agent_name)
        
        if dataset is not None:
            query += " AND dataset = ?"
            params.append(dataset)
        
        if experiment_type is not None:
            query += " AND experiment_type = ?"
            params.append(experiment_type)
        
        query += " ORDER BY timestamp DESC"
        
        if limit is not None:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, self.conn, params=params)
        
        # Parse JSON columns
        if len(df) > 0:
            df['config'] = df['config'].apply(json.loads)
            df['performance_metrics'] = df['performance_metrics'].apply(json.loads)
        
        return df
    
    def query_comparisons(
        self,
        experiment_id: Optional[int] = None,
        significant_only: bool = False,
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Query comparison results from database.
        
        Args:
            experiment_id: Filter by experiment ID (either exp1 or exp2)
            significant_only: Only return significant results
            alpha: Significance level
            
        Returns:
            DataFrame with matching comparisons
        """
        query = "SELECT * FROM comparisons WHERE 1=1"
        params = []
        
        if experiment_id is not None:
            query += " AND (experiment_id_1 = ? OR experiment_id_2 = ?)"
            params.extend([experiment_id, experiment_id])
        
        if significant_only:
            query += f" AND p_value < {alpha}"
        
        query += " ORDER BY timestamp DESC"
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    def get_experiment_by_id(self, experiment_id: int) -> Dict[str, Any]:
        """Get a single experiment by ID.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Dictionary with experiment data
        """
        self.cursor.execute("SELECT * FROM experiments WHERE experiment_id = ?", 
                          (experiment_id,))
        row = self.cursor.fetchone()
        
        if row is None:
            raise ValueError(f"No experiment found with ID {experiment_id}")
        
        columns = [desc[0] for desc in self.cursor.description]
        result = dict(zip(columns, row))
        result['config'] = json.loads(result['config'])
        result['performance_metrics'] = json.loads(result['performance_metrics'])
        
        return result
    
    def get_best_experiments(
        self,
        metric: str = "sharpe_ratio",
        n: int = 10,
        agent_name: Optional[str] = None,
        dataset: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get top N experiments by performance metric.
        
        Args:
            metric: Metric name to rank by
            n: Number of top experiments to return
            agent_name: Optional filter by agent
            dataset: Optional filter by dataset
            
        Returns:
            DataFrame with top experiments
        """
        df = self.query_experiments(agent_name=agent_name, dataset=dataset)
        
        if len(df) == 0:
            return df
        
        # Extract metric from performance_metrics JSON
        df['metric_value'] = df['performance_metrics'].apply(
            lambda x: x.get(metric, float('-inf'))
        )
        
        # Sort and return top N
        df = df.sort_values('metric_value', ascending=False).head(n)
        
        return df
    
    def delete_experiment(self, experiment_id: int):
        """Delete an experiment and related comparisons.
        
        Args:
            experiment_id: Experiment ID to delete
        """
        # Delete related comparisons
        self.cursor.execute('''
            DELETE FROM comparisons 
            WHERE experiment_id_1 = ? OR experiment_id_2 = ?
        ''', (experiment_id, experiment_id))
        
        # Delete experiment
        self.cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", 
                          (experiment_id,))
        
        self.conn.commit()
    
    def export_to_csv(self, output_path: str):
        """Export all experiments to CSV.
        
        Args:
            output_path: Path to output CSV file
        """
        df = self.query_experiments()
        
        # Flatten JSON columns for CSV export
        if len(df) > 0:
            # Expand config
            config_df = pd.json_normalize(df['config'])
            config_df.columns = ['config_' + col for col in config_df.columns]
            
            # Expand performance
            perf_df = pd.json_normalize(df['performance_metrics'])
            perf_df.columns = ['perf_' + col for col in perf_df.columns]
            
            # Combine
            df_export = pd.concat([
                df.drop(['config', 'performance_metrics'], axis=1),
                config_df,
                perf_df
            ], axis=1)
        else:
            df_export = df
        
        df_export.to_csv(output_path, index=False)
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the database.
        
        Returns:
            Dictionary with summary statistics
        """
        # Count experiments
        self.cursor.execute("SELECT COUNT(*) FROM experiments")
        n_experiments = self.cursor.fetchone()[0]
        
        # Count comparisons
        self.cursor.execute("SELECT COUNT(*) FROM comparisons")
        n_comparisons = self.cursor.fetchone()[0]
        
        # Count meta-analyses
        self.cursor.execute("SELECT COUNT(*) FROM meta_analyses")
        n_meta = self.cursor.fetchone()[0]
        
        # Unique agents
        self.cursor.execute("SELECT COUNT(DISTINCT agent_name) FROM experiments")
        n_agents = self.cursor.fetchone()[0]
        
        # Unique datasets
        self.cursor.execute("SELECT COUNT(DISTINCT dataset) FROM experiments")
        n_datasets = self.cursor.fetchone()[0]
        
        return {
            "n_experiments": n_experiments,
            "n_comparisons": n_comparisons,
            "n_meta_analyses": n_meta,
            "n_unique_agents": n_agents,
            "n_unique_datasets": n_datasets,
            "db_path": str(self.db_path),
        }
    
    def close(self):
        """Close database connection."""
        self.conn.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def create_results_summary(db: ResultsDB) -> str:
    """Create a text summary of results database.
    
    Args:
        db: ResultsDB instance
        
    Returns:
        Formatted text summary
    """
    stats = db.get_summary_statistics()
    
    lines = []
    lines.append("=" * 70)
    lines.append("RESULTS DATABASE SUMMARY")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Database: {stats['db_path']}")
    lines.append("")
    lines.append(f"Total experiments: {stats['n_experiments']}")
    lines.append(f"Total comparisons: {stats['n_comparisons']}")
    lines.append(f"Total meta-analyses: {stats['n_meta_analyses']}")
    lines.append(f"Unique agents tested: {stats['n_unique_agents']}")
    lines.append(f"Unique datasets used: {stats['n_unique_datasets']}")
    lines.append("")
    
    # Top agents
    top_agents = db.get_best_experiments(metric="sharpe_ratio", n=5)
    if len(top_agents) > 0:
        lines.append("### Top 5 Experiments (by Sharpe Ratio) ###")
        lines.append("")
        for _, row in top_agents.iterrows():
            perf = row['performance_metrics']
            sharpe = perf.get('sharpe_ratio', 'N/A')
            lines.append(f"  {row['agent_name']} on {row['dataset']}: {sharpe}")
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)

