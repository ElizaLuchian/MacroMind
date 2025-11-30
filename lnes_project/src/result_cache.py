"""
Result Caching System for MacroMind (LNES) Experiments

This module provides utilities for caching experiment results to avoid
re-computation, with support for metadata tracking and cache invalidation.

Author: MacroMind Research Team
Date: December 2024
"""

import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd


class ResultCache:
    """
    Manage caching of experiment results.
    
    Features:
    - Save/load experiment results
    - Generate unique cache keys from configurations
    - Track metadata (timestamp, git commit, config)
    - Support multiple serialization formats
    - List and filter cached results
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize ResultCache.
        
        Parameters
        ----------
        cache_dir : str, optional
            Directory for storing cached results.
            Defaults to 'results/' relative to project root.
        """
        if cache_dir is None:
            # Assume we're in src/ and results/ is at project root
            project_root = Path(__file__).parent.parent
            cache_dir = project_root / "results"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_cache_key(self, config: Dict[str, Any], prefix: str = "") -> str:
        """
        Generate unique cache key from configuration.
        
        Parameters
        ----------
        config : dict
            Experiment configuration
        prefix : str, optional
            Prefix for cache key (e.g., experiment name)
        
        Returns
        -------
        cache_key : str
            Unique identifier for this configuration
        """
        # Convert config to JSON string (sorted keys for consistency)
        config_str = json.dumps(config, sort_keys=True)
        
        # Generate hash
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:12]
        
        # Create cache key
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if prefix:
            cache_key = f"{prefix}_{timestamp}_{config_hash}"
        else:
            cache_key = f"experiment_{timestamp}_{config_hash}"
        
        return cache_key
    
    def save(self, 
             results: Dict[str, Any], 
             config: Dict[str, Any],
             cache_key: Optional[str] = None,
             format: str = "pickle") -> str:
        """
        Save experiment results to cache.
        
        Parameters
        ----------
        results : dict
            Experiment results to cache
        config : dict
            Configuration used for this experiment
        cache_key : str, optional
            Cache key. If None, generates automatically.
        format : str, optional
            Serialization format: "pickle" or "json"
            Default: "pickle" (supports numpy arrays, pandas DataFrames)
        
        Returns
        -------
        cache_key : str
            Cache key used for saving
        """
        if cache_key is None:
            exp_name = config.get('experiment', {}).get('name', 'experiment')
            cache_key = self.generate_cache_key(config, prefix=exp_name)
        
        # Create subdirectory for this cache key
        result_dir = self.cache_dir / cache_key
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        if format == "pickle":
            results_path = result_dir / "results.pkl"
            with open(results_path, 'wb') as f:
                pickle.dump(results, f)
        elif format == "json":
            results_path = result_dir / "results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(self._make_json_serializable(results), f, indent=2)
        else:
            raise ValueError(f"Invalid format: {format}. Use 'pickle' or 'json'.")
        
        # Save configuration
        config_path = result_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # Save metadata
        metadata = self._create_metadata(config)
        metadata_path = result_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Results cached: {cache_key}")
        return cache_key
    
    def load(self, cache_key: str, format: str = "pickle") -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load experiment results from cache.
        
        Parameters
        ----------
        cache_key : str
            Cache key to load
        format : str, optional
            Serialization format: "pickle" or "json"
        
        Returns
        -------
        results : dict
            Cached experiment results
        config : dict
            Configuration used for this experiment
        
        Raises
        ------
        FileNotFoundError
            If cache key not found
        """
        result_dir = self.cache_dir / cache_key
        
        if not result_dir.exists():
            raise FileNotFoundError(f"Cache key not found: {cache_key}")
        
        # Load results
        if format == "pickle":
            results_path = result_dir / "results.pkl"
            with open(results_path, 'rb') as f:
                results = pickle.load(f)
        elif format == "json":
            results_path = result_dir / "results.json"
            with open(results_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        else:
            raise ValueError(f"Invalid format: {format}")
        
        # Load configuration
        config_path = result_dir / "config.json"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"Results loaded: {cache_key}")
        return results, config
    
    def exists(self, cache_key: str) -> bool:
        """
        Check if cache key exists.
        
        Parameters
        ----------
        cache_key : str
            Cache key to check
        
        Returns
        -------
        exists : bool
            True if cache key exists
        """
        result_dir = self.cache_dir / cache_key
        return result_dir.exists()
    
    def list_cached_results(self, pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all cached results with metadata.
        
        Parameters
        ----------
        pattern : str, optional
            Filter by experiment name pattern
        
        Returns
        -------
        results : list of dict
            List of cached results with metadata
        """
        cached = []
        
        for result_dir in self.cache_dir.iterdir():
            if not result_dir.is_dir():
                continue
            
            metadata_path = result_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Filter by pattern if provided
            if pattern is not None:
                if pattern.lower() not in metadata['experiment_name'].lower():
                    continue
            
            cached.append({
                'cache_key': result_dir.name,
                'experiment_name': metadata['experiment_name'],
                'timestamp': metadata['timestamp'],
                'dataset_type': metadata.get('dataset_type', 'unknown'),
                'agents': metadata.get('agents', []),
            })
        
        # Sort by timestamp (most recent first)
        cached.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return cached
    
    def delete(self, cache_key: str) -> None:
        """
        Delete cached results.
        
        Parameters
        ----------
        cache_key : str
            Cache key to delete
        """
        result_dir = self.cache_dir / cache_key
        
        if not result_dir.exists():
            print(f"Cache key not found: {cache_key}")
            return
        
        # Delete all files in directory
        for file in result_dir.iterdir():
            file.unlink()
        
        # Delete directory
        result_dir.rmdir()
        
        print(f"Deleted cache: {cache_key}")
    
    def clear_cache(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cache directory.
        
        Parameters
        ----------
        older_than_days : int, optional
            Only delete results older than this many days
        
        Returns
        -------
        count : int
            Number of cache entries deleted
        """
        count = 0
        now = datetime.now()
        
        for result_dir in self.cache_dir.iterdir():
            if not result_dir.is_dir():
                continue
            
            # Check age if specified
            if older_than_days is not None:
                metadata_path = result_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    timestamp = datetime.fromisoformat(metadata['timestamp'])
                    age_days = (now - timestamp).days
                    
                    if age_days < older_than_days:
                        continue
            
            # Delete
            self.delete(result_dir.name)
            count += 1
        
        print(f"Cleared {count} cache entries")
        return count
    
    def _create_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create metadata for cached results."""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'experiment_name': config.get('experiment', {}).get('name', 'unknown'),
            'dataset_type': config.get('dataset', {}).get('type', 'unknown'),
            'agents': config.get('agents', {}).get('enabled', []),
            'embedder_backend': config.get('embedder', {}).get('backend', 'unknown'),
            'clustering_k': config.get('clustering', {}).get('k', 0),
            'random_seed': config.get('experiment', {}).get('random_seed', None),
        }
        
        # Try to get git commit (if available)
        try:
            import subprocess
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                metadata['git_commit'] = result.stdout.strip()
        except Exception:
            pass
        
        return metadata
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def save_csv(self, 
                 data: pd.DataFrame, 
                 cache_key: str,
                 filename: str) -> None:
        """
        Save DataFrame as CSV in cache directory.
        
        Parameters
        ----------
        data : pd.DataFrame
            Data to save
        cache_key : str
            Cache key
        filename : str
            CSV filename (e.g., "actions.csv")
        """
        result_dir = self.cache_dir / cache_key
        result_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = result_dir / filename
        data.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
    
    def load_csv(self, cache_key: str, filename: str) -> pd.DataFrame:
        """
        Load DataFrame from CSV in cache directory.
        
        Parameters
        ----------
        cache_key : str
            Cache key
        filename : str
            CSV filename
        
        Returns
        -------
        data : pd.DataFrame
            Loaded data
        """
        result_dir = self.cache_dir / cache_key
        csv_path = result_dir / filename
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        return pd.read_csv(csv_path)


def save_results(results: Dict[str, Any], 
                config: Dict[str, Any],
                cache_dir: Optional[str] = None) -> str:
    """
    Convenience function to save results.
    
    Parameters
    ----------
    results : dict
        Experiment results
    config : dict
        Configuration
    cache_dir : str, optional
        Cache directory
    
    Returns
    -------
    cache_key : str
        Cache key used
    
    Examples
    --------
    >>> from src.config_loader import load_config
    >>> config = load_config("small_dataset")
    >>> results = {"sim_prices": [...], "metrics": {...}}
    >>> cache_key = save_results(results, config)
    """
    cache = ResultCache(cache_dir)
    return cache.save(results, config)


def load_results(cache_key: str, 
                cache_dir: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function to load results.
    
    Parameters
    ----------
    cache_key : str
        Cache key
    cache_dir : str, optional
        Cache directory
    
    Returns
    -------
    results : dict
        Experiment results
    config : dict
        Configuration
    
    Examples
    --------
    >>> results, config = load_results("small_dataset_20241217_a1b2c3")
    """
    cache = ResultCache(cache_dir)
    return cache.load(cache_key)


if __name__ == "__main__":
    # Example usage
    print("=== ResultCache Demo ===\n")
    
    cache = ResultCache()
    
    # Example: Save dummy results
    dummy_config = {
        'experiment': {'name': 'test_experiment', 'random_seed': 42},
        'dataset': {'type': 'small'},
        'agents': {'enabled': ['Random', 'Momentum']},
        'embedder': {'backend': 'tfidf'},
        'clustering': {'k': 3},
    }
    
    dummy_results = {
        'sim_prices': [100, 102, 101, 105],
        'metrics': {'directional_accuracy': 0.65},
        'actions': [['buy', 'hold'], ['sell', 'buy']],
    }
    
    print("Saving dummy results...")
    cache_key = cache.save(dummy_results, dummy_config)
    print(f"Saved with cache_key: {cache_key}\n")
    
    # List cached results
    print("=== Cached Results ===")
    cached = cache.list_cached_results()
    for item in cached:
        print(f"  {item['cache_key']}: {item['experiment_name']} ({item['timestamp']})")
    
    print("\n=== Loading Results ===")
    loaded_results, loaded_config = cache.load(cache_key)
    print(f"Loaded sim_prices: {loaded_results['sim_prices']}")
    print(f"Loaded config: {loaded_config['experiment']['name']}")

