"""
Configuration Management System for MacroMind (LNES) Experiments

This module provides utilities for loading, validating, and merging
experiment configurations from YAML files with command-line overrides.

Author: MacroMind Research Team
Date: December 2024
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import copy


class ConfigLoader:
    """
    Load and manage experiment configurations.
    
    Features:
    - Load YAML configuration files
    - Merge with default configuration
    - Support command-line overrides
    - Validate configuration structure
    - Provide default values
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize ConfigLoader.
        
        Parameters
        ----------
        config_dir : str, optional
            Directory containing configuration files.
            Defaults to 'config/' relative to project root.
        """
        if config_dir is None:
            # Assume we're in src/ and config/ is at project root
            project_root = Path(__file__).parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self.default_config_path = self.config_dir / "default_config.yaml"
        
    def load(self, config_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Parameters
        ----------
        config_name : str, optional
            Name of configuration file (without .yaml extension).
            If None, loads only default configuration.
            Examples: "small_dataset", "fnspid_aapl", "ai_agents"
        
        Returns
        -------
        config : dict
            Complete configuration dictionary
        """
        # Load default configuration
        config = self._load_yaml(self.default_config_path)
        
        # If specific config provided, merge with default
        if config_name is not None:
            config_path = self.config_dir / f"{config_name}.yaml"
            if config_path.exists():
                specific_config = self._load_yaml(config_path)
                config = self._merge_configs(config, specific_config)
            else:
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        return config
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Parameters
        ----------
        base : dict
            Base configuration (typically default_config)
        override : dict
            Override configuration (specific experiment config)
        
        Returns
        -------
        merged : dict
            Merged configuration with overrides applied
        """
        merged = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                merged[key] = self._merge_configs(merged[key], value)
            else:
                # Override value
                merged[key] = value
        
        return merged
    
    def override_from_dict(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply command-line style overrides to configuration.
        
        Parameters
        ----------
        config : dict
            Base configuration
        overrides : dict
            Dictionary of overrides using dot notation keys
            Example: {"dataset.type": "fnspid", "simulator.alpha": 0.02}
        
        Returns
        -------
        config : dict
            Configuration with overrides applied
        """
        config = copy.deepcopy(config)
        
        for key, value in overrides.items():
            # Split dot notation: "dataset.type" -> ["dataset", "type"]
            keys = key.split('.')
            
            # Navigate to the nested location
            target = config
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            # Set the value
            target[keys[-1]] = value
        
        return config
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration structure.
        
        Parameters
        ----------
        config : dict
            Configuration to validate
        
        Returns
        -------
        valid : bool
            True if configuration is valid
        
        Raises
        ------
        ValueError
            If configuration is invalid
        """
        required_sections = [
            'experiment',
            'dataset',
            'embedder',
            'clustering',
            'agents',
            'simulator',
            'metrics',
            'visualization',
            'output'
        ]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate dataset type
        if config['dataset']['type'] not in ['small', 'fnspid']:
            raise ValueError(f"Invalid dataset type: {config['dataset']['type']}")
        
        # Validate embedder backend
        valid_backends = ['tfidf', 'sentence-transformer', 'finbert']
        if config['embedder']['backend'] not in valid_backends:
            raise ValueError(f"Invalid embedder backend: {config['embedder']['backend']}")
        
        # Validate clustering algorithm
        valid_algorithms = ['kmeans', 'dbscan', 'hierarchical']
        if config['clustering']['algorithm'] not in valid_algorithms:
            raise ValueError(f"Invalid clustering algorithm: {config['clustering']['algorithm']}")
        
        # Validate agents
        valid_agents = ['Random', 'Momentum', 'Contrarian', 'NewsReactive', 'FinBERT', 'Groq']
        for agent in config['agents']['enabled']:
            if agent not in valid_agents:
                raise ValueError(f"Invalid agent type: {agent}")
        
        return True
    
    def get_param(self, config: Dict[str, Any], key: str, default: Any = None) -> Any:
        """
        Get parameter from configuration using dot notation.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary
        key : str
            Parameter key in dot notation (e.g., "simulator.alpha")
        default : any, optional
            Default value if key not found
        
        Returns
        -------
        value : any
            Parameter value
        """
        keys = key.split('.')
        value = config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def list_available_configs(self) -> List[str]:
        """
        List all available configuration files.
        
        Returns
        -------
        configs : list of str
            List of configuration names (without .yaml extension)
        """
        if not self.config_dir.exists():
            return []
        
        configs = []
        for path in self.config_dir.glob("*.yaml"):
            if path.stem != "default_config":
                configs.append(path.stem)
        
        return sorted(configs)
    
    def print_config(self, config: Dict[str, Any], indent: int = 0) -> None:
        """
        Pretty print configuration.
        
        Parameters
        ----------
        config : dict
            Configuration to print
        indent : int, optional
            Indentation level
        """
        for key, value in config.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key}:")
                self.print_config(value, indent + 1)
            else:
                print("  " * indent + f"{key}: {value}")


def load_config(config_name: Optional[str] = None, 
                config_dir: Optional[str] = None,
                overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Parameters
    ----------
    config_name : str, optional
        Name of configuration file to load
    config_dir : str, optional
        Directory containing configuration files
    overrides : dict, optional
        Dictionary of parameter overrides
    
    Returns
    -------
    config : dict
        Complete configuration dictionary
    
    Examples
    --------
    >>> # Load default configuration
    >>> config = load_config()
    
    >>> # Load specific configuration
    >>> config = load_config("small_dataset")
    
    >>> # Load with overrides
    >>> config = load_config("fnspid_aapl", overrides={"simulator.alpha": 0.02})
    """
    loader = ConfigLoader(config_dir)
    config = loader.load(config_name)
    
    if overrides is not None:
        config = loader.override_from_dict(config, overrides)
    
    # Validate configuration
    loader.validate(config)
    
    return config


def get_config_param(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Convenience function to get configuration parameter.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary
    key : str
        Parameter key in dot notation
    default : any, optional
        Default value if not found
    
    Returns
    -------
    value : any
        Parameter value
    
    Examples
    --------
    >>> config = load_config("small_dataset")
    >>> alpha = get_config_param(config, "simulator.alpha", 0.01)
    >>> k = get_config_param(config, "clustering.k", 3)
    """
    loader = ConfigLoader()
    return loader.get_param(config, key, default)


if __name__ == "__main__":
    # Example usage
    print("=== Available Configurations ===")
    loader = ConfigLoader()
    configs = loader.list_available_configs()
    print(f"Found {len(configs)} configurations:")
    for cfg in configs:
        print(f"  - {cfg}")
    
    print("\n=== Loading Default Configuration ===")
    config = load_config()
    loader.print_config(config)
    
    print("\n=== Loading Small Dataset Configuration ===")
    config = load_config("small_dataset")
    print(f"Experiment name: {config['experiment']['name']}")
    print(f"Dataset type: {config['dataset']['type']}")
    print(f"Enabled agents: {config['agents']['enabled']}")
    
    print("\n=== Testing Overrides ===")
    config = load_config("small_dataset", overrides={"simulator.alpha": 0.05})
    print(f"Overridden alpha: {config['simulator']['alpha']}")

