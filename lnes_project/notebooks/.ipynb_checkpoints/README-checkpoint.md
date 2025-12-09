# MacroMind Analysis Notebooks

This directory contains Jupyter notebooks for interactive analysis, exploration, and visualization of the Latent News Event Simulation (LNES) system.

## 📚 Notebooks Overview

### 01_main_analysis.ipynb
**Complete End-to-End Pipeline**

Demonstrates the full LNES workflow using the small curated dataset:
- Data loading and exploration
- News embedding (TF-IDF or transformers)
- Clustering analysis with visualization
- Agent initialization and simulation
- Performance metrics and evaluation
- Comprehensive visualizations

**Recommended for**: First-time users, understanding the system

---

### 02_fnspid_analysis.ipynb
**Real-World Dataset Validation**

Analyzes the FNSPID (Financial News and Stock Price Integration Dataset) for AAPL:
- Exploratory data analysis of real financial news
- Time series analysis of actual stock prices
- News frequency and sentiment distribution
- System validation on real-world data
- Comparison with small dataset results

**Recommended for**: Validating system performance, real-world applicability

---

### 03_agent_comparison.ipynb
**Systematic Agent Evaluation**

Comprehensive comparison of all agent types:
- Performance metrics table (PnL, win rate, directional accuracy)
- Statistical significance testing
- Decision pattern analysis
- Correlation matrix (agent diversity)
- Risk-adjusted performance metrics

**Recommended for**: Agent selection, strategy analysis

---

### 04_ai_agents_analysis.ipynb
**AI Agents Deep Dive**

In-depth analysis of AI-powered agents (FinBERT, Groq):
- FinBERT sentiment analysis examples
- Groq LLM decision reasoning
- AI vs rule-based comparison
- Cost-benefit analysis
- Failure case analysis
- Usage recommendations

**Recommended for**: Understanding AI agents, cost-benefit evaluation

---

### 05_sensitivity_analysis.ipynb
**Hyperparameter Optimization**

Explores impact of key hyperparameters:
- Number of clusters (K) sensitivity
- Price impact (alpha) sensitivity
- Noise level sensitivity
- Embedding backend comparison
- Optimal configuration recommendations

**Recommended for**: System tuning, robustness analysis

---

### 06_results_summary.ipynb
**Publication-Ready Summary**

Aggregates all results with publication-quality outputs:
- Summary tables and figures
- Hypothesis testing results
- Key findings and insights
- Limitations discussion
- Future work suggestions
- Export figures for papers

**Recommended for**: Final results, paper writing

---

## 🚀 Getting Started

### 1. Installation

Install notebook dependencies:

```bash
# From the lnes_project directory
pip install -r requirements-notebooks.txt
```

This installs:
- Jupyter notebook
- Plotting libraries (plotly, seaborn)
- Dimensionality reduction (umap-learn)
- Configuration management (pyyaml)

### 2. Launch Jupyter

```bash
cd notebooks
jupyter notebook
```

### 3. Run Notebooks

Open any notebook and run cells sequentially. The `notebook_utils.py` module is automatically loaded, providing:
- Common imports (pandas, numpy, matplotlib, etc.)
- Helper functions for experiments
- Plotting utilities
- Result loading/saving

---

## 📁 File Structure

```
notebooks/
├── README.md                          # This file
├── notebook_utils.py                  # Helper functions and utilities
├── 01_main_analysis.ipynb            # Main pipeline walkthrough
├── 02_fnspid_analysis.ipynb          # FNSPID dataset analysis
├── 03_agent_comparison.ipynb         # Agent performance comparison
├── 04_ai_agents_analysis.ipynb       # AI agents deep dive
├── 05_sensitivity_analysis.ipynb     # Hyperparameter sensitivity
└── 06_results_summary.ipynb          # Publication-ready summary
```

---

## 🔧 Configuration

Notebooks use YAML configuration files from `../config/`:

- **`default_config.yaml`**: Default parameters
- **`small_dataset.yaml`**: Small dataset experiment
- **`fnspid_aapl.yaml`**: FNSPID with AAPL
- **`ai_agents.yaml`**: AI agents comparison

Load configurations in notebooks:

```python
from notebook_utils import load_config

# Load specific configuration
config = load_config("small_dataset")

# Override parameters
config = load_config("small_dataset", overrides={"simulator.alpha": 0.02})
```

---

## 📊 Sample Workflow

### Quick Start (5 minutes)

1. Open `01_main_analysis.ipynb`
2. Run all cells (Kernel → Restart & Run All)
3. Review results and visualizations

### Complete Analysis (30 minutes)

1. **Exploratory**: Run `01_main_analysis.ipynb`
2. **Validation**: Run `02_fnspid_analysis.ipynb`
3. **Comparison**: Run `03_agent_comparison.ipynb`
4. **Summary**: Run `06_results_summary.ipynb`

### Research Workflow

1. **Baseline**: Establish results with default configs
2. **Tuning**: Use `05_sensitivity_analysis.ipynb` to optimize
3. **AI Integration**: Explore `04_ai_agents_analysis.ipynb`
4. **Publication**: Generate figures with `06_results_summary.ipynb`

---

## 💾 Results Caching

Notebooks support result caching to avoid re-computation:

```python
from src.result_cache import save_results, load_results

# Save results
cache_key = save_results(results, config)

# Load results
results, config = load_results(cache_key)

# List cached results
cache = ResultCache()
cached_list = cache.list_cached_results()
```

Results are saved to `../results/` with metadata (timestamp, git commit, config).

---

## 🎨 Plotting

### Export Figures

```python
from notebook_utils import export_figure

# After creating a matplotlib figure
export_figure(fig, "my_figure.png", plots_dir="../plots", dpi=300)
```

Figures are saved to `../plots/` for use in papers.

### Plotting Style

All notebooks use consistent plotting style:
- Seaborn theme
- Publication-quality fonts
- Consistent color palette for agents

---

## 🛠 Troubleshooting

### Issue: "Module not found: notebook_utils"

**Solution**: Ensure you're running notebooks from the `notebooks/` directory, or the path setup in `notebook_utils.py` handles it automatically.

### Issue: "Configuration file not found"

**Solution**: Check that config files exist in `../config/`. Create them if missing using examples.

### Issue: "FNSPID data not found"

**Solution**: Download FNSPID data first:
```bash
cd ../scripts
python download_fnspid.py
```

### Issue: "FinBERT/Groq not available"

**Solution**: 
- FinBERT: `pip install transformers torch`
- Groq: Set `GROQ_API_KEY` environment variable

---

## 📖 Additional Resources

### Documentation

- **Main README**: `../README.md` - Project overview
- **Architecture**: `../docs/ARCHITECTURE.md` - System design
- **Experimental Design**: `../docs/experimental_design.md` - Methodology
- **Reproducibility**: `../docs/REPRODUCIBILITY.md` - Reproduction guide
- **Contributing**: `../CONTRIBUTING.md` - Extension guide

### Example Experiments

```python
# Quick experiment with default config
from notebook_utils import quick_experiment

results = quick_experiment("small_dataset", verbose=True)

# Access results
print(f"Directional Accuracy: {results['metrics']['directional_accuracy']:.2%}")
print(f"Agents: {list(results['action_log'].keys())}")
```

### Visualization Utilities

```python
from notebook_utils import (
    plot_agent_comparison,      # Compare all agents
    plot_cluster_analysis,       # Visualize clusters (t-SNE/UMAP)
    create_summary_table,        # Generate summary table
)

# Use in notebooks
fig = plot_agent_comparison(action_log, prices)
plt.show()
```

---

## 🤝 Contributing

To add new notebooks:

1. Create notebook: `07_my_analysis.ipynb`
2. Import utilities: `from notebook_utils import *`
3. Document clearly with markdown cells
4. Add to this README
5. Test from fresh kernel

---

## 📝 Best Practices

### Notebook Organization

- **Start with overview**: Explain notebook goal and contents
- **Table of contents**: Link to major sections
- **Clear sections**: Use markdown headers
- **Inline comments**: Explain non-obvious code
- **Takeaways**: Summarize findings at end

### Code Quality

- **Use utilities**: Leverage `notebook_utils.py`
- **Config-driven**: Load parameters from YAML
- **Cache results**: Save expensive computations
- **Error handling**: Use try-except for optional features

### Reproducibility

- **Fix seeds**: Set `np.random.seed(42)`
- **Document versions**: Note package versions
- **Clear outputs**: Clear before committing (avoid large diffs)
- **Relative paths**: Use Path objects for portability

---

## 🎯 Quick Reference

| Task | Code |
|------|------|
| Load config | `config = load_config("small_dataset")` |
| Run experiment | `results = quick_experiment("small_dataset")` |
| Save results | `cache_key = save_results(results, config)` |
| Load results | `results, config = load_results(cache_key)` |
| Plot agents | `plot_agent_comparison(action_log, prices)` |
| Export figure | `export_figure(fig, "name.png", dpi=300)` |

---

## 📬 Support

- **Issues**: File on GitHub issue tracker
- **Questions**: See `../docs/ARCHITECTURE.md` for technical details
- **Extensions**: See `../CONTRIBUTING.md` for contribution guide

---

**Happy Analyzing! 📊🚀**
