# Latent News Event Simulation (MacroMind)

Modular research codebase for detecting latent news events with open embeddings and evaluating how heterogeneous trading agents respond inside a lightweight market simulator.

## Project Goals

- Process ~100 combined rows of news and prices for controlled experimentation.
- Generate text embeddings with `sentence-transformers/all-MiniLM-L6-v2` (FinBERT optional).
- Detect latent events via modular clustering (KMeans baseline, extensible to DBSCAN/GMM).
- Simulate trading agents: rule-based (momentum, contrarian) and **AI-powered** (FinBERT sentiment, Groq LLMs).
- Quantify performance with profitability, volatility, clustering correlations, and decision metrics.
- Provide repeatable pipelines, plots, and unit tests suitable for research iteration.
- Support literature-grade validation by linking to public datasets such as FNSPID.
- **NEW:** 100% FREE AI agents for sentiment analysis and LLM-based reasoning!

## Repository Layout

```
lnes_project/
├── data/                      # curated sample CSVs (~100 rows each)
├── src/                       # core modules (data, preprocessing, embeddings, agents, sim, metrics)
├── tests/                     # pytest-based unit + integration tests
├── requirements.txt           # python dependencies
└── README.md                  # project overview
```

## Pipeline Overview

1. **Data ingestion** (`data_loader`) – load, validate, and merge news/prices.
2. **Preprocessing** (`preprocess`) – normalize text and assemble combined strings.
3. **Embedding** (`embedder`) – default SentenceTransformer with TF-IDF fallback and FinBERT option.
4. **Clustering** (`clusterer`) – discover latent event regimes and visualize frequencies.
5. **Agents** (`agents`) – Random, Momentum, Contrarian, NewsReactive (cluster-aware), FinBERT (AI sentiment), and Groq (LLM reasoning) behaviors.
6. **Simulation** (`simulator`) – day-level price updates driven by aggregate order flow.
7. **Metrics** (`metrics`) – accuracy, profitability, volatility clustering, correlation diagnostics.
8. **Experiment scripts** (`experiment_smallset`, `experiment_fnspid`) – run curated or real-data studies.

## Getting Started

```bash
python -m venv .venv
.venv\Scripts\activate  # or source .venv/bin/activate
pip install -r requirements.txt
pytest
```

### 🤖 Using AI Agents (FREE!)

```bash
# Install AI dependencies
pip install transformers torch groq

# Run with FinBERT sentiment agent (100% free, runs locally)
python scripts/run_ai_experiment.py

# Run with Groq LLM agent (free tier, requires API key)
export GROQ_API_KEY="your-key"  # Get free key at console.groq.com
python scripts/run_ai_experiment.py --groq
```

See **[AI_AGENTS_GUIDE.md](AI_AGENTS_GUIDE.md)** for detailed setup and comparison of AI agents.

> 📚 **New to the project?** See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design overview.

To replicate the case study end-to-end:

```bash
python -c "from src.experiment_smallset import run_small_dataset_experiment; run_small_dataset_experiment()"
```

Pass `embedder_kwargs={'backend': 'tfidf'}` for fast CPU-only experimentation.

### Visual walkthrough (small dataset)

```bash
python scripts/run_experiment.py --backend tfidf
```

This prints headline metrics and opens Matplotlib windows for price trajectories, cluster distribution, and agent profitability.

### Viewing Agent Actions

To see what each agent does during the simulation, use these options:

```bash
# Display action log in console (first 20 rows)
python scripts/run_experiment.py --backend tfidf --show-actions

# Save full action log to CSV
python scripts/run_experiment.py --backend tfidf --save-actions-csv actions.csv

# Generate a plot showing agent actions over time
python scripts/run_experiment.py --backend tfidf --plot-actions

# Combine all options
python scripts/run_experiment.py --backend tfidf --show-actions --save-actions-csv actions.csv --plot-actions
```

The action log CSV contains:
- `date`: Trading date
- `close`: Reference price
- `simulated_close`: Simulated price
- `order_flow`: Net order flow (sum of all agent actions)
- `cluster_id`: News event cluster assignment
- `headline` or `combined_text`: News headline/content
- `action_{agent_name}`: Individual agent decisions (`buy`, `sell`, or `hold`)

### Running against the FNSPID dataset

Install the optional Hugging Face dependency (already listed in `requirements.txt`) and provide the ticker plus slicing knobs:

```bash
python scripts/run_experiment.py \
  --dataset fnspid \
  --fnspid-tickers AAPL \
  --fnspid-start-date 2023-01-01 \
  --fnspid-end-date 2023-03-31 \
  --fnspid-news-limit 5000 \
  --fnspid-price-limit 20000 \
  --backend tfidf
```

The loader (`load_fnspid`) pulls aligned news + price tables directly from Hugging Face (or optional local CSVs), filters by ticker/date, and feeds the same preprocessing, embedding, clustering, and simulation pipeline. Use `--fnspid-local-news` / `--fnspid-local-prices` if you have offline copies and `--fnspid-cache-dir` to control download caching.

## Experimental Modeling & Validation

- Detailed mathematical modeling, metrics, and comparison plan are documented in **[docs/experimental_design.md](docs/experimental_design.md)**.  
- Small dataset case study (`data/small_*.csv`) demonstrates the methodology end-to-end and provides interpretable plots/metrics for reports.  
- The integrated FNSPID experiment enables benchmarking against a widely cited news+price dataset (tickers, date windows, Hugging Face cache control).  
- Validation roadmap includes ablation against baselines (TF-IDF vs transformer embeddings, random vs momentum agents) and outlines how to extend to additional literature-grade datasets (LOBSTER, Reuters, etc.).

## Repository & Reproducibility

- Entire source, tests, and docs live in this git repository; commits show incremental progress.  
- Use `pytest` for regression checks before committing new experiments.  
- For real-data validation, drop replacement CSVs (or add loaders) and document runs with configuration files or notebooks referencing this codebase.
- For step-by-step reproduction instructions, see **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**.

## Extending the Project

- Swap clustering algorithms by plugging DBSCAN/GMM calls into `clusterer`.
- Introduce new agents by subclassing `BaseAgent`.
- Enhance the simulator with transaction costs, liquidity, or stochastic shocks.
- Replace the sample dataset with proprietary feeds via the existing loader/preprocess APIs.

For detailed extension guidelines, see **[CONTRIBUTING.md](CONTRIBUTING.md)**.

## Interactive Analysis Notebooks

The `notebooks/` directory contains Jupyter notebooks for interactive exploration, analysis, and visualization:

- **[01_main_analysis.ipynb](notebooks/01_main_analysis.ipynb)** - Complete end-to-end pipeline walkthrough with small dataset
- **[02_fnspid_analysis.ipynb](notebooks/02_fnspid_analysis.ipynb)** - Real-world validation using FNSPID (AAPL stock)
- **[03_agent_comparison.ipynb](notebooks/03_agent_comparison.ipynb)** - Systematic comparison of all agent types
- **[04_ai_agents_analysis.ipynb](notebooks/04_ai_agents_analysis.ipynb)** - Deep dive into FinBERT and Groq AI agents
- **[05_sensitivity_analysis.ipynb](notebooks/05_sensitivity_analysis.ipynb)** - Hyperparameter sensitivity and optimization
- **[06_results_summary.ipynb](notebooks/06_results_summary.ipynb)** - Publication-ready figures and summary tables

### Quick Start with Notebooks

```bash
# Install notebook dependencies
pip install -r requirements-notebooks.txt

# Launch Jupyter
cd notebooks
jupyter notebook
```

See **[notebooks/README.md](notebooks/README.md)** for detailed notebook documentation and usage guide.

### Configuration Management

Experiments can be configured using YAML files in `config/`:

```python
from src.config_loader import load_config

# Load predefined configuration
config = load_config("small_dataset")

# Override specific parameters
config = load_config("fnspid_aapl", overrides={"simulator.alpha": 0.02})
```

Available configurations:
- `default_config.yaml` - Default parameters for all experiments
- `small_dataset.yaml` - Small curated dataset experiment
- `fnspid_aapl.yaml` - FNSPID with AAPL stock
- `ai_agents.yaml` - AI agents comparison experiment

### Results Caching

Save and load experiment results for faster re-analysis:

```python
from src.result_cache import save_results, load_results

# Save results
cache_key = save_results(results, config)

# Load cached results
results, config = load_results(cache_key)
```

Results are stored in `results/` with metadata including timestamp, git commit, and configuration.

## Advanced Visualization Suite

The project includes a comprehensive visualization system with 50+ plot functions, interactive dashboard, and publication-ready figure generation.

### Interactive Dashboard

Launch a web-based dashboard for real-time experiment exploration:

```bash
streamlit run src/dashboard.py
```

**Features**:
- 📊 6 tabbed views (Overview, Performance, Agents, Embeddings, Time Series, Data)
- ⚙️ Configuration panel with parameter controls
- 🤖 Real-time experiment execution
- 📈 Interactive plots with zoom/pan
- 📥 CSV data export

### Gallery Generation

Generate complete visualization galleries with one command:

```bash
python scripts/generate_all_plots.py --config small_dataset --output-dir output/plots
```

**Generates**:
- 16+ plots in PNG and PDF formats
- HTML index page with navigation
- Metrics reports and agent comparisons
- Organized by category (Performance, Agents, Embeddings)

### Publication Figures

Create journal-ready figures optimized for academic papers:

```bash
python scripts/create_publication_figures.py --config small_dataset --output-dir paper/figures
```

**Outputs**:
- 4 multi-panel figures with subfigure labels
- LaTeX-compatible fonts and sizing
- Vector formats (PDF, SVG)
- Grayscale-friendly and colorblind-safe
- Supplementary metrics table

### Available Visualizations

**Performance** (6 types):
- Equity curves with drawdown shading
- Underwater plots
- Returns distribution with Q-Q plots
- Rolling Sharpe ratio and volatility

**Agent Analysis** (6 types):
- P&L and win rate comparisons
- Action heatmaps and frequency charts
- Correlation network graphs
- Rolling P&L evolution

**Embeddings** (3 types):
- t-SNE and UMAP projections
- Silhouette analysis for cluster quality

**Time Series & Comparative** (4 types):
- Order flow waterfalls
- Regime detection
- Radar charts for multi-metric comparison
- Correlation matrices

See **[docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md)** for complete usage guide and API reference.

## Documentation

Comprehensive documentation is available:

- **[AI_AGENTS_GUIDE.md](AI_AGENTS_GUIDE.md)** - Setup and usage for FinBERT and Groq AI agents
- **[docs/experimental_design.md](docs/experimental_design.md)** - Formal research methodology and mathematical formulation
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System architecture and design patterns
- **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)** - Step-by-step reproduction guide
- **[docs/VISUALIZATION_GUIDE.md](docs/VISUALIZATION_GUIDE.md)** - Complete visualization reference with 50+ plot functions
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to extend the project (agents, metrics, datasets)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{macromind2024,
  title={MacroMind: Latent News Event Simulation},
  author={MacroMind Research Team},
  year={2024},
  url={https://github.com/YOUR_USERNAME/MacroMind}
}
```

For AI agents, also cite:
- **FinBERT**: Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models. arXiv:1908.10063.
- **Groq**: https://groq.com

## Acknowledgments

- **FNSPID Dataset**: Zihan1004/FNSPID on Hugging Face
- **SentenceTransformers**: UKPLab for pre-trained models
- **FinBERT**: ProsusAI for financial sentiment model
- **Groq**: Fast LLM inference platform

## License

This research scaffold is provided as-is for experimentation. Adapt freely within your institutional guidelines.

