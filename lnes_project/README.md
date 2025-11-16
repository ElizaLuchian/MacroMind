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

- Detailed mathematical modeling, metrics, and comparison plan are documented in `docs/experimental_design.md`.  
- Small dataset case study (`data/small_*.csv`) demonstrates the methodology end-to-end and provides interpretable plots/metrics for reports.  
- The integrated FNSPID experiment enables benchmarking against a widely cited news+price dataset (tickers, date windows, Hugging Face cache control).  
- Validation roadmap includes ablation against baselines (TF-IDF vs transformer embeddings, random vs momentum agents) and outlines how to extend to additional literature-grade datasets (LOBSTER, Reuters, etc.).

## Repository & Reproducibility

- Entire source, tests, and docs live in this git repository; commits show incremental progress.  
- Use `pytest` for regression checks before committing new experiments.  
- For real-data validation, drop replacement CSVs (or add loaders) and document runs with configuration files or notebooks referencing this codebase.

## Extending the Project

- Swap clustering algorithms by plugging DBSCAN/GMM calls into `clusterer`.
- Introduce new agents by subclassing `BaseAgent`.
- Enhance the simulator with transaction costs, liquidity, or stochastic shocks.
- Replace the sample dataset with proprietary feeds via the existing loader/preprocess APIs.

## License

This research scaffold is provided as-is for experimentation. Adapt freely within your institutional guidelines.

