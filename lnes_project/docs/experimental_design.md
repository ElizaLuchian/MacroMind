# MacroMind Experimental Modeling

This note formalizes the experimental layer specified in the project brief and clarifies how we validate results and compare the approach to existing work.

---

## 1. Data Sources & Variables

- **News dataset** (`small_news.csv`, extensible to external corpora): daily records with fields  
  `date`, `headline`, `body`, `sentiment_hint`.
- **Market data** (`small_prices.csv` or external OHLCV feeds): `date`, OHLC, `volume`.

Derived variables:

- Clean textual tokens \\(x_i\\) per day \\(i\\).
- Embeddings \\(e_i = f_\theta(x_i)\\) using SentenceTransformer or TF-IDF.
- Cluster labels \\(c_i = g_k(e_i)\\) via the KMeans operator \\(g_k\\).
- Agent actions \\(a_i^{(j)} \in \{-1, 0, 1\}\\) (sell, hold, buy) for agent \\(j\\).
- Net order flow \\(\phi_i = \sum_j a_i^{(j)}\\).
- Simulated close \\(\hat{p}_i\\).

---

## 2. Experimental Model

### 2.1 Embedding and Clustering

1. Text preprocessing \\(\tilde{x}_i = \text{clean}(x_i)\\).
2. Embeddings  
   \\[
   e_i = f_\theta(\tilde{x}_i), \quad f_\theta \in \{\text{MiniLM}, \text{FinBERT}, \text{TF-IDF}\}
   \\]
3. Clustering  
   \\[
   c_i = \arg\min_{c\in\{1,\dots,k\}} \| e_i - \mu_c \|_2^2
   \\]

### 2.2 Agent Decisions

Let \\(\Delta p_i = p_i - p_{i-1}\\). Then

- Momentum: \\(a_i^{(\text{mom})} = \text{sign}(\Delta p_i)\\)
- Contrarian: \\(a_i^{(\text{con})} = -a_i^{(\text{mom})}\\)
- Random: \\(a_i^{(\text{rnd})} \sim \text{Uniform}\{-1,0,1\}\\)
- News reactive:  
  \\[
  a_i^{(\text{news})} =
  \begin{cases}
  1 & s_{c_i} > \tau \\
  -1 & s_{c_i} < -\tau \\
  0 & \text{otherwise}
  \end{cases}
  \\]
  where \\(s_{c} = \mathbb{E}[\text{sentiment} \mid c]\\).

### 2.3 Price Update

\\[
\hat{p}_{i+1} = \hat{p}_i + \alpha \phi_i + \varepsilon_i, \qquad
\varepsilon_i \sim \mathcal{N}(0, \sigma^2)
\\]

This forms the simulation engine documented in `src/simulator.py`.

---

## 3. Validation Plan

| Objective | Metric | Baseline |
|-----------|--------|----------|
| Track price direction | Directional accuracy vs actual close | Momentum-only agent, random walk |
| Measure trading benefit | Agent P&L vs zero baseline | Random agent, buy-and-hold |
| Check news linkage | Pearson correlation between cluster IDs and price changes | Clustering on raw TF-IDF vs advanced embeddings |
| Volatility structure | Autocorr of squared returns | Actual market volatility |
| Decision diversity | Correlation matrix of agent actions | Ideally low cross-correlation unless news drives convergence |

To assert benefits vs literature:

1. Compare accuracy/P&L when embeddings use MiniLM vs TF-IDF (ablation).  
2. Compare Alpha-driven simulator vs baseline random-walk using the same agent signals.  
3. Introduce benchmark agent definitions from cited agent-based finance papers and evaluate on shared metrics.

---

## 4. Case Study (Initial Dataset)

Steps already implemented:

1. Curate ~100 paired news/price rows (January–April 2024).  
2. Run full pipeline (`run_small_dataset_experiment`).  
3. Collect metrics + decision logs + plots via `scripts/run_experiment.py`.  
4. Interpret results for reports: highlight cluster distribution, agent profitability, and when news-reactive agent outperforms contrarian/momentum.

Outputs (examples):

- `SimulationResult.to_frame()` – time series table for annexes.
- Metric tables – inserted into research write-ups to showcase method potential.

---

## 5. Roadmap for Real-World Validation

1. **Dataset selection**  
   - News: Reuters TRC2, Bloomberg ESG, or FinBERT benchmark corpora.  
   - Prices: S&P 500 constituents, FX pair data (e.g., EUR/USD) from Quandl or WRDS.  
   - Goal: choose sources used in existing studies to enable like-for-like comparisons.

2. **Pipeline integration**  
   - Update `load_news`/`load_prices` paths or add connectors for APIs.  
   - Use the same preprocessing + embedding modules; re-fit `NewsReactiveAgent` sentiment map using either FinBERT sentiment or external labels.

3. **Evaluation**  
   - Produce metrics over rolling windows.  
   - Compare against published baselines: e.g., event-study abnormal returns, standard agent-based simulations, or LSTM sentiment models.  
   - Document statistical significance (paired tests on accuracy/P&L).

4. **Reproducibility**  
   - Version datasets or provide download scripts.  
   - Capture experiment configs in YAML/JSON for reproducibility.

---

## 6. Repository & Provenance

- Git repository (`lnes_project`) contains modular code + docs.  
- Tests cover pipeline components; CI-ready via `python -m pytest`.  
- Documentation: `README`, `PROJECT_REPORT`, and this design note.  
- Future work: tag releases, maintain commit logs, attach experiment artifacts (plots, CSV outputs).

This satisfies the requirement for transparent progress tracking and forms the basis for research dissemination.


