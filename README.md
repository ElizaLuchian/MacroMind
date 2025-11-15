# MacroMind

### *An LLM-Driven Market Simulation Framework Powered by News, Policy Signals, and Economic Indicators*

## 📌 Overview
MacroMind is a research-oriented simulation framework that models how a synthetic financial market evolves in response to real-world news, macroeconomic indicators, and policy events, with decision-making powered by a single centralized Large Language Model (LLM).

Instead of relying on rule-based or statistical market models, MacroMind explores whether an LLM can act as a market oracle, interpreting external information and generating agent-level or market-level actions across simulation time.

This project is designed as a research study combined with a bachelor thesis.

## 🎯 Objectives
- Model a simplified economy driven by LLM-interpreted external information.
- Convert heterogeneous datasets into a unified market state vector.
- Generate agent actions and market signals using a single LLM call per tick.
- Run many simulation iterations for statistical evaluation.
- Evaluate whether LLM-driven decision systems can approximate real-world market behavior.

## 🧱 Key Features
- **Centralized LLM reasoning**
- **Real-world data integration**
- **Unified Market State Vector**
- **Configurable agent types**
- **Optimized for 50–200 runs within a $500 Azure budget**

## 🏗️ Project Structure
```
MacroMind/
├── data/
├── src/
│   ├── simulation/
│   ├── agents/
│   ├── preprocessing/
│   ├── llm_interface/
│   └── utils/
├── experiments/
├── notebooks/
└── README.md
```
