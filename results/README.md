# Spark run results (tables, figures, submission files only — caches are not versioned)

| folder | code | design | IR (gross) | beta (NW SE) | max DD |
|---|---|---|---|---|---|
| `run2_beta_neutral/` | commit f06af0e + `alphas=20` fix in src/selection.py | fusion beta + beta target 0.075, short-side screen, per-year headline choice, FinBERT tone + event geometry | 0.70 | -0.11 (0.10) | -10.4% |
| `run1_dollar_neutral/` | commit b2aa6e2 | equal-weight headline, original beta (missing -> 1.0), no short screen | 1.30 | -0.33 (0.16) | -22.5% |

Main chart: `figures/cumulative_returns.png` (strategy vs T-bill + 4% vs S&P 500, 01/2021-08/2026).
Neutrality: `tables/neutrality_table.csv`, `figures/beta_by_year.png` (run2 only).
