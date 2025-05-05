# Credit‑Risk Modelling Toolkit

This mini‑package turns a raw CSV snapshot of start‑up / funding data into

1. an engineered **training table** (`load_dataset`)
2. a calibrated **probability‑of‑default model** (`train`)
3. a fully scored portfolio with credit‑policy decisions, drift & fairness analytics, and diagnostic plots (`score`)

Everything else in the code base is an internal helper.

---

## Installation

```bash
# clone repo, then:
pip install -r requirements.txt
```

## Quick-Start
| Function                                                 | What it does                                                                                                                       | Essential I/O                                                                                 |
| -------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **`load_dataset(csv, cfg)`**                             | Parses the raw snapshot, repairs dates, adds funding momentum, joins macro & governance indicators, URL heuristics, etc.           | **in** CSV path + `BusinessFrame` config<br>**out** `pandas.DataFrame` (\~350 features)       |
| **`train(csv, model_out, cfg_path, model_type='xgb')`**  | Full training loop → tuned, calibrated model bundle + SHAP plots, ROC/PR data, calibration band, optimal threshold & rating edges. | **in** CSV, output file/dir, JSON/YAML config<br>**out** `.joblib` bundle + PNG/CSV artefacts |
| **`score(model_path, csv_in, csv_out, plots_dir=None)`** | Reloads a bundle, scores new data, tags credit decisions, checks drift/fairness, saves scored CSV + dashboard plots.               | **in** model bundle, CSV, output path<br>**out** scored CSV (+ optional plot folder)          |

## Examples

### load_dataset
<pre> ```python from credit_risk import load_dataset, read_cfg

cfg = read_cfg(None)                 # built‑in defaults
df  = load_dataset("snapshot.csv", cfg)
print(df.shape)                      # (rows, engineered features)
 ``` </pre>

 ### train a model
<pre> ```python from credit_risk import train
train(
    csv="snapshot.csv",
    model_out="models/xgb_model.joblib",
    cfg_path="config.json",          # or None
    model_type="xgb"                 # or "rf"
)                    # (rows, engineered features)
 ``` </pre>

 ### score new
<pre> ```python from credit_risk import score
score(
    model_path="models/xgb_model.joblib",
    csv_in="new_apps.csv",
    csv_out="new_apps_scored.csv",
    plots_dir="reports/2025‑05‑05"   # optional
)
 ``` </pre>