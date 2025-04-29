from __future__ import annotations

# --------------------------------------------------------------------------- #
# Standard-library imports
# --------------------------------------------------------------------------- #
import os
import argparse
import json
import logging
import sys
from dataclasses import dataclass,field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Sequence,Iterable
import matplotlib as plt
import requests

# --------------------------------------------------------------------------- #
# Third-party imports
# --------------------------------------------------------------------------- #
import joblib
import numpy as np
import pandas as pd
import polars as pl
import pycountry
from numpy.typing import NDArray
from optbinning import OptimalBinning
import optuna
from rapidfuzz import process, fuzz
import shap
import re
from scipy.optimize import minimize
from scipy.stats import ks_2samp
from sklearn.calibration import IsotonicRegression, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    StandardScaler,
)

try:
    from xgboost import XGBClassifier
    USE_XGB = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier as GBC  # fallback
    USE_XGB = False

try:
    from xgboost import XGBClassifier
    import xgboost as _xgb
    _XGB_OK = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier as GBC  # fallback
    _XGB_OK = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import logging


# --------------------------------------------------------------------------- #
# Globals & paths
# --------------------------------------------------------------------------- #
TODAY: date = date.today()
GLOBAL_SEED: int = 42
rng_global = np.random.default_rng(GLOBAL_SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

MACRO_DIR = Path(r"C:\Users\Romain\OneDrive - KU Leuven\Masters\MBIS\Year 2\Semester 2\Statistical Consulting\ExternalData") 
META_FILE = Path(__file__).with_name("country_meta.csv")
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
WGI_INDICATORS: Dict[str, str] = {
    "regulatory_quality": "RQ.EST",
    "rule_of_law": "RL.EST",
    "control_of_corruption": "CC.EST",
}
WGI_YEARS: List[int] = list(range(2020, 2024))  # 2020-2023 inclusive


# --------------------------------------------------------------------------- #
# External macro files (path, derive_growth)
# --------------------------------------------------------------------------- #
MACRO_FILES: Dict[str, Tuple[Path, bool]] = {
    "inflation":         (MACRO_DIR / "inflation.csv",        False),
    "gdp_growth":        (MACRO_DIR / "gdp.csv",              True),
    "unemployment_rate": (MACRO_DIR / "unemployment.csv",     False),
}
_META_FILE = Path(__file__).with_name("country_meta.csv")
# ╭──────────────────── Business‑frame & defaults ───────────────────────────╮
@dataclass
class BusinessFrame:
    horizon_months: int
    good_statuses: Tuple[str, ...]
    bad_statuses: Tuple[str, ...]
    pd_table: Dict[str, float]

    labels: Tuple[str, ...] = ("R1", "R2", "R3", "R4", "R5")
    acquired_weight: float = 0.5
    ipo_weight: float = 1.5
    cost_fn: float = 5.0
    cost_fp: float = 1.0

    use_lgd: bool = False
    rating_edges: Tuple[float, ...] | None = None  # length = len(labels)-1

    decision_matrix: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=dict
    )

    def with_edges(self, edges: Sequence[float]) -> "BusinessFrame":
        return replace(self, rating_edges=tuple(sorted(edges)))


DEFAULT_DEC_MATRIX: Dict[str, Dict[str, Dict[str, Any]]] = {
    "R1": {
        "<1M": {"action": "APPROVE", "max_limit": 1_000_000},
        ">=1M": {"action": "APPROVE", "max_limit": 5_000_000},
    },
    "R2": {
        "<1M": {"action": "COLLATERAL", "haircut": 0.2, "max_limit": 750_000},
        ">=1M": {"action": "COLLATERAL", "haircut": 0.3, "max_limit": 3_000_000},
    },
    "R3": {
        "<1M": {"action": "PRICE_UP", "spread_bps": 250},
        ">=1M": {"action": "PRICE_UP", "spread_bps": 400},
    },
    "R4": {
        "*": {"action": "DECLINE"},
    },
    "R5": {
        "*": {"action": "DECLINE"},
    },
}

_DEFAULT_CFG = BusinessFrame(
    horizon_months=24,
    good_statuses=("operating", "operation", "acquired", "ipo"),
    bad_statuses=("closed",),
    pd_table={"R1": 0.02, "R2": 0.05, "R3": 0.12, "R4": 0.20, "R5": 0.35},
)
# ╭──────────────────────────── Helpers ─────────────────────────────────────╮
MACRO_COLS = ["inflation", "gdp_growth", "unemployment_rate"]

NUM_WOE = [
    "funding_total_usd",
    "years_since_founded",
    "years_since_last_funding",
    "funding_rounds",
    "funding_per_year",
]

NUM_EXTRA = list({ 
    "funding_per_round",
    "rounds_per_year",
    "months_since_last_round",
    "avg_interround_days",
    "momentum_recency_ratio",
    "diff_founded_last_funding_days",
    "diff_between_fundings_days",
    "diff_founded_first_funding_days",
    *MACRO_COLS,
    "capital_dummy",
    "category_list_len", "domain_length","has_https","has_www","url_complexity",
    'regulatory_quality', 'rule_of_law', 'control_of_corruption', 'startup_regulation_index', 'investor_protection', 'intellectual_property_rights', 'financial_regulation_banking_supervision', 'financial_regulation_securities_regulation', 'financial_regulation_insurance_regulation'
})

CAT_COLS = [
    "country_code",
    "primary_sector",
    "subregion", 
]

MOMENTUM_COLS = [
    "funding_per_year",
    "rounds_per_year",
]

WINSOR = {"funding_total_usd": (0.01, 0.01), "funding_rounds": (0.0, 0.02)}


def read_cfg(path: str | Path | None) -> BusinessFrame:
    if path is None:
        return _DEFAULT_CFG
    with open(path) as fh:
        raw = json.load(fh) if Path(path).suffix.lower() == ".json" else __import__("yaml").safe_load(fh)
    return BusinessFrame(**raw)

def _flatten(d: Dict[str, object], parent: str = "") -> Dict[str, float]:
    """Recursively flatten a nested dict using snake_child keys."""
    items: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{parent}_{k}" if parent else k
        if isinstance(v, dict):
            items.update(_flatten(v, key))
        else:
            items[key] = v
    return items

GOV_COL_SEED: List[str] = [
'regulatory_quality', 'rule_of_law', 'control_of_corruption', 'startup_regulation_index', 'investor_protection', 'intellectual_property_rights', 'financial_regulation_banking_supervision', 'financial_regulation_securities_regulation', 'financial_regulation_insurance_regulation'
]

GOV_COLS: List[str] = GOV_COL_SEED.copy()

import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="'force_all_finite' was renamed to 'ensure_all_finite'",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="The default of observed=False is deprecated.*",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="invalid value encountered in scalar.*",
)


################################################################################
# Rating‑edge optimiser (vectorised & deterministic)                            
################################################################################

from scipy.optimize import minimize  # re‑import in case not in scope
import pandas as pd


def optimise_edges(
    y: np.ndarray,
    pd_hat: np.ndarray,
    pd_table: Dict[str, float],
    *,
    seed: int = 42,
) -> List[float]:
    """Return monotonic PD edges (R1–R5) minimising MAE to target PD table.
    """

    rng = np.random.default_rng(seed)
    tgt = np.array(list(pd_table.values()))
    q = np.percentile(pd_hat, np.linspace(5, 95, 51))
    n = len(y)
    min_obs = max(50, int(0.03 * n))

    def mae(edges_vect):
        edges_sorted = np.sort(edges_vect)
        # ensure strictly increasing
        if np.any(np.diff(edges_sorted) <= 0):
            return np.inf
        try:
            bins = pd.cut(pd_hat, [-np.inf, *edges_sorted, np.inf], labels=list(pd_table))
        except ValueError:
            return np.inf
        if (pd.Series(bins).value_counts() < min_obs).any():
            return np.inf
        emp = pd.Series(y).groupby(bins).mean().reindex(pd_table.keys()).values
        return np.abs(emp - tgt).mean()

    init_edges = np.percentile(pd_hat, [40, 65, 80, 92])
    bounds = [(q[0], q[-1])] * 4

    res = minimize(mae, x0=init_edges, method="Powell", options={"maxiter": 500})
    best_edges = sorted(res.x) if res.success and not np.isinf(res.fun) else init_edges.tolist()
    return best_edges


################################################################################
# Threshold optimiser                                                           
################################################################################

def optimise_threshold(
    y_true: np.ndarray,
    pd_hat: np.ndarray,
    sample_weight: np.ndarray,
    *,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> float:
    """Return threshold that minimises expected mis‑classification cost."""

    sort_idx = np.argsort(pd_hat)
    pd_sorted, y_sorted, w_sorted = (
        pd_hat[sort_idx],
        y_true[sort_idx],
        sample_weight[sort_idx],
    )

    tp = np.cumsum(w_sorted[::-1] * y_sorted[::-1])[::-1]
    fp = np.cumsum(w_sorted[::-1] * (1 - y_sorted[::-1]))[::-1]
    fn = tp[-1] - tp  # positives missed below threshold
    cost = cost_fn * fn + cost_fp * fp
    thr_idx = np.argmin(cost)
    return float(pd_sorted[thr_idx])

class DecisionEngine:
    def __init__(
        self, cfg: BusinessFrame, *, edges: Sequence[float] | None = None
    ) -> None:
        self.cfg = cfg
        self.labels = list(cfg.labels)
        self.mat = cfg.decision_matrix

        self.edges = (
            np.array(edges, dtype=float)
            if edges is not None
            else np.asarray(cfg.rating_edges, dtype=float)
        )
        if self.edges is None or len(self.edges) != len(self.labels) - 1:
            raise ValueError(
                "DecisionEngine requires valid 'rating_edges' – inject them "
                "during model load: cfg = cfg.with_edges(trained_edges)"
            )

    @staticmethod
    def _band(exposure: float) -> str:
        return "<1M" if exposure < 1_000_000 else ">=1M"

    def decide(self, pd_hat: float, exposure: float, **extra: Any) -> Dict[str, Any]:
        idx = np.searchsorted(self.edges, pd_hat, side="right")
        rating = self.labels[idx]
        rule = (
            self.mat.get(rating, {}).get(self._band(exposure))
            or self.mat.get(rating, {}).get("*")
            or {"action": "DECLINE"}
        )
        out = {"rating": rating, **rule}
        out.update(extra)
        return out


# ── drift metrics ───────────────────────────────────────────────────────────

def psi(expected: NDArray[np.float_], actual: NDArray[np.float_], bins: int = 10) -> float:
    cuts = np.quantile(expected, np.linspace(0, 1, bins + 1))
    e_cnt, _ = np.histogram(expected, bins=cuts)
    a_cnt, _ = np.histogram(actual,   bins=cuts)
    e_pct, a_pct = e_cnt / e_cnt.sum(), a_cnt / a_cnt.sum()
    return float(((a_pct - e_pct) * np.log((a_pct + 1e-9) / (e_pct + 1e-9))).sum())


# ── transformers ────────────────────────────────────────────────────────────
class WoETransformer(FunctionTransformer):
    def __init__(self, col: str):
        super().__init__(validate=False, feature_names_out="one-to-one")
        self.col = col
    def fit(self, X, y=None):  # noqa: N803
        ob = OptimalBinning(name=self.col, dtype="numerical", max_n_bins=5, solver="cp")
        ob.fit(X[self.col].values, y)
        self.bin_ = ob
        return self
    def transform(self, X):
        return pd.DataFrame({f"woe_{self.col}": self.bin_.transform(X[self.col].values, metric="woe")}, index=X.index)

################################################################################
# Helper: winsor‑clip transformer                                               
################################################################################
class WinsorClipper(FunctionTransformer):
    """Clip numerical columns to percentile range at *score* time as well."""

    def __init__(self, win_dict: Dict[str, Tuple[float, float]]):
        super().__init__(validate=False, feature_names_out="one-to-one")
        self.win_dict = win_dict

    def fit(self, X, y=None):  # noqa: N803
        self.bounds_: Dict[str, Tuple[float, float]] = {}
        for col, (lo, hi) in self.win_dict.items():
            if col in X.columns:
                self.bounds_[col] = (np.nanquantile(X[col], lo), np.nanquantile(X[col], 1 - hi))
        return self

    def transform(self, X):
        X = X.copy()
        for col, (lo, hi) in self.bounds_.items():
            X[col] = X[col].clip(lo, hi)
        return X

################################################################################
# Helper: polynomial interaction transformer                                    
################################################################################
class InteractionGenerator(FunctionTransformer):
    """Generate pairwise interactions between macro and momentum columns."""

    def __init__(self, macro_cols: List[str], momentum_cols: List[str]):
        super().__init__(validate=False, feature_names_out="one-to-one")
        self.macro_cols = macro_cols
        self.momentum_cols = momentum_cols

    def fit(self, X, y=None):  # noqa: N803
        return self  # stateless

    def transform(self, X):
        X = X.copy()
        for m in self.macro_cols:
            if m not in X.columns:
                continue
            for mom in self.momentum_cols:
                if mom in X.columns:
                    X[f"{m}_x_{mom}"] = X[m] * X[mom]
        return X

################################################################################
# Helper: K‑fold Empirical‑Bayes encoder                                        
################################################################################
class KFoldEBTargetEncoder(FunctionTransformer):
    """Empirical‑Bayes target encoder with out‑of‑fold encoding during fit."""

    def __init__(self, col: str, n_splits: int = 5, prior: Tuple[float, float] = (2.0, 98.0), seed: int = GLOBAL_SEED):
        super().__init__(validate=False, feature_names_out="one-to-one")
        self.col, self.n_splits, self.prior, self.seed = col, n_splits, prior, seed

    def fit(self, X, y=None):  # noqa: N803
        assert y is not None, "y is required for target encoding"
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        a0, b0 = self.prior
        # compute full posterior for inference
        grp = pd.DataFrame({self.col: X[self.col].astype(str), "y": y}).groupby(self.col)["y"].agg(["sum", "count"])
        self.post_full_ = (grp["sum"] + a0) / (grp["count"] + a0 + b0)
        self.global_ = y.mean()
        # build OOF vector (not strictly needed unless downstream wants it)
        kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        oof = np.zeros(len(X))
        for tr_idx, val_idx in kf.split(X, y):
            grp_tr = pd.DataFrame({self.col: X.loc[tr_idx, self.col].astype(str), "y": y.loc[tr_idx]}).groupby(self.col)["y"].agg(["sum", "count"])
            post = (grp_tr["sum"] + a0) / (grp_tr["count"] + a0 + b0)
            oof[val_idx] = X.loc[val_idx, self.col].astype(str).map(post).fillna(self.global_)
        self.oof_ = oof  # can be exposed for analysis
        return self

    def transform(self, X):  # noqa: N803
        x = X[self.col].astype(str)
        enc = x.map(self.post_full_).fillna(self.global_)
        return pd.DataFrame({f"eb_{self.col}": enc}, index=X.index)

# ╭────────────────────────── EB for list‑column ────────────────────────────╮
class EBListTargetEncoder(FunctionTransformer):
    def __init__(self, col: str = "category_list", target: str = "target", prior: tuple[float, float] = (2.0, 98.0)):
        super().__init__(validate=False, feature_names_out="one-to-one")
        self.col, self.target, self.prior = col, target, prior

    def fit(self, X, y=None):  # noqa: N803 – sklearn API
        if y is None:
            raise ValueError("y is required for EBListTargetEncoder")
        if not hasattr(y, "loc"):
            y = pd.Series(y, index=X.index)
        s = X[self.col].explode().astype(str)
        tbl = pd.DataFrame({"tag": s, self.target: y.loc[s.index]})
        grp = tbl.groupby("tag")[self.target].agg(["sum", "count"])
        a0, b0 = self.prior
        self.post_ = (grp["sum"] + a0) / (grp["count"] + a0 + b0)
        self.global_ = y.mean()
        return self

    def transform(self, X):  # noqa: N803
        def row_mean(tags):
            vals = [self.post_.get(str(t), np.nan) for t in tags]
            vals = [v for v in vals if pd.notna(v)]
            return np.mean(vals) if vals else self.global_
        out = X[self.col].apply(row_mean)
        return pd.DataFrame({"eb_sector_mean": out}, index=X.index)


# ╭────────────────────── Hierarchical median imputer ───────────────────────╮
class HierarchicalMedianImputer(FunctionTransformer):
    """Fit medians by geo × sector hierarchy, then impute in cascade order."""

    def __init__(self, num_cols: list[str]):
        super().__init__(validate=False, feature_names_out="one-to-one")
        self.num_cols = num_cols

    def fit(self, X, y=None):  # noqa: N803
        expl = X.explode("category_list")[["country_code", "subregion", "category_list", *self.num_cols]].copy()
        self.glob_ = expl[self.num_cols].median()
        self.med_ctry_cat_   = expl.groupby(["country_code", "category_list"])[self.num_cols].median()
        self.med_ctry_       = expl.groupby("country_code")[self.num_cols].median()
        self.med_subr_cat_   = expl.groupby(["subregion", "category_list"])[self.num_cols].median()
        self.med_subr_       = expl.groupby("subregion")[self.num_cols].median()
        self.med_cat_        = expl.groupby("category_list")[self.num_cols].median()
        return self

    def transform(self, X):  # noqa: N803
        X = X.copy()
        for idx in X[self.num_cols].isna().any(axis=1).pipe(lambda s: s[s].index):
            c, r, cats = X.at[idx, "country_code"], X.at[idx, "subregion"], X.at[idx, "category_list"]
            cats = cats if isinstance(cats, list) else []
            for col in self.num_cols:
                if pd.notna(X.at[idx, col]):
                    continue
                # cascade searches
                for cat_set in [cats, cats[::-1], [None]]:
                    if cat_set:
                        for cat in cat_set:
                            if pd.notna(X.at[idx, col]):
                                break
                            # country + cat
                            if (c, cat) in self.med_ctry_cat_.index:
                                val = self.med_ctry_cat_.at[(c, cat), col]
                                if pd.notna(val):
                                    X.at[idx, col] = val
                    if pd.notna(X.at[idx, col]):
                        break
                # country only
                if pd.isna(X.at[idx, col]) and c in self.med_ctry_.index:
                    X.at[idx, col] = self.med_ctry_.at[c, col]
                # subregion + cat set
                if pd.isna(X.at[idx, col]):
                    for cat_set in [cats, cats[::-1], [None]]:
                        for cat in cat_set:
                            if (r, cat) in self.med_subr_cat_.index:
                                val = self.med_subr_cat_.at[(r, cat), col]
                                if pd.notna(val):
                                    X.at[idx, col] = val
                                    break
                        if pd.notna(X.at[idx, col]):
                            break
                # subregion only
                if pd.isna(X.at[idx, col]) and r in self.med_subr_.index:
                    X.at[idx, col] = self.med_subr_.at[r, col]
                # global + cat
                if pd.isna(X.at[idx, col]):
                    for cat in cats[::-1]:
                        if cat in self.med_cat_.index:
                            val = self.med_cat_.at[cat, col]
                            if pd.notna(val):
                                X.at[idx, col] = val
                                break
                # global fallback
                if pd.isna(X.at[idx, col]):
                    X.at[idx, col] = self.glob_[col]
        return X


# ───────────────────────────── Utilities (geo) ──────────────────────────────
_country_2_3 = {c.alpha_2.upper(): c.alpha_3 for c in pycountry.countries}
_sub_map = {c.alpha_3: getattr(c, "subregion", None) for c in pycountry.countries}

# ─────────────────────── Robust date‑parser helpers ─────────────────────────
_date_regex = re.compile(r"(\d{1,4}[-/]){2}\d{2,4}")


def _parse_date(val):  # type: ignore[override]
    if pd.isna(val) or (isinstance(val, str) and not _date_regex.match(val)):
        return pd.NaT
    out = pd.to_datetime(val, errors="coerce", utc=True)
    return out if not pd.isna(out) else pd.to_datetime(val, errors="coerce", utc=True, dayfirst=True)

# ─────────────────────────── Macro join stub ───────────────────────────────

def _load_macro_file(path: Path, colname: str, *, derive_growth: bool, ma_years: int = 2) -> pd.DataFrame:
    """Read wide CSV → latest (country, value) pairs."""
    if not path.exists():
        return pd.DataFrame(columns=["country", colname])

    wide = pd.read_csv(path)
    wide = wide.rename(columns={wide.columns[0]: "country"})

    long = (
        wide.melt(id_vars="country", var_name="year", value_name="value")
        .assign(year=lambda d: pd.to_numeric(d.year, errors="coerce"),
                value=lambda d: pd.to_numeric(d.value, errors="coerce"))
        .dropna(subset=["year", "value"])
    )
    long = long[long.year <= TODAY.year]

    if derive_growth:
        long = long.sort_values(["country", "year"])
        long["value"] = (
            long.groupby("country")["value"]
            .pct_change()
            .mul(100)
            .pipe(lambda s: s.rolling(ma_years, min_periods=1).mean())
        )
    elif ma_years > 1:
        long["value"] = long.groupby("country")["value"].transform(
            lambda s: s.rolling(ma_years, min_periods=1).mean()
        )

    latest_idx = long.groupby("country")["year"].idxmax()
    latest = long.loc[latest_idx, ["country", "value"]].rename(columns={"value": colname})
    return latest


def _join_macro_pl(df: pl.DataFrame) -> pl.DataFrame:
    meta = load_country_meta()
    if not any(p.exists() for p,u in MACRO_FILES.values()):
        return df

    frames = [
        _load_macro_file(path, col, derive_growth=derive)
        for col, (path, derive) in MACRO_FILES.items()
    ]

    macro_df = frames[0]
    for f in frames[1:]:
        macro_df = macro_df.merge(f, on="country", how="outer")
        macro_df
    macro_df["country_code"] = (macro_df["country"].astype(str).apply(lambda x: resolve_country_fuzzy(x, meta)))

    macro_pl = pl.from_pandas(macro_df.drop(columns=["country"])).drop_nulls("country_code")

    return df.join(macro_pl, on="country_code", how="left")

def _generate_country_meta() -> pd.DataFrame:
    """Generate *once* a meta table with ISO3, ISO2, official/common
    names, capital, subregion. Requires `pycountry` and `CountryInfo` on a
    machine with internet (first run only)."""
    import pycountry
    from countryinfo import CountryInfo

    records: List[Dict[str, str]] = []
    for c in pycountry.countries:
        iso3 = c.alpha_3
        iso2 = c.alpha_2
        names = set(filter(None, [
            getattr(c, "name", None),
            getattr(c, "official_name", None),
            getattr(c, "common_name", None),
        ]))
        try:
            info = CountryInfo(c.name)
            capital = info.capital() or ""
            subregion = info.subregion() or ""
        except Exception:  # noqa: BLE001
            capital, subregion = "", ""
        records.append({
            "iso3": iso3,
            "iso2": iso2,
            "names": ";".join(sorted(names)),
            "capital": capital,
            "subregion": subregion,
        })
    meta = pd.DataFrame.from_records(records)
    meta.to_csv(_META_FILE, index=False)
    return meta


def load_country_meta(force_refresh: bool = False) -> pd.DataFrame:
    """Return the **static** country meta table, generating it on first call."""
    if force_refresh or not _META_FILE.exists():
        return _generate_country_meta()
    return pd.read_csv(_META_FILE)


def resolve_country_fuzzy(name: str, meta: pd.DataFrame, *, scorer=fuzz.token_sort_ratio,
                           min_score: int = 85) -> Optional[str]:
    """Return the best ISO‑3 code for an arbitrary country string.

    Parameters
    ----------
    name : str
        Raw name as appears in dataset (e.g. "Korea, Rep.").
    meta : pd.DataFrame
        Table created by :func:`load_country_meta`.
    scorer : Callable
        RapidFuzz similarity metric (defaults to token_sort_ratio).
    min_score : int, optional
        Minimum accepted similarity (0‑100). Returns *None* if below.
    """
    if not isinstance(name, str) or not name:
        return None  # type: ignore[return-value]

    # build cache of { alias : iso3 } on first run
    if not hasattr(resolve_country_fuzzy, "_alias_map"):
        alias_map = {}
        for _, row in meta.iterrows():
            for alias in row["names"].split(";"):
                alias_map[alias.lower()] = row["iso3"]
        resolve_country_fuzzy._alias_map = alias_map  # type: ignore[attr-defined]
        resolve_country_fuzzy._choices = list(alias_map.keys())  # type: ignore[attr-defined]

    # 1. direct lower‑case match
    iso = resolve_country_fuzzy._alias_map.get(name.lower())  # type: ignore[attr-defined]
    if iso:
        return iso

    # 2. fuzzy
    choice, score, *_ = process.extractOne(
        name.lower(), resolve_country_fuzzy._choices, scorer=scorer
    )  # type: ignore[attr-defined]
    return resolve_country_fuzzy._alias_map[choice] if score >= min_score else None  # type: ignore[attr-defined]

def _join_macro(df: pd.DataFrame) -> pd.DataFrame:
    if not MACRO_FILES:  # noqa: F821  – comes from previous sections
        return df
    pl_df = _join_macro_pl(pl.from_pandas(df))  # type: ignore  # noqa: F821 – defined elsewhere
    return pl_df.unique(maintain_order=True).to_pandas()


# load WGI
def _load_regulatory_framework(json_path: Path=r"C:\Users\Romain\OneDrive - KU Leuven\Masters\MBIS\Year 2\Semester 2\Statistical Consulting\regulatory_framework.json") -> pd.DataFrame:
    """Read JSON → DataFrame[iso3 + flattened governance columns]."""
    global GOV_COLS, MACRO_COLS  # capture for later interaction block

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw: Dict[str, Dict[str, object]] = json.load(f)
    except FileNotFoundError:
        logging.warning("Regulatory framework not found at %s – governance features NaN", json_path)
        return pd.DataFrame(columns=["iso3"] + GOV_COLS)
    except json.JSONDecodeError as err:
        logging.error("Bad JSON in %s: %s", json_path, err)
        return pd.DataFrame(columns=["iso3"] + GOV_COLS)

    rows: List[Dict[str, object]] = []
    col_set: set[str] = set()
    for iso3, rec in raw.items():
        flat = _flatten(rec)
        col_set.update(flat.keys())
        rows.append({"iso3": iso3, **flat})

    GOV_COLS = sorted(col_set)  # overwrite with discovered list
    MACRO_COLS = GOV_COLS.copy()  # make interactions include all

    return pd.DataFrame(rows)


# URL feature engineering ----------------------------------------------------

def _add_url_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain‑level heuristics: HTTPS flag, www prefix, complexity, etc."""

    url_cols = [c for c in df.columns if "url" in c.lower() or "homepage" in c.lower()]
    if not url_cols:
        return df  # nothing to do

    url_col = url_cols[0]
    df[url_col] = df[url_col].fillna("")

    # Basic syntactic properties
    df["domain_length"] = df[url_col].str.len()
    df["has_https"] = df[url_col].str.lower().str.startswith("https://").astype(int)
    df["has_www"] = df[url_col].str.lower().str.contains(r"^https?://(?:www\.)").astype(int)

    # Approximate complexity by counting path segments (slashes after protocol)
    df["url_complexity"] = (
        df[url_col]
        .str.replace(r"^https?://", "", regex=True)
        .str.count(r"/")
        .fillna(0)
    ).clip(upper=10)

    return df
# ───────────────────────────── load_dataset ────────────────────────────────

def load_dataset(csv: str | Path, cfg: BusinessFrame, *, training: bool) -> pd.DataFrame:
    """Snapshot → pandas DataFrame with momentum & macro‑ready features."""
    df = pd.read_csv(csv)

    # 1. status cleaning ----------------------------------------------------
    df["status"] = df["status"].astype(str).str.lower().str.strip()
    df = df[df["status"].isin(cfg.good_statuses + cfg.bad_statuses)].copy()

    # 2. core dates ---------------------------------------------------------
    for col in ["founded_at", "first_funding_at", "last_funding_at"]:
        df[col] = df[col].apply(_parse_date).dt.tz_localize(None)

    # founded_at repair
    cutoff = pd.Timestamp(year=1315,month=1,day=1)
    bad = (df["founded_at"] < cutoff) | (df["founded_at"] > pd.Timestamp(TODAY))
    df.loc[bad, "founded_at"] = df.loc[bad, "first_funding_at"]

    # 3. horizon filter ----------------------------------------------------
    age_months = (pd.Timestamp(TODAY) - df["founded_at"]).dt.days / 30.44
    df = df[age_months >= cfg.horizon_months]

    # 4. target & weights --------------------------------------------------
    df["target"] = df["status"].isin(cfg.bad_statuses).astype(int)
    df["weight"] = 1.0
    df.loc[df["status"] == "acquired", "weight"] = cfg.acquired_weight
    df.loc[df["status"] == "ipo", "weight"] = cfg.ipo_weight

    # 5. temporal deltas ---------------------------------------------------
    df["years_since_founded"] = (pd.Timestamp(TODAY) - df["founded_at"]).dt.days / 365.25
    df["years_since_last_funding"] = (pd.Timestamp(TODAY) - df["last_funding_at"]).dt.days / 365.25
    df["diff_founded_last_funding_days"] = (df["last_funding_at"] - df["founded_at"]).dt.days
    df["diff_founded_first_funding_days"] = (df["first_funding_at"] - df["founded_at"]).dt.days
    df["diff_between_fundings_days"] = (df["last_funding_at"] - df["first_funding_at"]).dt.days

    # 6. numeric sanitation -------------------------------------------------
    df["funding_total_usd"] = pd.to_numeric(df["funding_total_usd"].astype(str).str.replace("-", ""), errors="coerce")

    # 7. category parsing ---------------------------------------------------
    df["category_list"] = df["category_list"].fillna("").astype(str).str.split("|")
    df["category_list_len"] = df["category_list"].apply(len)
    df["primary_sector"] = df["category_list"].apply(lambda lst: (lst[0] if lst else "").lower())

    # 8. funding momentum (raw) -------------------------------------------
    df["funding_per_round"] = df["funding_total_usd"] / (df["funding_rounds"] + 1e-6)
    df["rounds_per_year"] = df["funding_rounds"] / (df["years_since_founded"] + 1e-6)
    df["months_since_last_round"] = df["years_since_last_funding"] * 12
    df["avg_interround_days"] = df["diff_between_fundings_days"] / np.maximum(df["funding_rounds"] - 1, 1)
    df["momentum_recency_ratio"] = df["years_since_last_funding"] / df["years_since_founded"].replace(0, np.nan)
    df["funding_per_year"] = df["funding_total_usd"] / df["years_since_founded"].replace(0, np.nan)

    # 9. ccTLD → country_code ---------------------------------------------
    need_cc = df["country_code"].isna() | (df["country_code"] == "")
    cc_from_domain = (
        df.loc[need_cc, "homepage_url"].astype(str).str.extract(r"\.([a-zA-Z]{2})$", expand=False).str.upper().map(_country_2_3)
    )
    df.loc[need_cc, "country_code"] = cc_from_domain
    df["subregion"] = df["country_code"].map(_sub_map)

    # 10. macro join -------------------------------------------------------
    df = _join_macro(df)

    # 10b Governance ------------------------------------------------------
    gov_df = _load_regulatory_framework()
    df = df.merge(gov_df, how="left", left_on="country_code", right_on="iso3")
    df.drop(columns=["iso3"], inplace=True, errors="ignore")

    # 11 Interactions -----------------------------------------------------
    for m in MACRO_COLS:
        if m in df.columns:
            df[f"{m}_x_funding_per_year"] = df[m] * df["funding_per_year"]

    # 12 Capital ----------------------------------------------------------
    meta = load_country_meta()
    df = df.merge(meta[["iso3", "capital"]], how="left", left_on="country_code", right_on="iso3")
    df["capital_dummy"] = (~df["capital"].isna()).astype(int)
    df.drop(columns=["iso3"], inplace=True, errors="ignore")

    # 13 URL features -----------------------------------------------------
    df = _add_url_features(df)

    return df.reset_index(drop=True)

# ╭────────────────────── Pipeline builder ───────────────────────────────────╮

def build_preprocessor() -> ColumnTransformer:
    base_pipe = Pipeline([
        ("hier", HierarchicalMedianImputer(NUM_EXTRA)),
        ("winsor", WinsorClipper(WINSOR)),  # NEW
        ("inter", InteractionGenerator(MACRO_COLS, MOMENTUM_COLS)),  # NEW
    ])

    tf: list = []
    # WoE numerical transforms
    for col in NUM_WOE:
        tf.append((f"woe_{col}", WoETransformer(col), [col]))

    # Standard numeric features
    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    tf.append(("num", num_pipe, NUM_EXTRA))

    # EB encoders with K‑fold OOF
    for col in CAT_COLS:
        tf.append((f"eb_{col}", KFoldEBTargetEncoder(col), [col]))  # NEW

    # EB for list column (unchanged)
    tf.append(("eb_sector", EBListTargetEncoder(), ["category_list"]))

    col_tx = ColumnTransformer(tf, remainder="drop")
    full = Pipeline([("base", base_pipe), ("cols", col_tx)])
    return full

def estimate_lgd(row: pd.Series) -> float:
    if row.get("capital_dummy", 0) == 1:
        return 0.25
    rating = str(row.get("rating", "R3"))
    return {"R1": 0.20, "R2": 0.30, "R3": 0.40, "R4": 0.55, "R5": 0.65}.get(rating, 0.45)

################################################################################
# Build classifier                                                              
################################################################################
def build_monotone_constraints(feature_names: Sequence[str]) -> Tuple[int, ...]:
    vec = tuple(1 if str(f).startswith("woe_") else 0 for f in feature_names)
    if len(vec) != len(feature_names):
        raise ValueError("feature_names length mismatch in monotone vector")
    return vec

################################################################################
# Hyper‑parameter optimisation                                                  
################################################################################

def _cv_auc(pipe: Pipeline, X: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> float:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=GLOBAL_SEED)
    try:
        from sklearn.model_selection import cross_val_score
        return float(
            cross_val_score(
                pipe,
                X,
                y,
                cv=cv,
                scoring="roc_auc",
                fit_params={"clf__sample_weight": w},
                error_score="raise",
            ).mean()
        )
    except TypeError as err:
        if "fit_params" not in str(err):
            raise
        aucs: List[float] = []
        for tr_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            w_tr, w_val = w[tr_idx], w[val_idx]
            pipe.fit(X_tr, y_tr, clf__sample_weight=w_tr)
            pd_hat = pipe.predict_proba(X_val)[:, 1]
            aucs.append(roc_auc_score(y_val, pd_hat, sample_weight=w_val))
        return float(np.mean(aucs))

def build_model(
    n_features: int,
    *,
    model_type: str = "xgb",          # "xgb" | "rf"
    monotone_constraints: Tuple[int, ...] | None = None,
    params: dict | None = None,
):
    if model_type not in {"xgb", "rf"}:
        raise ValueError("model_type must be 'xgb' or 'rf'")

    params = {} if params is None else dict(params)  # shallow copy

    # ── XGBoost ---------------------------------------------------------
    if model_type == "xgb":
        if not _XGB_OK:
            raise ImportError("xgboost is not installed; install or choose model_type='rf'.")

        xgb_defaults = dict(
            n_estimators=700,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        xgb_defaults.update(params)

        ver = tuple(map(int, _xgb.__version__.split(".")[:2]))
        if ver < (2, 0):
            if monotone_constraints is None:
                raise ValueError("XGBoost <2.0 requires monotone_constraints.")
            if len(monotone_constraints) != n_features:
                raise ValueError("monotone_constraints length mismatch.")
            xgb_defaults["monotone_constraints"] = tuple(monotone_constraints)
        return XGBClassifier(**xgb_defaults)

    rf_defaults = dict(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    rf_defaults.update(params)
    return RandomForestClassifier(**rf_defaults)


def tune_hyperparams(
    pipe_base: Pipeline,
    X: pd.DataFrame,
    y: NDArray[np.int_],
    w: NDArray[np.float_],
    *,
    model_type: str="xgb",
    n_trials: int = 2,
) -> dict:
    """Return best params (conditional on *model_type*)."""

    def objective(trial: optuna.Trial):
        if model_type == "xgb":
            params = {
                "clf__n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
                "clf__learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "clf__max_depth": trial.suggest_int("max_depth", 3, 7),
                "clf__subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "clf__colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            }
        else:  # RandomForest
            params = {
                "clf__n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
                "clf__max_depth": trial.suggest_int("max_depth", 5, 30),
                "clf__min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "clf__min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "clf__max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            }
        pipe = pipe_base.set_params(**params)
        return _cv_auc(pipe, X, y, w)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logging.info("Best CV AUC: %.4f", study.best_value)
    return study.best_params

################################################################################
# Training                                                                      
################################################################################
def _calibration_curve_weighted(y, y_hat, *, sample_weight, n_bins):
    """Compatibility wrapper: weighted reliability curve even on sklearn<1.3."""
    if sample_weight is None:
        prob_true, prob_pred = calibration_curve(y, y_hat, n_bins=n_bins, strategy="quantile")
        return prob_true, prob_pred

    # manual binning when sklearn lacks sample_weight -------------------
    df_tmp = (
        pd.DataFrame({"y": y, "p": y_hat, "w": sample_weight})
        .sort_values("p")
        .reset_index(drop=True)
    )
    df_tmp["bin"] = pd.qcut(df_tmp.index + 1, q=n_bins, labels=False)
    grp = df_tmp.groupby("bin")
    prob_pred = grp["p"].mean().values
    prob_true = (grp.apply(lambda g: np.average(g["y"], weights=g["w"]))).values
    return prob_true, prob_pred


def bootstrap_calibration_band(
    pd_hat: np.ndarray,
    y_true: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    n_boot: int = 1000,
    n_bins: int = 10,
    ci: float = 0.90,
) -> Dict[str, np.ndarray]:
    """Return reliability curve mean ± CI via bootstrap (weight‑aware)."""

    lower_q = (1 - ci) / 2 * 100
    upper_q = 100 - lower_q

    prob_true, prob_pred = _calibration_curve_weighted(
        y_true, pd_hat, sample_weight=sample_weight, n_bins=n_bins
    )

    rng = np.random.default_rng(42)
    idx_all = np.arange(len(pd_hat))
    boot_mat = np.empty((n_boot, n_bins))

    for i in range(n_boot):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        bt_true, _ = _calibration_curve_weighted(
            y_true[idx],
            pd_hat[idx],
            sample_weight=None if sample_weight is None else sample_weight[idx],
            n_bins=n_bins,
        )
        boot_mat[i] = bt_true

    lower = np.percentile(boot_mat, lower_q, axis=0)
    upper = np.percentile(boot_mat, upper_q, axis=0)

    return {"prob_pred": prob_pred, "prob_true": prob_true, "lower": lower, "upper": upper}


def plot_calibration_band(band: Dict[str, np.ndarray], *, ax: plt.Axes | None = None):
    ax = ax or plt.gca()
    ax.plot(band["prob_pred"], band["prob_true"], "-o", label="observed")
    ax.fill_between(band["prob_pred"], band["lower"], band["upper"], alpha=0.25)
    ax.plot([0, 1], [0, 1], "--", color="grey", label="ideal 45°")
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Observed default rate")
    ax.legend()
    return ax

GLOBAL_SEED = 42

def train(csv: str | Path, model_out: str | Path, cfg_path: str | Path | None):
    """Full training pipeline with patches and calibration bands."""

    # --- load config ----------------------------------------------------
    cfg = read_cfg(cfg_path)  # noqa: F821
    if not cfg.decision_matrix:
        cfg.decision_matrix = DEFAULT_DEC_MATRIX  # noqa: F821

    cfg.use_lgd = getattr(cfg, "use_lgd", False)
    cfg.rating_edges = getattr(cfg, "rating_edges", None)

    # --- data -----------------------------------------------------------
    df = load_dataset(csv, cfg, training=True)  # noqa: F821
    X = df.drop(columns=["target", "weight"])
    y = df["target"].values
    w = df["weight"].values

    X_build, X_hold, y_build, y_hold, w_build, w_hold = train_test_split(
        X, y, w, test_size=0.20, stratify=y, random_state=GLOBAL_SEED
    )
    X_cal, X_thr, y_cal, y_thr, w_cal, w_thr = train_test_split(
        X_hold, y_hold, w_hold, test_size=0.50, stratify=y_hold, random_state=GLOBAL_SEED
    )

    # --- preprocessing & monotone vector --------------------------------
    pre = build_preprocessor()  # noqa: F821
    pre.fit(X_build, y_build)
    feat_names = pre["cols"].get_feature_names_out()
    mono_vec = build_monotone_constraints(feat_names)

    # --- model & hyper‑param tuning ------------------------------------
    base_clf = build_model(len(feat_names), monotone_constraints=mono_vec)
    pipe_base = Pipeline([("pre", clone(pre)), ("clf", base_clf)])
    best_params = tune_hyperparams(pipe_base, X_build, y_build, w_build)  # noqa: F821

    final_clf = build_model(len(feat_names), monotone_constraints=mono_vec, params=best_params)
    pipe = Pipeline([("pre", pre), ("clf", final_clf)])
    pipe.fit(X_build, y_build, clf__sample_weight=w_build)

    # --- calibration ----------------------------------------------------
    cal_pred_raw = pipe.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(cal_pred_raw, y_cal, sample_weight=w_cal)

    cal_pred = iso.predict(cal_pred_raw)
    thr_pred = iso.predict(pipe.predict_proba(X_thr)[:, 1])

    thr = optimise_threshold(
        y_thr,
        thr_pred,
        w_thr,
        cost_fn=cfg.cost_fn,
        cost_fp=cfg.cost_fp,
    )  # noqa: F821

    # --- performance metrics ------------------------------------------
    auc_thr = roc_auc_score(y_thr, thr_pred, sample_weight=w_thr)
    brier_thr = brier_score_loss(y_thr, thr_pred, sample_weight=w_thr)
    logging.info("AUC=%.4f, Brier=%.4f, Threshold=%.4f", auc_thr, brier_thr, thr)

    # --- rating edges --------------------------------------------------
    edges = optimise_edges(
        y_build,
        iso.predict(pipe.predict_proba(X_build)[:, 1]),
        cfg.pd_table,
    )  # noqa: F821
    cfg.rating_edges = tuple(edges)

    # ── SHAP explainability ────────────────────────────────────────────────
    shap_imp: NDArray | None = None
    shap_paths: dict[str, Path] | None = None
    try:
        # 1. Build the explainer on the *pre-processed* design matrix ----------------
        X_build_pre = pre.transform(X_build)
        explainer = shap.Explainer(final_clf, X_build_pre, feature_names=feat_names)

        # 2. Compute the SHAP values only once ---------------------------------------
        shap_values = explainer(X_build_pre)          # a shap.Values object
        shap_array   = shap_values.values             # (n_samples, n_features)
        shap_imp     = np.abs(shap_array).mean(axis=0)

        # 3. Produce two ready-to-share figures --------------------------------------
        #    (a) bar plot with mean |SHAP| of the 20 most important features
        bar_path = Path(model_out).with_name(f"{Path(model_out).stem}_shap_bar_top20.png")
        shap.plots.bar(shap_values, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(bar_path, dpi=180)
        plt.close()

        #    (b) beeswarm plot for the same 20 features (nice for distribution view)
        bee_path = Path(model_out).with_name(f"{Path(model_out).stem}_shap_beeswarm_top20.png")
        shap.plots.beeswarm(shap_values, max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(bee_path, dpi=180)
        plt.close()

        shap_paths = {"bar": bar_path, "beeswarm": bee_path}
        logging.info("SHAP plots saved to %s and %s", bar_path, bee_path)

    except Exception as exc:
        # SHAP can fail if the model type is not supported; we don’t want to block training.
        logging.warning("Skipping SHAP explainability – %s: %s", type(exc).__name__, exc)
        shap_imp   = None
        shap_paths = None



    # --- calibration band (stored for dashboards) ----------------------
    calib_band = bootstrap_calibration_band(cal_pred, y_cal, sample_weight=w_cal)

    meta = {
        "trained": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "auc_thr": auc_thr,
        "brier_thr": brier_thr,
        "threshold": thr,
        "edges": edges,
        "best_params": best_params,
        "cfg": cfg.__dict__,
        "calib_band": calib_band,
    }

    bundle = {
        "pipe": pipe,
        "iso": iso,
        "meta": meta,
        "psi_ref": iso.predict(pipe.predict_proba(X)[:, 1]),
    }
    joblib.dump(bundle, model_out)
    logging.info("✔︎ Saved model (with calibration band) to %s", model_out)


################################################################################
# Scoring                                                                       
################################################################################

def score(model_path: str | Path, csv_in: str | Path, csv_out: str | Path):
    bundle = joblib.load(model_path)
    cfg = BusinessFrame(**bundle["meta"]["cfg"])  # upgraded dataclass

    # --- data -----------------------------------------------------------
    df = load_dataset(csv_in, cfg, training=False)  # noqa: F821
    pd_hat = bundle["iso"].predict(bundle["pipe"].predict_proba(df)[:, 1])
    df["pd_hat"] = pd_hat

    # ensure edges present ----------------------------------------------
    cfg.rating_edges = tuple(bundle["meta"]["edges"])
    df["rating"] = np.searchsorted(cfg.rating_edges, pd_hat, side="right")
    df["rating"] = df["rating"].map(dict(enumerate(cfg.labels)))

    if "exposure" not in df:
        df["exposure"] = 500_000

    # --- optional LGD ---------------------------------------------------
    if cfg.use_lgd:
        df["lgd_est"] = df.apply(estimate_lgd, axis=1)  # noqa: F821
    else:
        df["lgd_est"] = np.nan

    dec_engine = DecisionEngine(cfg)
    decisions = [
        dec_engine.decide(p, e, lgd=(l if cfg.use_lgd else None))
        for p, e, l in zip(df["pd_hat"], df["exposure"], df["lgd_est"])
    ]
    df = pd.concat([df, pd.DataFrame(decisions)], axis=1)

    # --- drift metrics --------------------------------------------------
    ref = bundle["psi_ref"]
    logging.info("PSI vs train: %.4f", psi(ref, pd_hat))  # noqa: F821
    logging.info("K‑S statistic: %.4f", ks_2samp(ref, pd_hat)[0])

    thr = bundle["meta"].get("threshold", 0.5)
    df["hard_decline"] = (pd_hat >= thr).astype(int)

    df.to_csv(csv_out, index=False)
    logging.info("✔︎ Scored → %s", csv_out)

def stress_test_threshold(
        thr: float,
        pd_hat: np.ndarray,
        *,
        shock_factor: float = 1.20,
        cost_fn: float = 5.0,
        cost_fp: float = 1.0,
        sample_weight: np.ndarray | None = None,
    ) -> Dict[str, float]:
        """Return new mis‑classification cost if macro shock ↑ PD by *shock_factor*.

        You can call this on the *hold‑out* set right after optimising the base
        threshold to understand portfolio sensitivity.
        """

        pd_shocked = np.clip(pd_hat * shock_factor, 0, 1)
        y_pred_base = (pd_hat >= thr).astype(int)
        y_pred_shock = (pd_shocked >= thr).astype(int)

        if sample_weight is None:
            sample_weight = np.ones_like(pd_hat)

        def _cost(y_hat):
            fn = ((1 - y_hat) * sample_weight).sum()  # false‑negatives (default mis‑classified)
            fp = (y_hat * sample_weight).sum()        # false‑positives (good rejected)
            return cost_fn * fn + cost_fp * fp

        return {
            "cost_base": _cost(y_pred_base),
            "cost_shocked": _cost(y_pred_shock),
            "pct_increase": (_cost(y_pred_shock) / _cost(y_pred_base) - 1) * 100,
        }