"""
#can look at bayesian smoothing, better country score, adjust better the imputation of dates, 
#better weighting etc., improve stress tests, add more features, use a different model,
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pycountry
from countryinfo import CountryInfo
import requests
from dateutil import parser as dtparse
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from collections import OrderedDict

# ╭─────────────────────────────── Constants ───────────────────────────────╮ #

TODAY = date.today()

MACRO_DIR = Path(r"C:\Users\Romain\OneDrive - KU Leuven\Masters\MBIS\Year 2\Semester 2\Statistical Consulting\ExternalData") 
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

MACRO_FILES: Dict[str, Tuple[Path, bool]] = {
    #             path                             derive_growth?
    "inflation":         (MACRO_DIR / "inflation.csv",        False),
    "gdp_growth":        (MACRO_DIR / "gdp.csv",              True),
    "unemployment_rate": (MACRO_DIR / "unemployment.csv",     False),
}

DEFAULT_CFG: Dict[str, Any] = {  # unchanged except for cosmetic re-order
    "score_buckets": {
        "capital": [0.2e6, 3e6, 15e6],
        "momentum_span": [12, 24, 48],
        "momentum_creation": [6, 12, 24],
        "momentum_freq": [0.5, 1.0, 2.0],
        "momentum_recency": [6, 12, 24],
        "momentum_interround": [3, 6, 12],
        "momentum_recency_ratio": [0.25, 0.5, 0.75],
        "momentum_avg_round": [0.1e6, 1e6, 5e6],
        "momentum_per_year": [0.2e6, 1e6, 5e6],
        "age": [3, 6, 10],
        "diversification": [1, 3, 5],
        "country": [0.05, 0.10, 0.20],
        "sector": [0.10, 0.20, 0.30],
    },
    "score_weights": {
        "capital": 0.15,
        "momentum_span": 0.025,
        "momentum_creation": 0.025,
        "momentum_freq": 0.025,
        "momentum_recency": 0.025,
        "momentum_interround": 0.025,
        "momentum_recency_ratio": 0.025,
        "momentum_avg_round": 0.025,
        "momentum_per_year": 0.025,
        "age": 0.10,
        "diversification": 0.10,
        "sector": 0.15,
        "country": 0.15,
    },
    "pd_table": {"R1": 0.02, "R2": 0.05, "R3": 0.12, "R4": 0.20, "R5": 0.35},
    "stress_scenarios": {
        "baseline": {},
        "mild": {"country_bucket_shift": 1},
        "severe": {"country_bucket_shift": 2, "capital_bucket_shift": 1},
    },
    "closure_weight": 2.0,
}


DEFAULT_CFG["score_buckets"] |= {
    "inflation":          [2.0, 4.0, 8.0],     # -► lower is better  (use inverse rule)
    "gdp_growth":         [0.0, 2.0, 4.0],     # -► higher is better
    "unemployment_rate":  [4.0, 6.0, 10.0],    # -► lower is better  (inverse)
}

DEFAULT_CFG["score_weights"] |= {
    "inflation":         0.05,
    "gdp_growth":        0.05,
    "unemployment_rate": 0.05,
}
RATING_ORDER = list(DEFAULT_CFG["pd_table"].keys())
PD_TABLE_FALLBACK = DEFAULT_CFG["pd_table"]

# ╭─────────────────────────────── Utilities ───────────────────────────────╮ #

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{hashlib.sha1(key.encode()).hexdigest()}.json"


def cache_get(key: str, *, ttl_days: int = 365) -> Optional[dict]:
    p = _cache_path(key)
    if p.exists() and (TODAY - date.fromtimestamp(p.stat().st_mtime)).days <= ttl_days:
        return json.loads(p.read_text())
    return None


def cache_set(key: str, data: dict):
    _cache_path(key).write_text(json.dumps(data))


def _country_name_to_iso3(name: str) -> Optional[str]:
    name=name.replace("P.D.R.","People's Democratic Republic").replace("Dem.","Democratic").replace("Rep.","Republic")
    country_obj=pycountry.countries.get(common_name=name.split(",")[0].strip()) or pycountry.countries.get(common_name=name.split(",")[0].strip()) or pycountry.countries.get(name=name) or pycountry.countries.get(official_name=name) or pycountry.countries.get(name=name.split(",")[0].strip()) or pycountry.countries.get(official_name=name.split(",")[0].strip())
    #country_obj = pycountry.countries.get(name=name) or pycountry.countries.get(official_name=name)
    return country_obj.alpha_3 if country_obj else None


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


def _join_macro(df: pl.DataFrame) -> pl.DataFrame:
    if not any(p.exists() for p,u in MACRO_FILES.values()):
        for p, u in MACRO_FILES.values():
            print(p, u)
            print(p.exists())
            print(u)    
        print("here")
        return df

    frames = [
        _load_macro_file(path, col, derive_growth=derive)
        for col, (path, derive) in MACRO_FILES.items()
    ]
    macro_df = frames[0]
    for f in frames[1:]:
        macro_df = macro_df.merge(f, on="country", how="outer")

    macro_df["country_code"] = macro_df.country.map(_country_name_to_iso3)
    macro_pl = pl.from_pandas(macro_df.drop(columns=["country"])).drop_nulls("country_code")
    return df.join(macro_pl, on="country_code", how="left")

# ╭────────────────────────────── Data Cleaning ────────────────────────────╮ #

def load_and_clean(path: str | Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    df.head()
    date_cols = ["founded_at", "first_funding_at", "last_funding_at"]

    def parse_two_patterns(col):
        cleaned = pl.col(col).str.replace_all("-", "/")
        dmy = cleaned.str.strptime(pl.Date, "%d/%m/%Y", strict=False)
        ymd = cleaned.str.strptime(pl.Date, "%Y/%m/%d", strict=False)
        return pl.coalesce([dmy, ymd]).alias(col)

    df = (
        pl.read_csv(path)
        .with_columns([parse_two_patterns(c) for c in date_cols])
    )
    cutoff = date(1315, 1, 1)

    df = df.with_columns(
        pl.when((pl.col("founded_at") < pl.lit(cutoff)) | (pl.col("founded_at") > pl.lit(TODAY)))
        .then(pl.col("first_funding_at"))
        .otherwise(pl.col("founded_at"))
        .alias("founded_at")
    )

    status_map = {"closed": 0, "operating": 1, "acquired": 2, "ipo": 3}
    df = (
        df.with_columns(
            pl.col("status").replace(status_map, default=None).alias("target"),
            (pl.col("status") == "closed").cast(pl.Int8).alias("target_binary"),
        )
        .drop("status")
    )

    df = df.with_columns(
        ((pl.lit(datetime.now()) - pl.col("founded_at")).dt.total_days() / 365).round(0).cast(pl.Int64).alias(
            "years_since_founded"
        ),
        ((pl.lit(datetime.now()) - pl.col("first_funding_at")).dt.total_days()/365).round(0).cast(pl.Int64).alias(
            "years_since_first_funding"
        ),
        ((pl.lit(datetime.now()) - pl.col("last_funding_at")).dt.total_days()/365).round(0).cast(pl.Int64).alias(
            "years_since_last_funding"
        ),
        pl.col("founded_at").dt.year().cast(pl.Int64).alias("foundation_year"),
        pl.col("first_funding_at").dt.year().cast(pl.Int64).alias("first_funding_year"),
        pl.col("last_funding_at").dt.year().cast(pl.Int64).alias("last_funding_year"),
        pl.col("funding_total_usd").str.replace("-", "").cast(pl.Float64, strict=False),
    (pl.col("last_funding_at") - pl.col("founded_at")).dt.total_days().alias("diff_founded_last_funding_days"),
    (pl.col("first_funding_at") - pl.col("founded_at")).dt.total_days().alias("diff_founded_first_funding_days"),
    (pl.col("last_funding_at") - pl.col("first_funding_at")).dt.total_days().alias("diff_between_fundings_days"),
    ) 

    # URL / domain / subregion enrichment
    country_2_3 = {country.alpha_2: country.alpha_3 for country in pycountry.countries}
    country_3_2 = {country.alpha_3: country.alpha_2 for country in pycountry.countries}

    sub_map = {c.alpha_2: getattr(c, 'subregion', None) for c in pycountry.countries if hasattr(c, 'alpha_2')}
    df = df.with_columns(
        pl.when(pl.col("homepage_url").is_not_null() & (pl.col("homepage_url") != ""))
        .then(1)
        .otherwise(0)
        .alias("has_homepage"),
        pl.col("homepage_url").str.extract(r"https?://([^/]+)").alias("domain"),
        pl.col("country_code").replace(sub_map, default=None).alias("subregion"),
        pl.col("category_list").fill_null("").str.split("|").alias("category_list"),
    )
    df = df.with_columns([
    pl.when(
        (pl.col("country_code").is_null() | (pl.col("country_code") == ""))
        & (~pl.col("homepage_url").str.ends_with(".com")) & (~pl.col("homepage_url").str.ends_with(".net"))
        & (pl.col("homepage_url").is_not_null())& (~pl.col("homepage_url").str.ends_with(".ai"))& (~pl.col("homepage_url").str.ends_with(".io"))
        & (~pl.col("homepage_url").str.ends_with(".org"))& (~pl.col("homepage_url").str.ends_with(".co"))
    )
    .then(
         pl.col("homepage_url")
         .str.extract(r"\.([a-zA-Z]+)$")
         .str.to_uppercase()
         .replace_strict(country_2_3, default=None)
    )
    .otherwise(pl.col("country_code"))
    .alias("country_code")
    ])
    #capitals={}
    subregions = {}
    for country in pycountry.countries:
        try:
            info = CountryInfo(country.name)
            #cap = info.capital()
            subreg = info.subregion()
            #capitals[country.alpha_3] = cap
            subregions[country.alpha_3] = subreg
        except Exception:
            continue
    df = df.with_columns(
        #pl.col("country_code").replace_strict(capitals,default=None).alias("capital"),
        pl.col("country_code").replace_strict(subregions,default=None).alias("subregion")
    )

    df = df.with_columns(pl.col("category_list").list.len().alias("category_list_len"))

    # macro join
    return _join_macro(df).unique(maintain_order=True)

# ╭─────────────────────────── Scoring primitives ──────────────────────────╮ #

def bucket_score(val: float | int | None, bands: List[float]) -> int:
    if pd.isna(val):
        return 5
    low, mid, high = bands
    return 1 if val >= high else 3 if val >= mid else 5 if val >= low else 7


def bucket_score_inv(val: float | int | None, bands: List[float]) -> int:
    if pd.isna(val):
        return 5
    low, mid, high = bands
    return 1 if val <= low else 3 if val <= mid else 5 if val <= high else 7

# ╭─────────────────────────── Feature engineering ─────────────────────────╮ #
def impute_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Fill numeric NaNs by (category, geo) hierarchy; works in-place."""

    # ------------------------------------------------------------------
    # 0) guarantee each column name appears only once
    # ------------------------------------------------------------------
    cols = list(OrderedDict.fromkeys(cols))

    # ------------------------------------------------------------------
    # 1) make sure category_list is always a Python list
    # ------------------------------------------------------------------
    df["category_list"] = (
        df["category_list"]
        .apply(
            lambda x: []
            if x is None or (isinstance(x, float) and np.isnan(x))
            else list(x)
        )
    )

    # ------------------------------------------------------------------
    # 2) pre-compute medians
    # ------------------------------------------------------------------
    global_meds = df[cols].median(numeric_only=True)

    exploded = df.explode("category_list")

    med_cc = (
        exploded.groupby(["country_code", "category_list"])[cols]
        .median(numeric_only=True)
    )
    med_sc = (
        exploded.groupby(["subregion", "category_list"])[cols]
        .median(numeric_only=True)
    )
    med_c = df.groupby("country_code")[cols].median(numeric_only=True)
    med_s = df.groupby("subregion")[cols].median(numeric_only=True)

    # ------------------------------------------------------------------
    # 3) row-wise hierarchical fill with NaN-aware checks
    # ------------------------------------------------------------------
    for col in cols:
        for idx in df.index[df[col].isna()]:
            row = df.loc[idx]
            cats = row["category_list"]             # always a list

            # ----- country + category ----------------------------------
            vals = [
                med_cc.loc[(row["country_code"], c)][col]
                for c in cats
                if (row["country_code"], c) in med_cc.index
            ]
            vals = [v for v in vals if pd.notna(v)]
            if vals:
                df.at[idx, col] = np.nanmedian(vals)
                continue

            # ----- subregion + category --------------------------------
            vals = [
                med_sc.loc[(row["subregion"], c)][col]
                for c in cats
                if (row["subregion"], c) in med_sc.index
            ]
            vals = [v for v in vals if pd.notna(v)]
            if vals:
                df.at[idx, col] = np.nanmedian(vals)
                continue

            # ----- country ---------------------------------------------
            if row["country_code"] in med_c.index:
                val = med_c.at[row["country_code"], col]
                if pd.notna(val):
                    df.at[idx, col] = val
                    continue

            # ----- subregion -------------------------------------------
            if row["subregion"] in med_s.index:
                val = med_s.at[row["subregion"], col]
                if pd.notna(val):
                    df.at[idx, col] = val
                    continue

            # ----- global (assumed non-NaN) ----------------------------
            df.at[idx, col] = global_meds[col]

    # ------------------------------------------------------------------
    # 4) quick report (optional – comment out in production)
    # ------------------------------------------------------------------
    nulls = df[cols].isna().sum()
    leftovers = nulls[nulls > 0]
    if not leftovers.empty:
        print("⚠️  Still missing after numeric impute:\n", leftovers)

    return df


def momentum_scores(row: pd.Series, cfg: Dict[str, Any]) -> Dict[str, int]:
    b = cfg["score_buckets"]
    s: Dict[str, int] = {}
    if pd.notna(row["diff_between_fundings_days"]):
        s["momentum_span"] = bucket_score(row["diff_between_fundings_days"]/30.4, b["momentum_span"])
    if pd.notna(row["diff_founded_first_funding_days"]):
        s["momentum_creation"] = bucket_score(row["diff_founded_first_funding_days"]/30.4, b["momentum_creation"])
    if row.get("funding_rounds", 0) and row.get("years_since_founded", 0) > 0:
        s["momentum_freq"] = bucket_score(row["funding_rounds"]/row["years_since_founded"], b["momentum_freq"])
    if pd.notna(row["years_since_last_funding"]):
        s["momentum_recency"] = bucket_score_inv(row["years_since_last_funding"]*12, b["momentum_recency"])
    if pd.notna(row["diff_between_fundings_days"]):
        if row.get("funding_rounds", 0) > 1:
            s["momentum_interround"] = bucket_score_inv(
                (row["diff_between_fundings_days"]/(row["funding_rounds"]-1))/30.4, b["momentum_interround"]
            )
        else:
            s["momentum_interround"] = row["years_since_founded"]
    if pd.notna(row["years_since_last_funding"]) and pd.notna(row["years_since_founded"]):
        s["momentum_recency_ratio"] = bucket_score_inv(
            row["years_since_last_funding"]/row["years_since_founded"], b["momentum_recency_ratio"]
        )
    if pd.notna(row.get("funding_total_usd")) and row.get("funding_rounds", 0) > 0:
        s["momentum_avg_round"] = bucket_score(
            row["funding_total_usd"]/row["funding_rounds"], b["momentum_avg_round"]
        )
    if pd.notna(row.get("funding_total_usd")) and pd.notna(row.get("years_since_founded")) and row["years_since_founded"] > 0:
        s["momentum_per_year"] = bucket_score(
            row["funding_total_usd"]/row["years_since_founded"], b["momentum_per_year"]
        )
    return s


def enrich_and_score(pl_df: pl.DataFrame, cfg: Dict[str, Any] = DEFAULT_CFG, *, learn_w: bool = False,
                     stress: str = "baseline", rdap: bool = False) -> pd.DataFrame:
    df = pl_df.to_pandas()

    # domain age back-fill
    def rdap_age(domain: str) -> Optional[float]:
        if not isinstance(domain, str) or not domain:
            return None
        cached = cache_get(domain)
        if cached is not None:
            return cached.get("age")
        try:
            resp = requests.get(f"https://rdap.org/domain/{domain}", timeout=10)
            resp.raise_for_status()
            ev = next((e for e in resp.json().get("events", []) if e.get("eventAction") == "registration"), None)
            age = None
            if ev and ev.get("eventDate"):
                age = (TODAY - dtparse.parse(ev["eventDate"]).date()).days / 365.25
            cache_set(domain, {"age": age})
            return age
        except Exception:  # noqa: BLE001
            cache_set(domain, {"age": None})
            return None
    if rdap:
        m = df["years_since_founded"].isna()
        if m.any():
            print("here")
            domain_ages = {d: rdap_age(d) for d in df.loc[m, "domain"].unique()}
            print("here2")
            df.loc[m, "domain_age_years"] = df.loc[m, "domain"].map(domain_ages)
            fill = df["years_since_founded"].isna() & df["domain_age_years"].notna()
            df.loc[fill, "years_since_founded"] = df.loc[fill, "domain_age_years"]
            if "years_since_first_funding" in df.columns:
                df.loc[fill, "diff_founded_first_funding_days"] = df.loc[fill, "years_since_first_funding"] * 365.25
            df.drop(columns="domain_age_years", inplace=True)
    else:
        fill = df["years_since_founded"].isna() & df["years_since_first_funding"].notna()
        df.loc[fill, "years_since_founded"] = df.loc[fill, "years_since_first_funding"]
        #if "years_since_first_funding" in df.columns:
        #    df.loc[fill, "diff_founded_first_funding_days"] = df.loc[fill, "years_since_first_funding"] * 365.25

    # hierarchical imputation
    numeric_cols = ['years_since_founded',
       'years_since_first_funding', 'years_since_last_funding','funding_rounds','funding_total_usd',"diff_founded_last_funding_days","diff_founded_first_funding_days", "diff_between_fundings_days","inflation",
        "gdp_growth",
        "unemployment_rate"]
    df = impute_numeric(df, numeric_cols)

    b = cfg["score_buckets"]

    # priors
    rates_c = df.groupby("country_code")["target_binary"].mean()
    default_mean = rates_c.mean()
    df["country_rate"] = df["country_code"].map(rates_c).fillna(default_mean)
    df["score_country"] = df["country_rate"].apply(bucket_score, bands=b["country"])

    exploded = df[["category_list", "target_binary"]].explode("category_list")
    cat_rates = exploded.groupby("category_list")["target_binary"].mean()
    df["sector_rate"] = df["category_list"].apply(
        lambda cats: np.nanmean([cat_rates.get(c, np.nan) for c in cats]) if cats else default_mean
    )
    df["score_sector"] = df["sector_rate"].apply(bucket_score, bands=b["sector"])

    df["score_capital"] = df["funding_total_usd"].apply(bucket_score, bands=b["capital"])

    tqdm.pandas(desc="momentum-scores")  # <- registers .progress_apply / .progress_map
    m_scores = df.progress_apply(lambda r: pd.Series(momentum_scores(r, cfg)), axis=1)
    for k in m_scores.columns:
        df[f"score_{k}"] = m_scores[k]

    df["score_age"] = df["years_since_founded"].apply(bucket_score, bands=b["age"])
    df["score_diversification"] = df["category_list_len"].apply(bucket_score, bands=b["diversification"])
    
    # ─── Macro-economic components ─────────────────────────────────────
    macro_b = cfg["score_buckets"]
    if {"inflation", "gdp_growth", "unemployment_rate"}.issubset(df.columns):
        df["score_inflation"]          = df["inflation"].apply(         bucket_score_inv, bands=macro_b["inflation"])
        df["score_gdp_growth"]         = df["gdp_growth"].apply(        bucket_score,     bands=macro_b["gdp_growth"])
        df["score_unemployment_rate"]  = df["unemployment_rate"].apply( bucket_score_inv, bands=macro_b["unemployment_rate"])

    components = [
        "capital",
        *m_scores.columns,
        "age",
        "diversification",
        "sector",
        "country",
        "inflation",
        "gdp_growth",
        "unemployment_rate",
    ]

    feat_cols = [f"score_{c}" for c in components]
    weights = cfg["score_weights"].copy()

    # optional weight learning (status-informed)
    if learn_w:
        nulls = df[feat_cols].isna().sum()
        leftovers = nulls[nulls > 0]
        if not leftovers.empty:
            print("⚠️  Still missing after numeric impute:\n", leftovers)
        # logistic on closed vs rest
        X = df[feat_cols].values
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        y = df["target_binary"].values
        w = np.where(y == 1, cfg["closure_weight"], 1.0)
        lr = LogisticRegression(max_iter=500).fit(Xs, y, sample_weight=w)
        fi = np.abs(lr.coef_[0]); fi /= fi.sum()
        weights.update(dict(zip(feat_cols, fi)))

    # aggregate score
    df["score_total"] = sum(df[c] * weights[c] for c in feat_cols)
    df["rating"] = pd.cut(df.score_total, bins=[-1, 2, 3.5, 5, 6.5, 9], labels=RATING_ORDER)
    df["pd_model"] = df.rating.map(cfg["pd_table"])

    # stress scenario
    scenario = cfg["stress_scenarios"].get(stress, {})
    if scenario:
        stressed = df.copy()
        for comp in ["country", "capital"]:
            shift = scenario.get(f"{comp}_bucket_shift")
            if shift:
                stressed[f"score_{comp}"] += shift
        stressed["score_total_stress"] = sum(stressed[c] * weights[c] for c in feat_cols)
        stressed["rating_stress"] = pd.cut(stressed.score_total_stress, [-1, 2, 3.5, 5, 6.5, 9], labels=RATING_ORDER)
        stressed["pd_model_stress"] = stressed.rating_stress.map(cfg["pd_table"])
        df = df.join(stressed[["score_total_stress", "rating_stress", "pd_model_stress"]])
    return df


# ╭──────────────────────────── CreditPolicy OO ────────────────────────────╮ #

def _derive_quantile_bands(
    s: pd.Series, qs: Tuple[float, float, float] = (0.3, 0.6, 0.85), *, min_obs: int = 200
) -> Tuple[float, float, float]:
    """Robust winsorised quantile bands."""
    s = s.dropna()
    if len(s) < min_obs or s.nunique() < 10:
        raise ValueError("Too few observations.")
    s = s.clip(lower=s.quantile(0.01), upper=s.quantile(0.99))
    q1, q2, q3 = s.quantile(qs)
    return float(q1), float(q2), float(q3)

@dataclass
class CreditPolicy:
    # ───────── stored parameters ───────── #
    buckets: Dict[str, Tuple[float, float, float]]
    weights: Dict[str, float]
    rating_cutoffs: List[float]              # **len = len(RATING_ORDER)-1**
    pd_mapping: Dict[str, float]
    scaler: StandardScaler = field(repr=False)
    iso_cal: IsotonicRegression = field(repr=False)
    provenance: Dict[str, str] = field(default_factory=dict)
    scenarios: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = field(
        default_factory=dict, repr=False
    )

    # ───────── static helpers ───────── #

    @staticmethod
    def _default_scenarios() -> Dict[str, Callable[[pd.DataFrame], pd.DataFrame]]:
        return {
            "mild": lambda d: d.assign(
                score_country=lambda x: x.score_country + 1
            ),
            "severe": lambda d: d.assign(
                score_country=lambda x: x.score_country + 2,
                score_capital=lambda x: x.score_capital + 1,
            ),
        }

    # ───────── construction ───────── #

    @classmethod
    def from_training(
        cls,
        df: pd.DataFrame,
        *,
        comp_cols: List[str],
        target_col: str = "target_binary",
        closure_weight: float = 2.0,
        seed: int = 42,
    ) -> "CreditPolicy":
        # 1) bucketise + score explanatory columns
        #print("Deriving quantile bands...")
        buckets = {c: _derive_quantile_bands(df[c]) for c in comp_cols}
        for c in comp_cols:
            df[f"score_{c}"] = df[c].apply(lambda x: bucket_score(x, list(buckets[c])))
        score_cols = [f"score_{c}" for c in comp_cols]

        # 2) logistic model
        scaler = StandardScaler().fit(df[score_cols])
        Xs = scaler.transform(df[score_cols])
        y = df[target_col].values
        w = np.where(y == 1, closure_weight, 1.0)
        lr = LogisticRegression(max_iter=1000, random_state=seed).fit(Xs, y, sample_weight=w)
        weights = dict(zip(score_cols, lr.coef_[0]))

        # 3) isotonic calibration
        raw = (Xs * lr.coef_).sum(axis=1)
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw, y)
        auc = roc_auc_score(y, lr.predict_proba(Xs)[:, 1])

        # 4) PD grid & cut-offs
        pd_grid = np.linspace(0.01, 0.40, num=len(RATING_ORDER))        # len == |ratings|
        rating_cutoffs = np.interp(                                     # INTERNAL break-points
            pd_grid[:-1],                                               # one fewer!
            iso.y_thresholds_,
            iso.X_thresholds_,
        ).astype(float).tolist()

        #print("Rating cutoffs:", rating_cutoffs)
        #print(len(rating_cutoffs), len(RATING_ORDER))
        pd_map = dict(zip(RATING_ORDER, pd_grid))

        # 5) provenance & scenarios
        prov = {
            "built": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "git": _sha1(Path.cwd().as_posix()),
            "auc_train": f"{auc:.4f}",
            "n_obs": str(len(df)),
        }
        scenarios = cls._default_scenarios()

        return cls(
            buckets, weights, rating_cutoffs, pd_map, scaler, iso, prov, scenarios
        )

    # ───────── API ───────── #

    def apply(self, df: pd.DataFrame, *, scenario: str = "baseline") -> pd.DataFrame:
        out = df.copy()

        # derive raw score & base PD/rating
        for col, bands in self.buckets.items():
            out[f"score_{col}"] = out[col].apply(lambda x: bucket_score(x, list(bands)))
        score_cols = [f"score_{c}" for c in self.buckets]

        raw = (
            self.scaler.transform(out[score_cols])
            * np.array([self.weights[c] for c in score_cols])
        ).sum(axis=1)

        out["pd"] = self.iso_cal.predict(raw)
        out["rating"] = pd.cut(
            raw,
            [-np.inf, *self.rating_cutoffs, np.inf],      # edges = labels + 1
            labels=RATING_ORDER,
        )

        # scenario overlay
        if scenario != "baseline":
            if scenario not in self.scenarios:
                raise KeyError(f"Unknown scenario '{scenario}'.")
            stressed = self.scenarios[scenario](out.copy())
            raw_s = (
                self.scaler.transform(stressed[score_cols])
                * np.array([self.weights[c] for c in score_cols])
            ).sum(axis=1)
            stressed["pd_stress"] = self.iso_cal.predict(raw_s)
            stressed["rating_stress"] = pd.cut(
                raw_s, [-np.inf, *self.rating_cutoffs, np.inf], labels=RATING_ORDER
            )
            out = out.join(stressed[["pd_stress", "rating_stress"]])

        return out

    # ───────── persistence ───────── #

    def _pyify(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return [self._pyify(x) for x in obj.tolist()]
        if isinstance(obj, dict):
            return {k: self._pyify(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._pyify(v) for v in obj]
        return obj

    def to_yaml(self, path: str | Path) -> None:
        import yaml, pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        d = asdict(self)
        for k in ("scaler", "iso_cal", "scenarios"):
            d.pop(k, None)                          # remove un-dumpable objects

        with path.open("w") as f:
            yaml.safe_dump(self._pyify(d), f, sort_keys=False, allow_unicode=True)

        with open(path.with_suffix(".scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        with open(path.with_suffix(".iso.pkl"), "wb") as f:
            pickle.dump(self.iso_cal, f)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "CreditPolicy":
        import yaml, pickle
        path = Path(path)
        with path.open() as f:
            d = yaml.safe_load(f)

        scaler  = pickle.load(open(path.with_suffix(".scaler.pkl"), "rb"))
        iso_cal = pickle.load(open(path.with_suffix(".iso.pkl"),    "rb"))

        d["scenarios"] = cls._default_scenarios()   # restore lambdas
        return cls(**d, scaler=scaler, iso_cal=iso_cal)

# ╭──────────────────────── Main CLI entry-point ───────────────────────────╮ #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Unified startup credit pipeline")
    p.add_argument("--input", required=True, help="Raw Crunchbase-style CSV")
    p.add_argument("--out", default="scored.parquet", help="Output path")
    p.add_argument("--policy-yaml", help="Pre-trained policy YAML; if absent we re-learn")
    p.add_argument("--stress", default="baseline", choices=["baseline", "mild", "severe"],
                   help="Scenario key")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1) ETL
    pl_raw = load_and_clean(args.input)

    # 2) Feature engineering (capital, momentum, priors, …)
    df_comp = enrich_and_score(pl_raw, learn_w=False, stress="baseline")

    # 3) Fit or load policy
    if args.policy_yaml and Path(args.policy_yaml).exists():
        policy = CreditPolicy.from_yaml(args.policy_yaml)
    else:
        policy = CreditPolicy.from_training(
            df=df_comp[df_comp["target_binary"].notna()],
            comp_cols=[c for c in df_comp.columns if c.startswith(
                ("funding_total_usd", "diff_", "years_since_", "country_rate",
                "sector_rate",                # existing
                "inflation", "gdp_growth", "unemployment_rate")  # NEW
            )],
        )
        policy.to_yaml("policy_latest.yaml")

    # 4) Apply
    df_scored = policy.apply(df_comp, scenario=args.stress)

    # 5) Persist
    df_scored.to_parquet(args.out, index=False)
    print(f"✅  {len(df_scored):,} rows written to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
# ╰──────────────────────────────────────────────────────────────────────────╯ #



# ---------------------------------------------------------------------------
# Override framework
# ---------------------------------------------------------------------------
#def log_override(obligor_id: str, new_rating: str, reason: str, user: str = os.getenv("USER", "unknown")):
#    header = not LOG_PATH.exists()
#    with LOG_PATH.open("a") as fh:
#        if header:
#            fh.write("date,obligor_id,new_rating,reason,user\n")
#        fh.write(f"{TODAY},{obligor_id},{new_rating},{reason},{user}\n")
#    print(f"✔ override logged to {LOG_PATH}")

# ---------------------------------------------------------------------------
# Exposure‑limits checker
# ---------------------------------------------------------------------------
#def load_limits(path: Path = LIMITS_PATH) -> dict:
#    if not path.exists():
#        return {}
#    with path.open() as fh:
#        return yaml.safe_load(fh)

#def check_limits(df: pd.DataFrame, limits: dict):
#    alerts = []
#    if not limits:
#        return alerts
#    groupers = ["rating", "sector", "country_code"]
#    exposure = df.groupby(groupers)["exposure_eur"].sum().reset_index()
#    for _, row in exposure.iterrows():
#        r, s, c, exp = row
#        lim = limits.get(r, {}).get(s, {}).get(c, None)
#        if lim and exp > lim:
#            alerts.append(f"Limit breach: {r}-{s}-{c} €{exp:,.0f} > €{lim:,.0f}")
#    return alerts

# ---------------------------------------------------------------------------
# Back‑testing & validation utilities
# ---------------------------------------------------------------------------

#def backtest(df: pd.DataFrame, pd_table: dict, report_path: Path):
#    rep_lines = ["# Back‑test report", f"Date: {TODAY}", ""]
#    # default rate per rating
#    grouped = df.groupby("rating")
#    for r, grp in grouped:
#        obs = grp["target_binary"].mean()
#        exp = pd_table[r]
#        z = stats.binom_test(int(grp["target_binary"].sum()), n=len(grp), p=exp, alternative='two-sided')
#        rep_lines.append(f"* {r}: observed {obs:.2%} vs exp {exp:.2%} (p={z:.3f})  n={len(grp)}")
#    # migration matrix (one‑year)
#    if "rating_prev" in df.columns:
#        mat = pd.crosstab(df["rating_prev"], df["rating"])
#        rep_lines.append("\nMigration matrix:\n")
#        rep_lines.extend(["|"+"|".join([str(x) for x in [idx]+row.tolist()])+"|" for idx,row in mat.iterrows()])
#    report_path.write_text("\n".join(rep_lines))
#    print(f"✔ back‑test report saved to {report_path}")

# ---------------------------------------------------------------------------
# Bias / fairness diagnostics (KS & PD gap)
# ---------------------------------------------------------------------------

#def fairness_report(df: pd.DataFrame, attr: str, path: Path):
#    groups = df[attr].dropna().unique()
#    lines = [f"# Fairness report on '{attr}'", ""]
#    base_pd = df["target_binary"].mean()
#    for g in groups:
#        sub = df[df[attr] == g]
#        pd_g = sub["target_binary"].mean()
#        ks = stats.ks_2samp(df["score_total"], sub["score_total"]).statistic
#        lines.append(f"* {g}: PD={pd_g:.2%} ΔPD={(pd_g-base_pd):+.1%} KS={ks:.2f} n={len(sub)}")
#    path.write_text("\n".join(lines))
#    print(f"✔ fairness report saved to {path}")