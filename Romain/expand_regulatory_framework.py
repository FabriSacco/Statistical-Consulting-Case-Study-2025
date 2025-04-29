# wgi_framework.py
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------- #
# configurable bits
# --------------------------------------------------------------------- #
INDICATORS = {
    "Regulatory Quality": "RQ.EST",
    "Rule of Law":        "RL.EST",
    "Control of Corruption": "CC.EST",
}
API_ROOT = "http://api.worldbank.org/v2"
OUT_PATH = Path(r"C:\Users\Romain\OneDrive - KU Leuven\Masters\MBIS\Year 2\Semester 2\Statistical Consulting\regulatory_framework.json")


# --------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------- #
def _fetch_indicator(indicator_code: str, years: str = "2000:2024") -> List[dict]:
    """Fetch raw WB data for *every* country for the requested indicator."""
    url = f"{API_ROOT}/country/all/indicator/{indicator_code}"
    params = {"format": "json", "per_page": 10_000, "date": years}

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if len(data) < 2:
        return []
    return data[1]


def _latest_per_country(rows: list[dict]) -> pd.DataFrame:
    """Return DataFrame with the *latest* value per country."""
    df = (
        pd.DataFrame(rows)[["countryiso3code", "date", "value"]]
        .rename(columns={"countryiso3code": "iso3", "date": "year"})
        .dropna(subset=["iso3", "value"])
    )
    df["year"] = df["year"].astype(int)
    idx = df.groupby("iso3")["year"].idxmax()
    return df.loc[idx].reset_index(drop=True)


# --------------------------------------------------------------------- #
# main routine
# --------------------------------------------------------------------- #
def build_regulatory_framework() -> Dict[str, Any]:
    print("Downloading World-Bank WGI indicators …")

    frames: list[pd.DataFrame] = []
    for nice, code in INDICATORS.items():
        rows = _fetch_indicator(code)
        if not rows:
            print(f" ⚠ No data for {nice}")
            continue
        df_ind = _latest_per_country(rows).rename(columns={"value": nice})
        frames.append(df_ind)
        time.sleep(0.3)  # gentle on the API

    if not frames:
        raise RuntimeError("No indicator data retrieved")

    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on=["iso3", "year"], how="outer")

    # normalise –2.5…+2.5 → 0…1
    for col in INDICATORS.keys():
        df[col] = (df[col] + 2.5) / 5.0

    # keep rows where **all** three indicators are present
    df = df.dropna(subset=INDICATORS.keys(), how="any")

    # build composite metrics -------------------------------------------
    out = {}
    for _, row in df.iterrows():
        rq, rl, cc = row["Regulatory Quality"], row["Rule of Law"], row["Control of Corruption"]
        out[row["iso3"]] = {
            "regulatory_quality": rq,
            "rule_of_law": rl,
            "control_of_corruption": cc,
            "startup_regulation_index": rq * 0.4 + rl * 0.3 + cc * 0.3,
            "investor_protection": rl,
            "intellectual_property_rights": rq,
            "financial_regulation": {
                "banking_supervision": rq,
                "securities_regulation": rq,
                "insurance_regulation": rq,
            },
        }

    # save ----------------------------------------------------------------
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as fh:
        json.dump(out, fh, indent=4)

    print(f"✔ Framework for {len(out)} countries saved → {OUT_PATH}")
    return out


if __name__ == "__main__":
    build_regulatory_framework()
