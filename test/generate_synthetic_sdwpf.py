"""Generate a tiny synthetic schema fixture for pipeline tests only.

The output is deliberately marked synthetic and must never be used as an SDWPF
result or copied to the official raw-data directory.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parents[1]
OUT = ROOT / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(20260813)
times = pd.date_range("2020-01-01", periods=10 * 24 * 6, freq="10min")
rows = []
for client in range(1, 6):
    keep = np.ones(len(times), dtype=bool)
    if client == 2:
        keep[100:112] = False
    if client == 4:
        keep[500:700] = False
    # Shared collection gap.
    keep[900:906] = False
    t = times[keep]
    wspd = np.clip(rng.weibull(2, len(t)) * (6 + client * 0.35), 0, 28)
    patv = np.clip((wspd / (10 + client * 0.2)) ** 3 * 1500, 0, 1500)
    patv += rng.normal(0, 40 + client * 3, len(t))
    patv = np.clip(patv, -20, 1600)
    if client == 3:
        patv[50:70] = 0
    wdir = rng.uniform(-180, 180, len(t))
    ndir = rng.uniform(-360, 360, len(t))
    if client == 5:
        wdir[10] = 181
        ndir[11] = 721
    pab = np.zeros(len(t))
    if client == 3:
        pab[60] = 90
    for i, ts in enumerate(t):
        rows.append(
            {
                "TurbID": client,
                "Day": ts.strftime("%Y-%m-%d"),
                "Tmstamp": ts.strftime("%H:%M"),
                "Wspd": wspd[i],
                "Wdir": wdir[i],
                "Etmp": 8 + rng.normal(0, 6),
                "Itmp": 15 + rng.normal(0, 5),
                "Ndir": ndir[i],
                "Pab1": pab[i],
                "Pab2": pab[i],
                "Pab3": pab[i],
                "Prtv": rng.normal(0, 25),
                "T2m": 7 + rng.normal(0, 5),
                "Sp": 90000 + rng.normal(0, 500),
                "RelH": np.clip(rng.normal(60, 15), 0, 100),
                "Wspd_w": max(0, wspd[i] * 0.7 + rng.normal(0, 0.4)),
                "Wdir_w": rng.uniform(0, 360),
                "Tp": max(0, rng.normal(0.0001, 0.0002)),
                "Patv": patv[i],
            }
        )

df = pd.DataFrame(rows)
df.loc[df.index[20], "Wspd"] = np.nan
df.to_parquet(OUT / "SYNTHETIC_sdwpf.parquet", index=False)
pd.DataFrame(
    {
        "TurbID": range(1, 6),
        "x": [0, 500, 1000, 0, 500],
        "y": [0, 0, 0, 700, 700],
        "elevation": [100, 105, 98, 112, 108],
    }
).to_csv(OUT / "SYNTHETIC_location.csv", index=False)
print(OUT)
