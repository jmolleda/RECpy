"""
Training / retraining time measurement.

We measure, on the largest training window (fold 4: 01-02 -> 11-30,
~11 months -> worst-case daily cost):
  * each deployed ML model: time to refit on the full univariate 192-lag design
    (the per-step walk-forward retrain cost), median of 5 fits;
  * each DL architecture (LSTM/GRU/TCN) with the per-SP selected configuration:
    time for one from-scratch fit (the DL_daily per-day retrain cost).

Hardware is a development workstation (CPU); labelled as such. ML refits are
light enough for the edge controllers, whereas DL retraining is far heavier and
motivates the Jetson GPU for any DL deployment.

Output: cross_validation/robustness/training_time.csv
"""

import time
import numpy as np
import pandas as pd
from pathlib import Path

import generate_ml_cv as ml
from generate_ml_cv import to_supervised as ml_to_supervised, get_models, N_INPUT, TRAIN_START, BASE

from recpy_config import CONFIG

# Calendar year of the dataset (placeholder;
# set this to the actual year of your data files).
YEAR = CONFIG.year

OUT = CONFIG.output_dir("cross_validation", "robustness") / "training_time.csv"
TRAIN_END = f"{YEAR}-11-30"          # fold 4 = largest expanding window

DEPLOYED = {"SP1": "Support Vector", "SP2": "k-Nearest Neighbors", "SP3": "Huber",
            "SP4": "Huber", "SP5": "Huber", "SP6": "Extra Trees"}
SP_NAME = {"SP1": "SP1", "SP2": "SP2", "SP3": "SP3",
           "SP4": "SP4", "SP5": "SP5", "SP6": "SP6"}


def time_ml():
    rows = []
    models = get_models()
    for sp in ml.SUPPLY_POINTS:
        series = pd.read_csv(BASE / sp["history_csv"],
                             parse_dates=["date_time"], index_col="date_time")[sp["history_col"]]
        s_train = series.loc[TRAIN_START:TRAIN_END]
        X, y = ml_to_supervised(s_train.values.astype(float), N_INPUT)
        name = DEPLOYED[sp["id"]]
        times = []
        for _ in range(5):
            model = get_models()[name]
            t0 = time.perf_counter()
            model.fit(X, y)
            times.append(time.perf_counter() - t0)
        rows.append({
            "supply_point": SP_NAME[sp["id"]], "type": "ML (deployed)",
            "model": name, "n_train_samples": len(y),
            "refit_ms_median": round(float(np.median(times)) * 1000, 1),
        })
        print(f"  ML  {SP_NAME[sp['id']]:<11} {name:<20} "
              f"refit={np.median(times)*1000:8.1f} ms  (n={len(y)})")
    return rows


def time_dl():
    import generate_dl_cv as dl
    from sklearn import preprocessing
    import random, tensorflow as tf

    rows = []
    for sp in dl.SUPPLY_POINTS:
        data = pd.read_csv(BASE / dl.METEO_CSV, sep=",", decimal=".",
                           index_col=0, parse_dates=["date_time"])
        data.drop(columns=sp["drop_cols"], inplace=True)
        train_df = data.loc[TRAIN_START:TRAIN_END]
        scaler = preprocessing.MinMaxScaler()
        train_scaled = scaler.fit_transform(train_df)
        train_w = np.array(np.split(train_scaled, len(train_scaled) // 24))

        for arch in ["lstm", "gru", "tcn"]:
            params = sp[arch]
            random.seed(dl.RS); np.random.seed(dl.RS); tf.random.set_seed(dl.RS)
            t0 = time.perf_counter()
            if arch == "lstm":
                n_input, nodes, epochs, batch, dropout = params
                dl.build_lstm(train_w, n_input, nodes, epochs, batch, dropout)
            elif arch == "gru":
                n_input, nodes, epochs, batch, dropout = params
                dl.build_gru(train_w, n_input, nodes, epochs, batch, dropout)
            else:
                n_input, filters, dilations, epochs, batch, dropout = params
                dl.build_tcn(train_w, n_input, filters, dilations, epochs, batch, dropout)
            dt = time.perf_counter() - t0
            rows.append({
                "supply_point": SP_NAME[sp["id"]], "type": f"DL_daily ({arch.upper()})",
                "model": arch.upper(), "n_train_samples": (len(train_w) * 24 - params[0]),
                "refit_ms_median": round(dt * 1000, 1),
            })
            print(f"  DL  {SP_NAME[sp['id']]:<11} {arch.upper():<20} train={dt:7.2f} s")
    return rows


def main():
    print("=== ML deployed-model refit time (fold 4, largest window) ===")
    rows = time_ml()
    print("\n=== DL from-scratch retrain time (per-SP selected config) ===")
    rows += time_dl()

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    ml_df = df[df.type == "ML (deployed)"]
    dl_df = df[df.type.str.startswith("DL")]
    print("\n=== Summary ===")
    print(f"ML deployed refit: {ml_df.refit_ms_median.min():.1f}-"
          f"{ml_df.refit_ms_median.max():.1f} ms per step")
    print(f"DL from-scratch retrain: {dl_df.refit_ms_median.min()/1000:.1f}-"
          f"{dl_df.refit_ms_median.max()/1000:.1f} s per day")
    print(f"DL/ML ratio (median): "
          f"{dl_df.refit_ms_median.median()/ml_df.refit_ms_median.median():.0f}x")
    print(f"Written {OUT}")


if __name__ == "__main__":
    main()
