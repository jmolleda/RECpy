"""
Blocked time-series cross-validation — DL predictions (LSTM, GRU, TCN).

Same four folds as generate_ml_cv.py. 
Uses the best hyperparameters identified (minimum MAE criterion).

Multivariate input: consumption + calendar + weather (18 features per SP).
MinMaxScaler fitted on the training portion of each fold only.

Output: cross_validation/fold{N}_{month}/Predictions_C_{SP}_fold{N}.csv
        Columns appended: LSTM, GRU, TCN  (to the ML CSV already written)
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from warnings import filterwarnings
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
from tcn import TCN

from recpy_config import CONFIG

# All settings come from the active configuration (config/default.toml unless
# RECPY_CONFIG points elsewhere); see code/recpy_config.py.
YEAR = CONFIG.year

filterwarnings("ignore")

os.environ["PYTHONHASHSEED"] = "0"
RS = CONFIG.random_seed
random.seed(RS)
np.random.seed(RS)
tf.random.set_seed(RS)

BASE        = Path(__file__).parent.parent
METEO_CSV   = "data/consumption_meteo_calendar.csv"
TRAIN_START = f"{YEAR}-01-02"

# ---------------------------------------------------------------------------
# Fold definitions (identical to generate_ml_cv.py)
# ---------------------------------------------------------------------------
FOLDS = [
    {k: f[k] for k in ("name", "train_end", "test_start", "test_end")}
    for f in CONFIG.folds
]

# ---------------------------------------------------------------------------
# Supply-point registry with best DL hyperparameters (by MAE)
#
# LSTM/GRU params: (n_input, nodes, epochs, batch, dropout)
# TCN params:      (n_input, filters, dilations, epochs, batch, dropout)
# ---------------------------------------------------------------------------
SUPPLY_POINTS = CONFIG.supply_points_dl()

# ---------------------------------------------------------------------------
# Core helpers (identical to generate_dl_predictions.py)
# ---------------------------------------------------------------------------
def to_supervised(train: np.ndarray, n_input: int, n_out: int = 24):
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))
    X, y = [], []
    for in_start in range(len(data)):
        in_end  = in_start + n_input
        out_end = in_end   + n_out
        if out_end <= len(data):
            X.append(data[in_start:in_end, :])
            y.append(data[in_end:out_end, 0])
    return np.array(X), np.array(y)


def forecast(model, history: list, n_input: int) -> np.ndarray:
    data    = np.array(history).reshape(-1, history[0].shape[1])
    input_x = data[-n_input:, :].reshape(1, n_input, data.shape[1])
    yhat    = model.predict(input_x, verbose=0)[0]
    yhat[yhat < 0] = 0.0
    return yhat


def build_lstm(train_w, n_input, nodes, epochs, batch, dropout):
    train_x, train_y = to_supervised(train_w, n_input)
    n_t, n_f, n_o = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_y = train_y.reshape(train_y.shape[0], n_o, 1)
    model = Sequential([LSTM(nodes, input_shape=(n_t, n_f), activation="tanh")])
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(n_o))
    model.compile(loss="mse", optimizer="adam")
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch, verbose=0)
    return model


def build_gru(train_w, n_input, nodes, epochs, batch, dropout):
    train_x, train_y = to_supervised(train_w, n_input)
    n_t, n_f, n_o = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_y = train_y.reshape(train_y.shape[0], n_o, 1)
    model = Sequential([GRU(nodes, input_shape=(n_t, n_f), activation="tanh")])
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(n_o))
    model.compile(loss="mse", optimizer="adam")
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch, verbose=0)
    return model


def build_tcn(train_w, n_input, filters, dilations, epochs, batch, dropout):
    train_x, train_y = to_supervised(train_w, n_input)
    n_t, n_f, n_o = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    train_y = train_y.reshape(train_y.shape[0], n_o, 1)
    model = Sequential([
        TCN(nb_filters=filters, kernel_size=3, nb_stacks=1,
            dilations=dilations, input_shape=(n_t, n_f)),
        Dropout(dropout),
        Dense(n_o),
    ])
    model.compile(loss="mse", optimizer="adam")
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch, verbose=0)
    return model


def walk_forward_predict(model, train_w, test_w, n_input):
    history = list(train_w)
    preds = []
    for i in range(len(test_w)):
        yhat = forecast(model, history, n_input)
        preds.append(yhat)
        history.append(test_w[i])   # expanding: no pop, matches notebook
    return np.concatenate(preds)


def inverse_scale(preds_scaled, test_scaled, scaler):
    p       = preds_scaled.reshape(-1, 1)
    combined = np.concatenate((p, test_scaled[:, 1:]), axis=1)
    inv     = scaler.inverse_transform(combined)
    return inv[:, 0].round(3)


# ---------------------------------------------------------------------------
# Per fold × supply point runner
# ---------------------------------------------------------------------------
def run_fold_sp(fold: dict, sp: dict, out_dir: Path) -> None:
    sp_id = sp["id"]

    # Load and prepare multivariate data
    data = pd.read_csv(
        BASE / METEO_CSV,
        sep=",", decimal=".",
        index_col=0, parse_dates=["date_time"]
    )
    data.drop(columns=sp["drop_cols"], inplace=True)

    train_df = data.loc[TRAIN_START:fold["train_end"]]
    test_df  = data.loc[fold["test_start"]:fold["test_end"]]

    scaler       = preprocessing.MinMaxScaler()
    train_scaled = scaler.fit_transform(train_df)
    test_scaled  = scaler.transform(test_df)

    train_w = np.array(np.split(train_scaled, len(train_scaled) // 24))
    test_w  = np.array(np.split(test_scaled,  len(test_scaled)  // 24))

    obs = scaler.inverse_transform(test_scaled)[:, 0]

    # Load the ML CSV written by generate_ml_cv.py to append DL columns
    csv_path = out_dir / f"Predictions_C_{sp_id}_{fold['name']}.csv"
    all_df   = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    for arch_name in ["lstm", "gru", "tcn"]:
        col_name = arch_name.upper() + "_single"
        if col_name in all_df.columns:
            print(f"    [{sp_id}] {col_name} already present — skipping")
            continue

        params = sp[arch_name]
        print(f"    [{sp_id}] {col_name} ...", end=" ", flush=True)

        random.seed(RS); np.random.seed(RS); tf.random.set_seed(RS)

        try:
            if arch_name == "lstm":
                n_input, nodes, epochs, batch, dropout = params
                model = build_lstm(train_w, n_input, nodes, epochs, batch, dropout)
            elif arch_name == "gru":
                n_input, nodes, epochs, batch, dropout = params
                model = build_gru(train_w, n_input, nodes, epochs, batch, dropout)
            else:
                n_input, filters, dilations, epochs, batch, dropout = params
                model = build_tcn(train_w, n_input, filters, dilations, epochs, batch, dropout)

            preds_scaled = walk_forward_predict(model, train_w, test_w, n_input)
            preds_inv    = inverse_scale(preds_scaled, test_scaled, scaler)

            mae  = mean_absolute_error(obs, preds_inv)
            rmse = np.sqrt(mean_squared_error(obs, preds_inv))
            print(f"MAE={mae:.4f}  RMSE={rmse:.4f}")
            all_df[col_name] = preds_inv

        except Exception as exc:
            print(f"FAILED ({exc})")

    all_df.to_csv(csv_path)
    n_models = len(all_df.columns) - 1
    print(f"    [{sp_id}] Updated -> {csv_path.name}  ({n_models} models)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out_base = CONFIG.output_dir("cross_validation")

    for fold in FOLDS:
        print(f"\n{'='*65}")
        print(f"  {fold['name'].upper()}  |  Test: {fold['test_start']} – {fold['test_end']}")
        print(f"{'='*65}")
        fold_dir = out_base / fold["name"]

        for sp in SUPPLY_POINTS:
            print(f"\n  --- {sp['id']} ---")
            run_fold_sp(fold, sp, fold_dir)

    print("\n\nDone.")
