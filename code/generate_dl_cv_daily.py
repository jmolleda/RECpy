"""
Blocked time-series cross-validation — DL predictions with daily retraining.

Same four folds and supply points as generate_dl_cv.py, but the model is
rebuilt from scratch at every test step on the full expanding history
(initial training data + all observed test days so far).

This mirrors the ML walk-forward approach and simulates a production scenario
where the DL model is retrained nightly as new data arrives.

Output: cross_validation/fold{N}_{month}/Predictions_C_{SP}_fold{N}.csv
        Columns appended: LSTM_daily, GRU_daily, TCN_daily
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from warnings import filterwarnings
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout
from tcn import TCN

# Calendar year of the dataset (placeholder;
# set this to the actual year of your data files).
YEAR = 2025

filterwarnings("ignore")

os.environ["PYTHONHASHSEED"] = "0"
RS = 123
random.seed(RS)
np.random.seed(RS)
tf.random.set_seed(RS)

BASE        = Path(__file__).parent.parent
METEO_CSV   = "data/consumption_meteo_calendar.csv"
TRAIN_START = f"{YEAR}-01-02"

# ---------------------------------------------------------------------------
# Fold definitions (identical to generate_dl_cv.py)
# ---------------------------------------------------------------------------
FOLDS = [
    {"name": "fold1_apr", "train_end": f"{YEAR}-03-31", "test_start": f"{YEAR}-04-01", "test_end": f"{YEAR}-04-30"},
    {"name": "fold2_jul", "train_end": f"{YEAR}-06-30", "test_start": f"{YEAR}-07-01", "test_end": f"{YEAR}-07-31"},
    {"name": "fold3_oct", "train_end": f"{YEAR}-09-30", "test_start": f"{YEAR}-10-01", "test_end": f"{YEAR}-10-31"},
    {"name": "fold4_dec", "train_end": f"{YEAR}-11-30", "test_start": f"{YEAR}-12-01", "test_end": f"{YEAR}-12-31"},
]

# ---------------------------------------------------------------------------
# Supply-point registry with best DL hyperparameters (identical to generate_dl_cv.py)
# ---------------------------------------------------------------------------
SUPPLY_POINTS = [
    {
        "id": "SP1", "folder": "SP1",
        "drop_cols": ["C_SP2", "C_SP3", "C_SP4", "C_SP5", "C_SP6"],
        "lstm": (72,  30, 50, 32, 0.0),
        "gru":  (192, 25, 20, 32, 0.0),
        "tcn":  (144, 18, [1, 2, 4], 20, 32, 0.0),
    },
    {
        "id": "SP2", "folder": "SP2",
        "drop_cols": ["C_SP1", "C_SP3", "C_SP4", "C_SP5", "C_SP6"],
        "lstm": (72,  18, 20, 64, 0.0),
        "gru":  (72,  25, 50, 32, 0.2),
        "tcn":  (72,  25, [1, 2, 4], 50, 32, 0.0),
    },
    {
        "id": "SP3", "folder": "SP3",
        "drop_cols": ["C_SP1", "C_SP2", "C_SP4", "C_SP5", "C_SP6"],
        "lstm": (144, 25, 50, 64, 0.0),
        "gru":  (72,  25, 50, 32, 0.0),
        "tcn":  (72,  25, [1, 2, 4], 50, 32, 0.0),
    },
    {
        "id": "SP4", "folder": "SP4",
        "drop_cols": ["C_SP1", "C_SP2", "C_SP3", "C_SP5", "C_SP6"],
        "lstm": (72,  30, 50, 32, 0.0),
        "gru":  (144, 25, 50, 64, 0.0),
        "tcn":  (192, 30, [1, 2, 4, 8], 50, 64, 0.2),
    },
    {
        "id": "SP5", "folder": "SP5",
        "drop_cols": ["C_SP1", "C_SP2", "C_SP3", "C_SP4", "C_SP6"],
        "lstm": (72,  25, 50, 32, 0.2),
        "gru":  (144, 25, 50, 32, 0.0),
        "tcn":  (72,  18, [1, 2, 4], 20, 32, 0.0),
    },
{
        "id": "SP6", "folder": "SP6",
        "drop_cols": ["C_SP1", "C_SP2", "C_SP3", "C_SP4", "C_SP5"],
        "lstm": (72,  25, 20, 32, 0.0),
        "gru":  (72,  18, 20, 32, 0.0),
        "tcn":  (144, 25, [1, 2, 4, 8], 50, 32, 0.0),
    },
]

# ---------------------------------------------------------------------------
# Core helpers
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


def build_model(arch: str, train_w: np.ndarray, params: tuple) -> Sequential:
    """Build and fit a fresh model on the current history."""
    random.seed(RS); np.random.seed(RS); tf.random.set_seed(RS)

    if arch in ("lstm", "gru"):
        n_input, nodes, epochs, batch, dropout = params
        train_x, train_y = to_supervised(train_w, n_input)
        n_t, n_f, n_o = train_x.shape[1], train_x.shape[2], train_y.shape[1]
        train_y = train_y.reshape(train_y.shape[0], n_o, 1)
        layer = LSTM(nodes, input_shape=(n_t, n_f), activation="tanh") if arch == "lstm" \
                else GRU(nodes, input_shape=(n_t, n_f), activation="tanh")
        model = Sequential([layer])
        if dropout > 0:
            model.add(Dropout(dropout))
        model.add(Dense(n_o))
    else:  # tcn
        n_input, filters, dilations, epochs, batch, dropout = params
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
    return model, n_input


def walk_forward_daily(arch: str, params: tuple,
                       train_w: np.ndarray, test_w: np.ndarray) -> np.ndarray:
    """
    Re-train from scratch at every test step on the full expanding history.
    Mirrors the ML walk_forward approach in generate_ml_cv.py.
    """
    history = list(train_w)
    preds   = []
    n_days  = len(test_w)

    for day_idx in range(n_days):
        history_arr = np.array(history)
        model, n_input = build_model(arch, history_arr, params)
        yhat = forecast(model, history, n_input)
        preds.append(yhat)
        history.append(test_w[day_idx])   # expand history with observed day

        if (day_idx + 1) % 5 == 0 or (day_idx + 1) == n_days:
            print(f"      step {day_idx + 1}/{n_days}", end="\r", flush=True)

    print()
    return np.concatenate(preds)


def inverse_scale(preds_scaled: np.ndarray, test_scaled: np.ndarray,
                  scaler) -> np.ndarray:
    p        = preds_scaled.reshape(-1, 1)
    combined = np.concatenate((p, test_scaled[:, 1:]), axis=1)
    inv      = scaler.inverse_transform(combined)
    return inv[:, 0].round(3)


# ---------------------------------------------------------------------------
# Per fold × supply point runner
# ---------------------------------------------------------------------------
def run_fold_sp(fold: dict, sp: dict, out_dir: Path) -> None:
    sp_id = sp["id"]

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

    csv_path = out_dir / f"Predictions_C_{sp_id}_{fold['name']}.csv"
    all_df   = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    for arch_name in ["lstm", "gru", "tcn"]:
        col_name = arch_name.upper() + "_daily"
        if col_name in all_df.columns:
            print(f"    [{sp_id}] {col_name} already present — skipping")
            continue

        params = sp[arch_name]
        print(f"    [{sp_id}] {col_name} (daily retrain, {len(test_w)} steps) ...")

        try:
            preds_scaled = walk_forward_daily(arch_name, params, train_w, test_w)
            preds_inv    = inverse_scale(preds_scaled, test_scaled, scaler)

            mae  = mean_absolute_error(obs, preds_inv)
            rmse = np.sqrt(mean_squared_error(obs, preds_inv))
            print(f"      MAE={mae:.4f}  RMSE={rmse:.4f}")
            all_df[col_name] = preds_inv

        except Exception as exc:
            print(f"      FAILED ({exc})")

    all_df.to_csv(csv_path)
    n_models = len(all_df.columns) - 1
    print(f"    [{sp_id}] Updated -> {csv_path.name}  ({n_models} models)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    out_base = Path(__file__).parent / "cross_validation"

    for fold in FOLDS:
        print(f"\n{'='*65}")
        print(f"  {fold['name'].upper()}  |  Test: {fold['test_start']} – {fold['test_end']}")
        print(f"{'='*65}")
        fold_dir = out_base / fold["name"]

        for sp in SUPPLY_POINTS:
            print(f"\n  --- {sp['id']} ---")
            run_fold_sp(fold, sp, fold_dir)

    print("\n\nDone.")
