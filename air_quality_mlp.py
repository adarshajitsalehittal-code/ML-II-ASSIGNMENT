"""
Air Quality Prediction using a Multi-Layer Perceptron (MLP)
Course : AIC205A - Machine Learning-II  |  Ramaiah University of Applied Sciences
-------------------------------------------------------------------------------
Task   : Regression - predict hourly CO concentration CO(GT) [mg/m^3] from a
         low-cost metal-oxide sensor array + meteorological + time features.
Data   : UCI "Air Quality" dataset (De Vito et al., 2008), 9,357 hourly records,
         Mar-2004 to Apr-2005, Italian city.

HOW TO RUN (local or Google Colab)
    1. Put AirQualityUCI.csv in the same folder (the script also auto-downloads it).
    2. pip install numpy pandas scikit-learn matplotlib seaborn
    3. python air_quality_mlp.py      (or paste into Colab, cell by cell)
All figures / tables are saved in the ./outputs folder.
"""
# %% 1. Imports & configuration -------------------------------------------------
import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib  # plots are shown on screen AND saved to ./outputs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             confusion_matrix, accuracy_score,
                             precision_recall_fscore_support, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
CSV = "AirQualityUCI.csv"
URL = ("https://raw.githubusercontent.com/PacktPublishing/"
       "Time-Series-Analysis-with-Python-Cookbook./main/datasets/Ch2/AirQualityUCI.csv")
sns.set_theme(style="whitegrid", context="notebook")
RES = {}  # everything we report is collected here and saved as results.json

# %% 2. Load data ---------------------------------------------------------------
raw = pd.read_csv(CSV if os.path.exists(CSV) else URL, sep=";")
raw = raw.dropna(how="all").reset_index(drop=True)
print("Raw shape:", raw.shape)
RES["raw_shape"] = list(raw.shape)

# %% 3. Missing values (-200 is the UCI missing-value tag) -----------------------
df = raw.copy()
df = df.replace(-200, np.nan)
miss = df.isna().sum()
miss_pct = (miss / len(df) * 100).round(2)
RES["missing_pct"] = miss_pct.to_dict()
print("\nMissing values (%):\n", miss_pct)

fig, ax = plt.subplots(figsize=(8, 4.2))
miss_pct.drop(["Date", "Time"]).sort_values().plot.barh(ax=ax, color="#d9534f")
ax.set_xlabel("Missing values (%)"); ax.set_title("Missing values per attribute (tag = -200)")
plt.tight_layout(); plt.savefig(f"{OUT}/fig01_missing_values.png", dpi=200); plt.show()

# %% 4. Pre-processing & feature engineering ------------------------------------
TARGET = "CO(GT)"
SENSORS = ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)"]
METEO = ["T", "RH", "AH"]
# NMHC(GT) is ~90 % missing -> dropped. C6H6(GT), NOx(GT), NO2(GT) are *reference
# analyser* outputs; they are excluded so the model learns from cheap sensors only.
df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"].str.replace(".", ":", regex=False),
                                format="%d/%m/%Y %H:%M:%S")
df = df.sort_values("datetime").reset_index(drop=True)
df["hour"] = df["datetime"].dt.hour
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)     # cyclic encoding of hour
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
df["dayofweek"] = df["datetime"].dt.dayofweek
df["month"] = df["datetime"].dt.month
FEATURES = SENSORS + METEO + ["hour_sin", "hour_cos", "dayofweek", "month"]

n0 = len(df)
df = df.dropna(subset=[TARGET])                  # cannot learn from unlabeled rows
n1 = len(df)
df = df.dropna(subset=SENSORS + METEO, how="all")  # rows where every sensor is missing
n2 = len(df)
RES["rows"] = {"start": n0, "after_target_drop": n1, "after_sensor_drop": n2}
print(f"\nRows: {n0} -> {n1} (target present) -> {n2} (sensors present)")
df = df.reset_index(drop=True)
df[["datetime"] + FEATURES + [TARGET]].to_csv(f"{OUT}/air_quality_clean.csv", index=False)

# descriptive plots
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(df[TARGET], bins=40, kde=True, ax=axs[0], color="#337ab7")
axs[0].set_xlabel("CO(GT) [mg/m$^3$]"); axs[0].set_title("Distribution of target")
df.set_index("datetime")[TARGET].resample("D").mean().plot(ax=axs[1], color="#5cb85c")
axs[1].set_ylabel("Daily mean CO(GT)"); axs[1].set_title("Daily mean CO over the year")
plt.tight_layout(); plt.savefig(f"{OUT}/fig02_target_distribution.png", dpi=200); plt.show()

fig, ax = plt.subplots(figsize=(8.5, 7))
corr = df[FEATURES + [TARGET]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, annot_kws={"size": 7})
ax.set_title("Correlation matrix"); plt.tight_layout()
plt.savefig(f"{OUT}/fig03_correlation.png", dpi=200); plt.show()
RES["corr_with_target"] = corr[TARGET].drop(TARGET).round(3).to_dict()
RES["target_stats"] = df[TARGET].describe().round(3).to_dict()

# %% 5. Train / validation / test split (70 / 15 / 15) --------------------------
X = df[FEATURES].values
y = df[TARGET].values
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, random_state=SEED)
X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, random_state=SEED)
imp = SimpleImputer(strategy="median").fit(X_tr)                  # fit on TRAIN only
sx = StandardScaler().fit(imp.transform(X_tr))
sy = StandardScaler().fit(y_tr.reshape(-1, 1))
prep = lambda A: sx.transform(imp.transform(A))
Xtr, Xva, Xte = prep(X_tr), prep(X_va), prep(X_te)
ytr_s = sy.transform(y_tr.reshape(-1, 1)).ravel()
RES["split"] = {"train": len(y_tr), "val": len(y_va), "test": len(y_te)}
print("Split:", RES["split"])


# %% 6. Helper functions --------------------------------------------------------
def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "R2": r2_score(y_true, y_pred)}


def inv(a):
    return sy.inverse_transform(np.asarray(a).reshape(-1, 1)).ravel()


def train_mlp(hidden=(64, 32), activation="relu", lr=1e-3, epochs=150, batch=64,
              alpha=1e-4, seed=SEED, Xa=None, ya=None, Xb=None, yb=None):
    """Mini-batch Adam training with back-propagation. One partial_fit call = one
    epoch, so train/validation loss can be recorded after every epoch."""
    Xa = Xtr if Xa is None else Xa
    ya = ytr_s if ya is None else ya
    Xb = Xva if Xb is None else Xb
    yb = sy.transform(y_va.reshape(-1, 1)).ravel() if yb is None else yb
    net = MLPRegressor(hidden_layer_sizes=hidden, activation=activation, solver="adam",
                       learning_rate_init=lr, batch_size=batch, alpha=alpha,
                       random_state=seed, max_iter=1)
    hist = {"train_loss": [], "val_loss": [], "train_r2": [], "val_r2": []}
    rng = np.random.RandomState(seed)
    for ep in range(epochs):
        idx = rng.permutation(len(Xa))                          # shuffle each epoch
        net.partial_fit(Xa[idx], ya[idx])
        ptr, pva = net.predict(Xa), net.predict(Xb)
        hist["train_loss"].append(mean_squared_error(ya, ptr))
        hist["val_loss"].append(mean_squared_error(yb, pva))
        hist["train_r2"].append(r2_score(ya, ptr))
        hist["val_r2"].append(r2_score(yb, pva))
    return net, hist


def val_scores(net):
    return metrics(y_va, inv(net.predict(Xva)))


# %% 7. Baseline model: Linear Regression ---------------------------------------
lin = LinearRegression().fit(Xtr, ytr_s)
RES["linear_val"] = metrics(y_va, inv(lin.predict(Xva)))
RES["linear_test"] = metrics(y_te, inv(lin.predict(Xte)))
print("\nLinear regression (val):", {k: round(v, 3) for k, v in RES["linear_val"].items()})

# %% 8. MLP architecture diagram ------------------------------------------------
def draw_arch(layers, labels, path):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    xs = np.linspace(0.1, 0.9, len(layers))
    cap = 8
    pos = []
    for x, n in zip(xs, layers):
        k = min(n, cap)
        ys = np.linspace(0.85, 0.15, k)
        pos.append([(x, yv) for yv in ys])
    for a, b in zip(pos[:-1], pos[1:]):
        for p in a:
            for q in b:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#bbbbbb", lw=0.4, zorder=1)
    cols = ["#5bc0de", "#f0ad4e", "#f0ad4e", "#5cb85c"]
    for i, (pl, n) in enumerate(zip(pos, layers)):
        for (px, py) in pl:
            ax.scatter(px, py, s=380, color=cols[min(i, 3) if i < len(layers) - 1 else 3],
                       edgecolor="k", zorder=3)
        if n > cap:
            ax.text(pl[0][0], 0.05, f"... ({n} neurons)", ha="center", fontsize=8)
        ax.text(pl[0][0], 0.95, labels[i], ha="center", fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("MLP architecture used: 12 -> 64 -> 32 -> 1 (ReLU hidden, linear output)")
    plt.tight_layout(); plt.savefig(path, dpi=200); plt.show()

draw_arch([len(FEATURES), 64, 32, 1],
          [f"Input\n({len(FEATURES)} features)", "Hidden 1\n64 | ReLU", "Hidden 2\n32 | ReLU", "Output\nCO(GT) | linear"],
          f"{OUT}/fig04_architecture.png")

# %% 9. Baseline MLP training ---------------------------------------------------
EPOCHS = 150
net, hist = train_mlp(hidden=(64, 32), activation="relu", lr=1e-3, epochs=EPOCHS)
best_ep = int(np.argmin(hist["val_loss"])) + 1
RES["baseline_hist_summary"] = {"epochs": EPOCHS, "best_val_epoch": best_ep,
                                "final_train_loss": hist["train_loss"][-1],
                                "final_val_loss": hist["val_loss"][-1],
                                "min_val_loss": min(hist["val_loss"])}
RES["mlp_val"] = val_scores(net)
print("MLP baseline (val):", {k: round(v, 3) for k, v in RES["mlp_val"].items()}, "best epoch", best_ep)

ep = np.arange(1, EPOCHS + 1)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(ep, hist["train_loss"], label="Training loss")
axs[0].plot(ep, hist["val_loss"], label="Validation loss")
axs[0].axvline(best_ep, ls="--", color="gray", lw=1, label=f"Min val. loss (epoch {best_ep})")
axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("MSE (standardised target)")
axs[0].set_title("Training vs validation loss"); axs[0].legend()
axs[1].plot(ep, hist["train_r2"], label="Training R$^2$")
axs[1].plot(ep, hist["val_r2"], label="Validation R$^2$")
axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("R$^2$"); axs[1].set_ylim(0.5, 1.0)
axs[1].set_title("R$^2$ (accuracy for regression) vs epochs"); axs[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig05_training_curves.png", dpi=200); plt.show()

# %% 10. Parameter experimentation (evaluated on VALIDATION set) ------------------
def run_grid(name, configs, key):
    rows = []
    for label, kw in configs:
        n_, h_ = train_mlp(**kw)
        m = val_scores(n_)
        rows.append({key: label, "Val MAE": m["MAE"], "Val RMSE": m["RMSE"], "Val R2": m["R2"],
                     "Final train loss": h_["train_loss"][-1], "Final val loss": h_["val_loss"][-1]})
        print(f"[{name}] {label:>18s}  RMSE={m['RMSE']:.4f}  R2={m['R2']:.4f}")
    t = pd.DataFrame(rows)
    t.to_csv(f"{OUT}/exp_{name}.csv", index=False)
    return t

E = 100  # epochs used in the sweeps
t_lr = run_grid("learning_rate", [(str(l), dict(lr=l, epochs=E)) for l in [1e-4, 1e-3, 1e-2, 1e-1]], "Learning rate")
t_arch = run_grid("architecture", [
    ("(16)", dict(hidden=(16,), epochs=E)),
    ("(64)", dict(hidden=(64,), epochs=E)),
    ("(64,32)", dict(hidden=(64, 32), epochs=E)),
    ("(128,64,32)", dict(hidden=(128, 64, 32), epochs=E)),
    ("(256,128,64,32)", dict(hidden=(256, 128, 64, 32), epochs=E))], "Hidden layers")
t_act = run_grid("activation", [(a, dict(activation=a, epochs=E)) for a in ["relu", "tanh", "logistic", "identity"]],
                 "Activation")
# epochs experiment: one long run, read validation score at checkpoints
net_e, h_e = train_mlp(epochs=300)
rows = []
for e in [10, 25, 50, 100, 150, 200, 300]:
    rows.append({"Epochs": e, "Train loss": h_e["train_loss"][e - 1], "Val loss": h_e["val_loss"][e - 1],
                 "Val R2": h_e["val_r2"][e - 1]})
t_ep = pd.DataFrame(rows); t_ep.to_csv(f"{OUT}/exp_epochs.csv", index=False)
print(t_ep)

fig, axs = plt.subplots(2, 2, figsize=(11, 8))
for ax, t, k, ttl in [(axs[0, 0], t_lr, "Learning rate", "Learning rate"),
                      (axs[0, 1], t_arch, "Hidden layers", "Hidden layers / neurons"),
                      (axs[1, 0], t_act, "Activation", "Activation function")]:
    sns.barplot(data=t, x=k, y="Val RMSE", ax=ax, color="#337ab7")
    ax.set_title(ttl); ax.set_ylabel("Validation RMSE [mg/m$^3$]")
    ax.tick_params(axis="x", rotation=15)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
axs[1, 1].plot(np.arange(1, 301), h_e["train_loss"], label="Train")
axs[1, 1].plot(np.arange(1, 301), h_e["val_loss"], label="Validation")
axs[1, 1].set_title("Number of epochs"); axs[1, 1].set_xlabel("Epoch"); axs[1, 1].set_ylabel("MSE")
axs[1, 1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig06_parameter_experiments.png", dpi=200); plt.show()

# %% 11. Select best configuration by validation RMSE ------------------------------
best_lr = float(t_lr.loc[t_lr["Val RMSE"].idxmin(), "Learning rate"])
best_arch = t_arch.loc[t_arch["Val RMSE"].idxmin(), "Hidden layers"]
best_act = t_act.loc[t_act["Val RMSE"].idxmin(), "Activation"]
arch_map = {"(16)": (16,), "(64)": (64,), "(64,32)": (64, 32), "(128,64,32)": (128, 64, 32),
            "(256,128,64,32)": (256, 128, 64, 32)}
RES["best_config"] = {"lr": best_lr, "hidden": best_arch, "activation": best_act}
print("Best config:", RES["best_config"])

# Final model: train on TRAIN, stop at the epoch with minimum validation loss
net_f, h_f = train_mlp(hidden=arch_map[best_arch], activation=best_act, lr=best_lr, epochs=120)
best_ep_f = int(np.argmin(h_f["val_loss"])) + 1
net_f, h_f = train_mlp(hidden=arch_map[best_arch], activation=best_act, lr=best_lr, epochs=best_ep_f)
RES["final_epochs"] = best_ep_f

pred_tr = inv(net_f.predict(Xtr)); pred_va = inv(net_f.predict(Xva)); pred_te = inv(net_f.predict(Xte))
RES["final_train"] = metrics(y_tr, pred_tr)
RES["final_val"] = metrics(y_va, pred_va)
RES["final_test"] = metrics(y_te, pred_te)
RES["baseline_mlp_test"] = metrics(y_te, inv(net.predict(Xte)))
print("FINAL TEST:", {k: round(v, 4) for k, v in RES["final_test"].items()})

# %% 12. Evaluation plots --------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
axs[0].scatter(y_te, pred_te, s=8, alpha=0.4, color="#337ab7")
lim = [0, max(y_te.max(), pred_te.max())]
axs[0].plot(lim, lim, "r--", lw=1); axs[0].set_xlabel("Actual CO(GT)"); axs[0].set_ylabel("Predicted CO(GT)")
axs[0].set_title(f"Actual vs predicted (test, R$^2$={RES['final_test']['R2']:.3f})")
res_ = y_te - pred_te
sns.histplot(res_, bins=40, kde=True, ax=axs[1], color="#5cb85c")
axs[1].set_title("Residual distribution (test)"); axs[1].set_xlabel("Actual - predicted [mg/m$^3$]")
axs[2].scatter(pred_te, res_, s=8, alpha=0.4, color="#f0ad4e"); axs[2].axhline(0, color="r", lw=1)
axs[2].set_xlabel("Predicted"); axs[2].set_ylabel("Residual"); axs[2].set_title("Residuals vs predicted")
plt.tight_layout(); plt.savefig(f"{OUT}/fig07_test_evaluation.png", dpi=200); plt.show()

# time-series view of one week from the test set (sorted by time)
te_idx = np.arange(len(df))
_, tmp_i = train_test_split(te_idx, test_size=0.30, random_state=SEED)
_, te_i = train_test_split(tmp_i, test_size=0.50, random_state=SEED)
tdf = pd.DataFrame({"t": df["datetime"].values[te_i], "actual": y_te, "pred": pred_te}).sort_values("t")
tdf = tdf.iloc[:150]
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(tdf["t"], tdf["actual"], label="Actual", lw=1.4)
ax.plot(tdf["t"], tdf["pred"], label="MLP prediction", lw=1.4, alpha=0.85)
ax.set_ylabel("CO(GT) [mg/m$^3$]"); ax.set_title("Test-set samples in time order (first 150)"); ax.legend()
plt.tight_layout(); plt.savefig(f"{OUT}/fig08_timeseries.png", dpi=200); plt.show()

# %% 13. Classification view -> confusion matrix (Low / Medium / High CO) ---------
q1, q2 = np.quantile(y_tr, [1 / 3, 2 / 3])
cls = lambda v: np.digitize(v, [q1, q2])
names = ["Low", "Medium", "High"]
yc, pc = cls(y_te), cls(pred_te)
cm = confusion_matrix(yc, pc)
acc = accuracy_score(yc, pc)
pr, rc, f1, _ = precision_recall_fscore_support(yc, pc, average="macro")
RES["classification"] = {"thresholds": [float(q1), float(q2)], "accuracy": acc, "precision_macro": pr,
                         "recall_macro": rc, "f1_macro": f1, "confusion_matrix": cm.tolist()}
fig, ax = plt.subplots(figsize=(5.2, 4.6))
ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion matrix (test)\nLow<{q1:.2f}, Medium<{q2:.2f}, High>={q2:.2f} mg/m$^3$")
plt.tight_layout(); plt.savefig(f"{OUT}/fig09_confusion_matrix.png", dpi=200); plt.show()
print("Pollution-level accuracy:", round(acc, 3), "macro-F1:", round(f1, 3))

# %% 14. Extra check: chronological split (honest estimate under sensor drift) -----
n_all = len(df); cut = int(0.8 * n_all)
Xc_tr, Xc_te = X[:cut], X[cut:]; yc_tr, yc_te = y[:cut], y[cut:]
imp2 = SimpleImputer(strategy="median").fit(Xc_tr)
sx2 = StandardScaler().fit(imp2.transform(Xc_tr)); sy2 = StandardScaler().fit(yc_tr.reshape(-1, 1))
A = sx2.transform(imp2.transform(Xc_tr)); B = sx2.transform(imp2.transform(Xc_te))
net_c = MLPRegressor(hidden_layer_sizes=arch_map[best_arch], activation=best_act,
                     learning_rate_init=best_lr, batch_size=64, alpha=1e-4,
                     random_state=SEED, max_iter=best_ep_f)
net_c.fit(A, sy2.transform(yc_tr.reshape(-1, 1)).ravel())
pc_ = sy2.inverse_transform(net_c.predict(B).reshape(-1, 1)).ravel()
RES["chronological_test"] = metrics(yc_te, pc_)
print("Chronological split test:", {k: round(v, 3) for k, v in RES["chronological_test"].items()})

# %% 15. Summary tables & save ---------------------------------------------------
summary = pd.DataFrame({
    "Linear Regression (baseline)": RES["linear_test"],
    "MLP baseline 64-32 ReLU": RES["baseline_mlp_test"],
    f"MLP tuned {best_arch} {best_act} lr={best_lr}": RES["final_test"]}).T
summary.to_csv(f"{OUT}/model_comparison_test.csv")
print(summary.round(4))
with open(f"{OUT}/results.json", "w") as f:
    json.dump(RES, f, indent=2, default=float)
print("\nDone. Files saved in ./outputs")
