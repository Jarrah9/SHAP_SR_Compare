import pandas as pd
import numpy as np
import logging
from itertools import product
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

# ── Data ──────────────────────────────────────────────────────────────────────
col_names = ["m2_m1", "x2_x1", "y1_x1", "y2_x1", "z1_x1", "z2_x1", "target"]
UGravSample = pd.read_csv(
    "Feynman_without_units/I.9.18_sample_10000",
    sep=" ",
    index_col=False,
    header=None,
    names=col_names
)

UGravX = UGravSample.drop(columns=["target"])
UGravY = UGravSample["target"]

X_train, X_test, y_train, y_test = train_test_split(
    UGravX, UGravY, test_size=0.2, random_state=42
)

# ── Configs ───────────────────────────────────────────────────────────────────
population_pairs = [(33, 15), (50, 20), (100, 40), (200, 80)]
niterations      = [200, 400, 1000] # 1000 not necessary
parsimony_values = [0.001, 0.0032, 0.01, 0.05]
adaptive_values  = [500, 1000, 3000, 10000] # Try lower as well

configs = [
    {
        "population_size": ps,
        "populations": p,
        "niterations": ni,
        "parsimony": par,
        "adaptive_parsimony_scaling": adp,
    }
    for (ps, p), ni, par, adp in product(
        population_pairs, niterations, parsimony_values, adaptive_values
    )
]
log.info(f"Total configs: {len(configs)}")

# ── Run ───────────────────────────────────────────────────────────────────────
summary_rows = []

for i, config in enumerate(configs):
    label = (f"ps{config['population_size']}_p{config['populations']}_"
             f"ni{config['niterations']}_par{config['parsimony']}_"
             f"adp{config['adaptive_parsimony_scaling']}")

    log.info(f"[{i+1}/{len(configs)}] Running: {label}")

    model = PySRRegressor(
        **config,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square"],
        extra_sympy_mappings={"square": lambda x: x**2},
        maxsize=20,
        parallelism="multithreading",
        random_state=42
    )

    model.fit(X_train, y_train)

    best_idx = model.equations_["loss"].idxmin()
    best_eq  = model.equations_.iloc[best_idx]

    y_pred   = model.predict(X_test, index=best_idx)
    test_mse = mean_squared_error(y_test, y_pred)
    r2       = r2_score(y_test, y_pred)

    summary_rows.append({
        "label":                    label,
        "population_size":          config["population_size"],
        "populations":              config["populations"],
        "niterations":              config["niterations"],
        "parsimony":                config["parsimony"],
        "adaptive_parsimony_scaling": config["adaptive_parsimony_scaling"],
        "best_loss":                best_eq["loss"],
        "complexity":               best_eq["complexity"],
        "test_mse":                 test_mse,
        "r2":                       r2,
        "equation":                 str(best_eq["equation"]),
    })

    # Save incrementally so results are not lost if the job times out
    pd.DataFrame(summary_rows).to_csv("pysr_results.csv", index=False)
    log.info(f"  loss={best_eq['loss']:.6f}  test_mse={test_mse:.6f}"
             f"  r2={r2:.6f}  complexity={best_eq['complexity']}")
    log.info(f"  equation: {best_eq['equation']}")

# ── Final summary ─────────────────────────────────────────────────────────────
summary = pd.DataFrame(summary_rows).sort_values("test_mse")
log.info("Final results (sorted by test MSE):\n%s",
         summary.to_string(index=False))
summary.to_csv("pysr_results_final.csv", index=False)
log.info("Done.")
