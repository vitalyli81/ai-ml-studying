"""Learn more Python by building Feature Engineering.

Part 12 — the pandas part. Eleven parts ate clean NumPy arrays; real data is a
CSV full of strings, blanks, dates, and one impossible mansion. This part is
the doc's end-to-end housing walkthrough: load → inspect → impute → encode →
create → cap → pipeline, with the model as the last 5%.
  STEP 1: load & inspect            (read_csv, dtypes, isna().sum(), value_counts)
  STEP 2: missing values            (fillna + an is_missing flag == SimpleImputer)
  STEP 3: encoding — the trap, measured  (label-encoded cities vs one-hot, in R²)
  STEP 4: feature creation          (.dt dates, ratios, groupby().transform, pd.cut)
  STEP 5: the outlier               (one fake mansion destroys R²; .clip rescues it)
  STEP 6: the ladder + the pipeline (features beat models; ColumnTransformer + GridSearchCV)

Theory companion: ../../ml/feature-engineering.md

Run from python/ml-practice/:
    uv run feature-engineering/feature_engineering.py   (~10s)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # render to file, no GUI window
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = Path(__file__).parent / "houses.csv"
TODAY = pd.Timestamp("2026-06-10")

# Deliberately NON-monotonic in alphabetical order — so numbering the cities
# 0,1,2,3,4 invents a ranking the prices don't follow. That's the trap.
CITY_EFFECT = {"austin": 120_000, "dallas": 0, "miami": 200_000,
               "nyc": 60_000, "seattle": 90_000}
CONDITION_EFFECT = {"poor": 0, "fair": 20_000, "good": 40_000, "excellent": 65_000}


def write_messy_csv(n: int = 600, seed: int = 42) -> None:
    """Real-world data, simulated: blanks, date strings, and one typo'd mansion."""
    rng = np.random.default_rng(seed)
    sqft = rng.uniform(500, 3500, n).round(0)
    bedrooms = np.clip((sqft / 700 + rng.normal(0, 0.8, n)).round(0), 1, 6)
    city = rng.choice(list(CITY_EFFECT), size=n)
    condition = rng.choice(list(CONDITION_EFFECT), size=n, p=[0.15, 0.3, 0.4, 0.15])
    built_year = rng.integers(1900, 2026, n)
    sold = np.array([f"{y}-{m:02d}-{d:02d}" for y, m, d in
                     zip(rng.integers(2000, 2026, n), rng.integers(1, 13, n),
                         rng.integers(1, 29, n))], dtype=object)
    years_since_sold = (TODAY - pd.to_datetime(pd.Series(sold))).dt.days / 365.25

    price = (60_000 + 140 * sqft + 9_000 * bedrooms
             + np.array([CITY_EFFECT[c] for c in city])
             + np.array([CONDITION_EFFECT[c] for c in condition])
             - 900 * (2026 - built_year)
             + 40_000 * (built_year >= 2000)            # modern-build premium (a step!)
             - 20_000 * (sqft / bedrooms < 350)         # cramped-layout penalty (a ratio!)
             - 1_200 * years_since_sold.to_numpy()      # recent sale ≈ recently renovated
             + rng.normal(0, 30_000, n))

    sold[rng.uniform(size=n) < 0.12] = ""               # 12% never recorded
    sqft_col = sqft.astype(object)
    sqft_col[rng.uniform(size=n) < 0.05] = ""           # 5% missing sqft
    price[0] = 85_000_000                               # the data-entry mansion

    pd.DataFrame({"sqft": sqft_col, "bedrooms": bedrooms.astype(int),
                  "city": city, "condition": condition, "built_year": built_year,
                  "last_sold": sold, "price": price.round(0).astype(np.int64)}
                 ).to_csv(CSV_PATH, index=False)


def ridge_r2(features: pd.DataFrame, target: pd.Series) -> float:
    pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    return float(cross_val_score(pipe, features, target, cv=5, scoring="r2").mean())


def main() -> None:
    write_messy_csv()

    # STEP 1 — load and LOOK before touching anything
    print("STEP 1 — load & inspect (the doc's rule: model.fit is the last 5%):")
    df = pd.read_csv(CSV_PATH, parse_dates=["last_sold"])   # strings → Timestamps
    print(f"    shape {df.shape}, dtypes: "
          f"{ {c: str(t) for c, t in df.dtypes.items()} }")
    print(f"    missing per column: {df.isna().sum().to_dict()}")
    print(f"    city counts: {df['city'].value_counts().to_dict()}")
    print(f"    price: median ${df['price'].median():,.0f}, "
          f"max ${df['price'].max():,.0f}  ← that max is no house\n")

    # STEP 2 — missing values: impute + CONFESS via an indicator column
    print("STEP 2 — impute with the median, and keep an is_missing flag:")
    df["sqft_was_missing"] = df["sqft"].isna().astype(int)
    median_sqft = df["sqft"].median()
    df["sqft"] = df["sqft"].fillna(median_sqft)
    sk_imputed = SimpleImputer(strategy="median").fit_transform(
        pd.read_csv(CSV_PATH)[["sqft"]])
    assert np.allclose(df["sqft"].to_numpy(), sk_imputed.ravel())
    print(f"    {df['sqft_was_missing'].sum()} blank sqft → filled with median "
          f"{median_sqft:.0f} (== SimpleImputer, asserted)")
    print("    → 0 would LIE (a 0-sqft house); the flag keeps "
          "'it was blank' as information\n")

    # STEP 4's date math needs doing before we drop the raw column
    df["years_since_sold"] = (TODAY - df["last_sold"]).dt.days / 365.25
    df["sold_was_missing"] = df["years_since_sold"].isna().astype(int)
    df["years_since_sold"] = df["years_since_sold"].fillna(
        df["years_since_sold"].median())

    # STEP 3 — encoding: quantify the red=1/blue=2/green=3 trap
    print("STEP 3 — the encoding trap, measured (Ridge, 5-fold CV R²):")
    target = df["price"].clip(upper=df["price"].quantile(0.99))  # mansion handled
    base = df[["sqft", "bedrooms", "built_year"]]

    labeled = base.assign(city=df["city"].astype("category").cat.codes)
    onehot = pd.concat([base, pd.get_dummies(df["city"], prefix="city", dtype=int)],
                       axis=1)
    ohe = OneHotEncoder(sparse_output=False).fit_transform(df[["city"]])
    assert np.array_equal(ohe, pd.get_dummies(df["city"], dtype=int).to_numpy())
    print(f"    city as 0..4 codes:  R² = {ridge_r2(labeled, target):.3f}   "
          "(model forced to believe austin < dallas < miami...)")
    print(f"    city one-hot:        R² = {ridge_r2(onehot, target):.3f}   "
          "(each city its own column — pd.get_dummies == OneHotEncoder, asserted)")
    df["condition_rank"] = df["condition"].map(
        {"poor": 1, "fair": 2, "good": 3, "excellent": 4})
    print("    condition IS ordered (poor<fair<good<excellent) → .map to 1..4 "
          "is correct there\n")

    # STEP 4 — feature creation: ratios, dates, group context, bins
    print("STEP 4 — created features (questions, encoded as columns):")
    df["house_age"] = 2026 - df["built_year"]
    df["sqft_per_bedroom"] = df["sqft"] / df["bedrooms"]
    df["city_avg_sqft"] = df.groupby("city")["sqft"].transform("mean")
    df["sqft_vs_city"] = df["sqft"] / df["city_avg_sqft"]
    df["age_bucket"] = pd.cut(df["house_age"], bins=[0, 10, 30, 60, 130],
                              labels=["new", "modern", "older", "historic"])
    print("    'how old?'            house_age, age_bucket (pd.cut bins 0-10-30-60-130)")
    print("    'how long since sale?' years_since_sold (Timestamp math via .dt.days)")
    print("    'cramped?'            sqft_per_bedroom (a ratio a linear model can't invent)")
    print("    'big FOR ITS city?'   sqft / groupby('city').transform('mean')")
    print(f"    age buckets: {df['age_bucket'].value_counts().to_dict()}\n")

    # STEP 5 — the outlier: measure the damage, then cap it
    print("STEP 5 — one typo'd mansion vs the IQR fence:")
    q1, q3 = np.percentile(df["price"], [25, 75])
    fence = q3 + 1.5 * (q3 - q1)
    outliers = df[df["price"] > fence]
    print(f"    IQR fence at ${fence:,.0f} → {len(outliers)} row(s) beyond it, "
          f"worst: ${df['price'].max():,.0f}")
    print(f"    Ridge CV R², price raw:    {ridge_r2(onehot, df['price']):.3f}  "
          "(one fake row poisons the squared loss)")
    print(f"    Ridge CV R², price capped: {ridge_r2(onehot, target):.3f}  "
          "(.clip at the 99th percentile)")
    print("    → but remember the doc: in fraud/churn, outliers ARE the signal — "
          "cap typos, not phenomena\n")

    # STEP 6 — the ladder, then the leak-free pipeline
    print("STEP 6 — the ladder: features first, fancy models second:")
    created = onehot.assign(
        condition_rank=df["condition_rank"],
        years_since_sold=df["years_since_sold"],
        sold_was_missing=df["sold_was_missing"],
        sqft_was_missing=df["sqft_was_missing"],
        sqft_per_bedroom=df["sqft_per_bedroom"],
        sqft_vs_city=df["sqft_vs_city"])
    ladder = [("raw numerics only", ridge_r2(base, target)),
              ("+ encoded categoricals", ridge_r2(
                  onehot.assign(condition_rank=df["condition_rank"]), target)),
              ("+ created features", ridge_r2(created, target))]
    forest_raw = float(cross_val_score(RandomForestRegressor(200, random_state=42),
                                       base, target, cv=5, scoring="r2").mean())
    for name, score in ladder:
        print(f"    Ridge, {name:<24} R² = {score:.3f}")
    print(f"    RandomForest on raw numerics  R² = {forest_raw:.3f}   "
          "← fancy model, garbage inputs")
    assert ladder[2][1] > ladder[0][1]
    print("    → the doc's claim, measured: a linear model with great features "
          "beats a forest with bad ones\n")

    print("    ...and the production version — every step above, leak-free:")
    raw = pd.read_csv(CSV_PATH, parse_dates=["last_sold"])
    raw["years_since_sold"] = (TODAY - raw["last_sold"]).dt.days / 365.25
    raw["price"] = raw["price"].clip(upper=raw["price"].quantile(0.99))
    X = raw.drop(columns=["price", "last_sold"])
    y = raw["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    numeric = ["sqft", "bedrooms", "built_year", "years_since_sold"]
    categorical = ["city", "condition"]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]),
         categorical)])
    pipe = Pipeline([("prep", preprocessor),
                     ("model", RandomForestRegressor(random_state=42))])
    search = GridSearchCV(pipe, {"model__max_depth": [8, None],
                                 "model__n_estimators": [100, 300]},
                          cv=3, scoring="r2")
    search.fit(X_train, y_train)
    print(f"    GridSearchCV best: {search.best_params_} "
          f"(CV R² {search.best_score_:.3f})")
    print(f"    test R² (one look): {search.score(X_test, y_test):.3f}")
    print("    → imputers/scalers/encoders re-fit inside every CV fold: "
          "leakage impossible by construction\n")

    # Plot: the ladder (left), what the final model valued (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    names = [n for n, _ in ladder] + ["RandomForest,\nraw numerics"]
    scores = [s for _, s in ladder] + [forest_raw]
    colors = ["steelblue"] * 3 + ["darkorange"]
    ax1.barh(names[::-1], scores[::-1], color=colors[::-1])
    ax1.set_xlabel("5-fold CV R²")
    ax1.set_title("Features first: each engineering step buys more than\n"
                  "switching to a fancier model does")

    final_model = search.best_estimator_
    feature_names = [n.split("__")[1] for n in
                     final_model["prep"].get_feature_names_out()]
    importances = final_model["model"].feature_importances_
    order = np.argsort(importances)
    ax2.barh(np.array(feature_names)[order], importances[order], color="seagreen")
    ax2.set_xlabel("RandomForest importance")
    ax2.set_title("What the final pipeline actually used")

    fig.tight_layout()
    out = "feature-engineering/features_plot.png"
    fig.savefig(out, dpi=120)
    print(f"    plot saved → {out}")


if __name__ == "__main__":
    main()
