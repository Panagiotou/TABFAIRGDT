"""
Example usage of the tabfairgdt package.

TabFairGDT generates synthetic tabular data using autoregressive decision trees
while enforcing fairness constraints on a target variable with respect to a
protected (sensitive) attribute.

The key idea:
  - Each column is modelled by a CART decision tree, conditioned on all
    previously generated columns (autoregressive).
  - The target column uses a special fair CART that adjusts leaf probability
    distributions to reduce demographic disparity.
  - The `lamda` parameter controls the fairness–utility tradeoff:
      lamda=0  →  standard CART, no fairness constraint
      lamda=1  →  maximum fairness, predictions near 50/50 at biased leaves
      lamda=0.5  →  balanced tradeoff (recommended starting point)
"""

import numpy as np
import pandas as pd
from tabfairgdt import TABFAIRGDT

# -----------------------------------------------------------------------------
# 1. Create or load a dataset
# -----------------------------------------------------------------------------
# We build a small synthetic dataset that mimics a binary income-prediction
# problem where "sex" is the protected attribute and "income" is the target.
#
# In a real use case you would load your own data, e.g.:
#   df = pd.read_csv("adult.csv")

rng = np.random.default_rng(0)
n = 500

age          = rng.integers(18, 65, size=n)
hours        = rng.integers(20, 60, size=n)
capital_gain = rng.uniform(0, 15000, size=n).round(2)
sex          = rng.choice(["male", "female"], size=n)
education    = rng.choice(["low", "medium", "high"], size=n)

# Target is correlated with age, hours, and education — and slightly biased
# toward "male" so the fairness constraint has something to fix.
edu_score = pd.Series(education).map({"low": 0, "medium": 1, "high": 2}).values
sex_bias  = np.where(sex == "male", 0.3, 0.0)
score     = age / 65 + hours / 60 + edu_score / 2 + sex_bias + rng.normal(0, 0.4, n)
income    = np.where(score > 1.5, ">50K", "<=50K")

df = pd.DataFrame({
    "age":          age.astype("int64"),
    "hours":        hours.astype("int64"),
    "capital_gain": capital_gain,
    "sex":          pd.Categorical(sex),
    "education":    pd.Categorical(education),
    "income":       pd.Categorical(income),
})

print("=== Original dataset ===")
print(df.head(8))
print(f"\nShape: {df.shape}")
print(f"Income distribution:\n{df['income'].value_counts(normalize=True).round(3)}")
print(f"\nIncome by sex (demographic parity check):")
print(df.groupby("sex")["income"].value_counts(normalize=True).unstack().round(3))

# -----------------------------------------------------------------------------
# 2. Define the dtype map
# -----------------------------------------------------------------------------
# TabFairGDT needs to know the type of every column so it can choose the right
# CART variant and apply the correct pre/post-processing.
#
# Supported type strings:
#   "int"      — integer numeric column
#   "float"    — floating-point numeric column
#   "category" — categorical (nominal) column, including the target and
#                protected attribute
#   "bool"     — boolean column
#   "datetime" — datetime column

dtype_map = {
    "age":          "int",
    "hours":        "int",
    "capital_gain": "float",
    "sex":          "category",
    "education":    "category",
    "income":       "category",   # target column
}

# -----------------------------------------------------------------------------
# 3. Initialise the generator
# -----------------------------------------------------------------------------
# Key parameters:
#   protected_attribute — column that defines the sensitive groups
#   target              — binary column whose distribution should be fair
#   criterion           — fairness criterion; "dp" = demographic parity
#   acc_threshold       — max allowed relative accuracy drop (0.05 = 5%);
#                         -1 means no limit
#   seed                — random seed for reproducibility
#   parallel            — fit column models in parallel (faster on large data)

generator = TABFAIRGDT(
    protected_attribute="sex",
    target="income",
    dtype_map=dtype_map,
    criterion="dp",
    acc_threshold=0.05,   # allow up to 5% accuracy drop for fairness
    seed=42,
    parallel=False,       # set True to speed up fitting on larger datasets
    verbose=False,
)

# -----------------------------------------------------------------------------
# 4. Fit the generator on the training data
# -----------------------------------------------------------------------------
# `lamda` is passed at fit time so you can reuse the same generator object
# with different fairness strengths without re-initialising it.
#
# lamda=0.0  →  purely data-driven, no fairness adjustment
# lamda=0.5  →  balanced tradeoff
# lamda=1.0  →  maximum fairness (may sacrifice accuracy)

print("\n=== Fitting the generator (lamda=0.5) ===")
generator.fit(df, lamda=0.5)
print("Fitting complete.")

# -----------------------------------------------------------------------------
# 5. Generate synthetic data
# -----------------------------------------------------------------------------
# `k` is the number of synthetic rows to generate.
# If omitted (k=None), it defaults to the size of the training dataset.

print("\n=== Generating 500 synthetic rows ===")
synthetic_df = generator.generate(k=500)

print(synthetic_df.head(8))
print(f"\nShape: {synthetic_df.shape}")
print(f"Columns: {list(synthetic_df.columns)}")

# -----------------------------------------------------------------------------
# 6. Inspect the synthetic data quality
# -----------------------------------------------------------------------------
print("\n=== Quality check ===")

# Column dtypes should match the original
print("Dtypes match?", dict(synthetic_df.dtypes) == dict(df.dtypes))

# All generated values must belong to the original domain
for col in ["sex", "education", "income"]:
    original_values = set(df[col].cat.categories)
    synthetic_values = set(synthetic_df[col].dropna().unique())
    print(f"  '{col}' values in domain: {synthetic_values.issubset(original_values)}")

# Marginal distributions should be similar
print("\nOriginal income distribution:")
print(df["income"].value_counts(normalize=True).round(3))
print("Synthetic income distribution:")
print(synthetic_df["income"].value_counts(normalize=True).round(3))

# -----------------------------------------------------------------------------
# 7. Fairness evaluation — demographic parity gap
# -----------------------------------------------------------------------------
# Demographic parity: the positive prediction rate should be equal across
# protected groups.  A gap close to 0 means fair synthetic data.

def dp_gap(data, target_col, protected_col, positive_label=">50K"):
    """Absolute difference in positive rates between the two protected groups."""
    groups = data[protected_col].unique()
    rates = {
        g: (data.loc[data[protected_col] == g, target_col] == positive_label).mean()
        for g in groups
    }
    vals = list(rates.values())
    return abs(vals[0] - vals[1]), rates

gap_orig, rates_orig = dp_gap(df, "income", "sex")
gap_synth, rates_synth = dp_gap(synthetic_df, "income", "sex")

print("\n=== Fairness (Demographic Parity) ===")
print(f"Original data:   rates={rates_orig}, gap={gap_orig:.3f}")
print(f"Synthetic (λ=0.5): rates={rates_synth}, gap={gap_synth:.3f}")

# -----------------------------------------------------------------------------
# 8. Compare across different lambda values
# -----------------------------------------------------------------------------
# Demonstrate how lamda controls the fairness–utility tradeoff.

print("\n=== Effect of lamda on demographic parity gap ===")
print(f"{'lamda':>8}  {'DP gap':>8}  {'>50K rate (male)':>20}  {'>50K rate (female)':>20}")

for lamda in [0.0, 0.25, 0.5, 0.75, 1.0]:
    gen = TABFAIRGDT(
        protected_attribute="sex",
        target="income",
        dtype_map=dtype_map,
        seed=42,
        parallel=False,
        verbose=False,
    )
    gen.fit(df, lamda=lamda)
    synth = gen.generate(k=500)
    gap, rates = dp_gap(synth, "income", "sex")
    print(f"  {lamda:>6.2f}  {gap:>8.3f}  {rates.get('male', 0):>20.3f}  {rates.get('female', 0):>20.3f}")

# -----------------------------------------------------------------------------
# 9. Save the synthetic dataset
# -----------------------------------------------------------------------------
output_path = "synthetic_data.csv"
synthetic_df.to_csv(output_path, index=False)
print(f"\nSynthetic dataset saved to '{output_path}'.")
