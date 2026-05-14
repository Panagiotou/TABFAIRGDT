# TabFairGDT

**TabFairGDT** — Fast Fair Tabular Data Generator using Autoregressive Decision Trees.

Generates synthetic tabular data with built-in fairness constraints via leaf-relabeling of CART models.

## Installation

```bash
pip install tabfairgdt
```

## Quick Start

```python
import pandas as pd
from tabfairgdt import TABFAIRGDT

df = pd.read_csv("data.csv")
df["income"] = df["income"].astype("category")
df["sex"] = df["sex"].astype("category")

dtype_map = {col: str(df[col].dtype) for col in df.columns}
# Map pandas dtypes to tabfairgdt types
dtype_map = {
    col: "category" if df[col].dtype.name in ("category", "object", "bool")
    else "float" if df[col].dtype.kind == "f"
    else "int"
    for col in df.columns
}

generator = TABFAIRGDT(
    protected_attribute="sex",
    target="income",
    criterion="dp",          # demographic parity
    dtype_map=dtype_map,
    seed=42,
)
generator.fit(df, lamda=0.5)   # lamda=0: no fairness, lamda=1: max fairness
synthetic_df = generator.generate(k=1000)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `protected_attribute` | required | Column name of the sensitive/protected attribute |
| `target` | required | Column name of the target variable |
| `dtype_map` | required | Dict mapping column names to dtypes (`'int'`, `'float'`, `'category'`, `'bool'`, `'datetime'`) |
| `criterion` | `"dp"` | Fairness criterion: `"dp"` (demographic parity) |
| `lamda` | `0.5` | Fairness–utility tradeoff passed to `fit()`: `0` = no fairness, `1` = max fairness |
| `acc_threshold` | `-1` | Max allowed accuracy drop as a fraction (e.g., `0.05` = 5%); `-1` = no limit |
| `seed` | `None` | Random seed for reproducibility |
| `parallel` | `True` | Use parallel fitting across columns |
| `re_order` | `False` | Reorder features by correlation (`"corr_asc_target"`, `"corr_desc_target"`, etc.) |
| `smoothing` | `False` | Apply kernel smoothing to continuous columns (`"density"` or column dict) |
| `proper` | `False` | Bootstrap training data during fitting |

## Citation

```bibtex
@inproceedings{panagiotou2025tabfairgdt,
  title     = {TABFAIRGDT: A Fast Fair Tabular Data Generator using Autoregressive Decision Trees},
  author    = {Panagiotou, Emmanouil and Ronval, Benoît and Roy, Arjun and Bothmann, Ludwig and Bischl, Bernd and Nijssen, Siegfried and Ntoutsi, Eirini},
  booktitle = {IEEE ICDM 2025},
  year      = {2025},
  url       = {https://arxiv.org/abs/2509.19927}
}
```
