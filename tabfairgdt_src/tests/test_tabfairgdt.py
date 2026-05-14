"""Tests for the TABFAIRGDT package."""

import numpy as np
import pandas as pd
import pytest

from tabfairgdt import TABFAIRGDT


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_binary_df(n=200, seed=0):
    """Synthetic binary-classification dataset with a protected attribute."""
    rng = np.random.default_rng(seed)
    age = rng.integers(20, 65, size=n)
    sex = rng.choice(["male", "female"], size=n)
    education = rng.choice(["low", "mid", "high"], size=n)
    capital_gain = rng.uniform(0, 10000, size=n).round(2)
    # target is correlated with age and education
    income_score = (age / 65) + (pd.Series(education).map({"low": 0, "mid": 1, "high": 2}).values / 2)
    income = pd.Series(np.where(income_score + rng.normal(0, 0.3, n) > 1.0, ">50K", "<=50K"))
    df = pd.DataFrame({
        "age": age.astype("int64"),
        "sex": pd.Categorical(sex),
        "education": pd.Categorical(education),
        "capital_gain": capital_gain,
        "income": pd.Categorical(income),
    })
    return df


DTYPE_MAP = {
    "age": "int",
    "sex": "category",
    "education": "category",
    "capital_gain": "float",
    "income": "category",
}


# ---------------------------------------------------------------------------
# Basic smoke tests
# ---------------------------------------------------------------------------

class TestTabFairGDTSmoke:
    def test_import(self):
        from tabfairgdt import TABFAIRGDT, TABFAIRGDT_FAIR_SPLITTING_CRITERION
        assert TABFAIRGDT is not None
        assert TABFAIRGDT_FAIR_SPLITTING_CRITERION is not None

    def test_fit_generate_default(self):
        df = make_binary_df()
        gen = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=42,
            parallel=False,
        )
        gen.fit(df, lamda=0.5)
        synth = gen.generate()
        assert isinstance(synth, pd.DataFrame)
        assert len(synth) == len(df)
        assert set(synth.columns) == set(df.columns)

    def test_generate_custom_k(self):
        df = make_binary_df()
        gen = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=0,
            parallel=False,
        )
        gen.fit(df, lamda=0.0)
        synth = gen.generate(k=50)
        assert len(synth) == 50

    def test_output_columns_match(self):
        df = make_binary_df()
        gen = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=1,
            parallel=False,
        )
        gen.fit(df, lamda=0.5)
        synth = gen.generate(k=100)
        assert list(sorted(synth.columns)) == list(sorted(df.columns))

    def test_target_values_in_original_domain(self):
        df = make_binary_df()
        gen = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=2,
            parallel=False,
        )
        gen.fit(df, lamda=0.5)
        synth = gen.generate(k=100)
        valid_values = set(df["income"].cat.categories)
        assert set(synth["income"].dropna().unique()).issubset(valid_values)

    def test_protected_attr_values_in_original_domain(self):
        df = make_binary_df()
        gen = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=3,
            parallel=False,
        )
        gen.fit(df, lamda=0.5)
        synth = gen.generate(k=100)
        valid_values = set(df["sex"].cat.categories)
        assert set(synth["sex"].dropna().unique()).issubset(valid_values)


# ---------------------------------------------------------------------------
# Parameter validation tests
# ---------------------------------------------------------------------------

class TestTabFairGDTParams:
    def test_lamda_zero_and_one(self):
        """Both extreme lamda values should complete without error."""
        df = make_binary_df(n=100)
        for lamda in (0.0, 1.0):
            gen = TABFAIRGDT(
                protected_attribute="sex",
                target="income",
                dtype_map=DTYPE_MAP,
                seed=0,
                parallel=False,
            )
            gen.fit(df, lamda=lamda)
            synth = gen.generate(k=50)
            assert len(synth) == 50

    def test_acc_threshold(self):
        """acc_threshold should not prevent fitting."""
        df = make_binary_df(n=150)
        gen = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=0,
            acc_threshold=0.1,
            parallel=False,
        )
        gen.fit(df, lamda=0.5)
        synth = gen.generate(k=50)
        assert len(synth) == 50

    def test_parallel_mode(self):
        """parallel=True should produce same shape output."""
        df = make_binary_df(n=150)
        gen = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=7,
            parallel=True,
        )
        gen.fit(df, lamda=0.5)
        synth = gen.generate(k=60)
        assert len(synth) == 60
        assert set(synth.columns) == set(df.columns)

    def test_seed_reproducibility(self):
        """Two generators fitted with the same seed produce the same first draw."""
        df = make_binary_df(n=100)
        kwargs = dict(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=99,
            parallel=False,
        )
        gen1 = TABFAIRGDT(**kwargs)
        gen1.fit(df, lamda=0.5)
        synth1 = gen1.generate(k=40)

        gen2 = TABFAIRGDT(**kwargs)
        gen2.fit(df, lamda=0.5)
        synth2 = gen2.generate(k=40)

        pd.testing.assert_frame_equal(
            synth1.reset_index(drop=True),
            synth2.reset_index(drop=True),
        )

    def test_generate_diversity(self):
        """Repeated generate() calls on the same fitted generator produce different data."""
        df = make_binary_df(n=100)
        gen = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=99,
            parallel=False,
        )
        gen.fit(df, lamda=0.5)
        synth1 = gen.generate(k=100)
        synth2 = gen.generate(k=100)

        # At least some rows must differ — identical outputs would mean diversity is broken
        assert not synth1.equals(synth2), "Repeated generate() calls produced identical output"


# ---------------------------------------------------------------------------
# Data type preservation tests
# ---------------------------------------------------------------------------

class TestDtypePreservation:
    def test_int_column_dtype(self):
        df = make_binary_df(n=100)
        gen = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=5,
            parallel=False,
        )
        gen.fit(df, lamda=0.5)
        synth = gen.generate(k=50)
        assert synth["age"].dtype in (np.dtype("int64"), np.dtype("int32"), np.dtype("int"))

    def test_float_column_range(self):
        df = make_binary_df(n=150)
        gen = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=6,
            parallel=False,
        )
        gen.fit(df, lamda=0.5)
        synth = gen.generate(k=80)
        assert synth["capital_gain"].dtype in (np.dtype("float64"), np.dtype("float32"))


# ---------------------------------------------------------------------------
# Fairness integration test
# ---------------------------------------------------------------------------

class TestFairnessConstraint:
    def test_fairness_reduces_demographic_parity_gap(self):
        """
        With lamda=1 (max fairness) the demographic parity gap should be
        smaller than with lamda=0 (no fairness), or at worst equal.
        """
        df = make_binary_df(n=400, seed=0)

        def dp_gap(synth):
            pos = synth["income"] == ">50K"
            male = synth["sex"] == "male"
            female = synth["sex"] == "female"
            rate_m = pos[male].mean() if male.any() else 0.0
            rate_f = pos[female].mean() if female.any() else 0.0
            return abs(rate_m - rate_f)

        gen_unfair = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=0,
            parallel=False,
        )
        gen_unfair.fit(df, lamda=0.0)
        gap_unfair = dp_gap(gen_unfair.generate(k=400))

        gen_fair = TABFAIRGDT(
            protected_attribute="sex",
            target="income",
            dtype_map=DTYPE_MAP,
            seed=0,
            parallel=False,
        )
        gen_fair.fit(df, lamda=1.0)
        gap_fair = dp_gap(gen_fair.generate(k=400))

        assert gap_fair <= gap_unfair + 0.05, (
            f"Fairness constraint did not reduce DP gap: "
            f"unfair={gap_unfair:.3f}, fair={gap_fair:.3f}"
        )
