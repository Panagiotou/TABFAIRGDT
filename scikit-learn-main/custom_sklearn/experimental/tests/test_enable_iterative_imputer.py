"""Tests for making sure experimental imports work as expected."""

import textwrap

import pytest

from custom_sklearn.utils._testing import assert_run_python_script_without_output
from custom_sklearn.utils.fixes import _IS_WASM


@pytest.mark.xfail(_IS_WASM, reason="cannot start subprocess")
def test_imports_strategies():
    # Make sure different import strategies work or fail as expected.

    # Since Python caches the imported modules, we need to run a child process
    # for every test case. Else, the tests would not be independent
    # (manually removing the imports from the cache (sys.modules) is not
    # recommended and can lead to many complications).
    pattern = "IterativeImputer is experimental"
    good_import = """
    from custom_sklearn.experimental import enable_iterative_imputer
    from custom_sklearn.impute import IterativeImputer
    """
    assert_run_python_script_without_output(
        textwrap.dedent(good_import), pattern=pattern
    )

    good_import_with_ensemble_first = """
    import custom_sklearn.ensemble
    from custom_sklearn.experimental import enable_iterative_imputer
    from custom_sklearn.impute import IterativeImputer
    """
    assert_run_python_script_without_output(
        textwrap.dedent(good_import_with_ensemble_first),
        pattern=pattern,
    )

    bad_imports = f"""
    import pytest

    with pytest.raises(ImportError, match={pattern!r}):
        from custom_sklearn.impute import IterativeImputer

    import custom_sklearn.experimental
    with pytest.raises(ImportError, match={pattern!r}):
        from custom_sklearn.impute import IterativeImputer
    """
    assert_run_python_script_without_output(
        textwrap.dedent(bad_imports),
        pattern=pattern,
    )
