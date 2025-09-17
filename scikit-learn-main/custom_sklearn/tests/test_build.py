import os
import textwrap

import pytest

from custom_sklearn import __version__
from custom_sklearn.utils._openmp_helpers import _openmp_parallelism_enabled


def test_openmp_parallelism_enabled():
    # Check that custom_sklearn is built with OpenMP-based parallelism enabled.
    # This test can be skipped by setting the environment variable
    # ``custom_sklearn_SKIP_OPENMP_TEST``.
    if os.getenv("custom_sklearn_SKIP_OPENMP_TEST"):
        pytest.skip("test explicitly skipped (custom_sklearn_SKIP_OPENMP_TEST)")

    base_url = "dev" if __version__.endswith(".dev0") else "stable"
    err_msg = textwrap.dedent(
        """
        This test fails because custom-scikit-learn has been built without OpenMP.
        This is not recommended since some estimators will run in sequential
        mode instead of leveraging thread-based parallelism.

        You can find instructions to build custom-scikit-learn with OpenMP at this
        address:

            https://custom-scikit-learn.org/{}/developers/advanced_installation.html

        You can skip this test by setting the environment variable
        custom_sklearn_SKIP_OPENMP_TEST to any value.
        """
    ).format(base_url)

    assert _openmp_parallelism_enabled(), err_msg
