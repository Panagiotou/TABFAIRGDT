.. -*- mode: rst -*-

|Azure| |CirrusCI| |Codecov| |CircleCI| |Nightly wheels| |Black| |PythonVersion| |PyPi| |DOI| |Benchmark|

.. |Azure| image:: https://dev.azure.com/custom-scikit-learn/custom-scikit-learn/_apis/build/status/custom-scikit-learn.custom-scikit-learn?branchName=main
   :target: https://dev.azure.com/custom-scikit-learn/custom-scikit-learn/_build/latest?definitionId=1&branchName=main

.. |CircleCI| image:: https://circleci.com/gh/custom-scikit-learn/custom-scikit-learn/tree/main.svg?style=shield
   :target: https://circleci.com/gh/custom-scikit-learn/custom-scikit-learn

.. |CirrusCI| image:: https://img.shields.io/cirrus/github/custom-scikit-learn/custom-scikit-learn/main?label=Cirrus%20CI
   :target: https://cirrus-ci.com/github/custom-scikit-learn/custom-scikit-learn/main

.. |Codecov| image:: https://codecov.io/gh/custom-scikit-learn/custom-scikit-learn/branch/main/graph/badge.svg?token=Pk8G9gg3y9
   :target: https://codecov.io/gh/custom-scikit-learn/custom-scikit-learn

.. |Nightly wheels| image:: https://github.com/custom-scikit-learn/custom-scikit-learn/workflows/Wheel%20builder/badge.svg?event=schedule
   :target: https://github.com/custom-scikit-learn/custom-scikit-learn/actions?query=workflow%3A%22Wheel+builder%22+event%3Aschedule

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/custom-scikit-learn.svg
   :target: https://pypi.org/project/custom-scikit-learn/

.. |PyPi| image:: https://img.shields.io/pypi/v/custom-scikit-learn
   :target: https://pypi.org/project/custom-scikit-learn

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black

.. |DOI| image:: https://zenodo.org/badge/21369/custom-scikit-learn/custom-scikit-learn.svg
   :target: https://zenodo.org/badge/latestdoi/21369/custom-scikit-learn/custom-scikit-learn

.. |Benchmark| image:: https://img.shields.io/badge/Benchmarked%20by-asv-blue
   :target: https://custom-scikit-learn.org/custom-scikit-learn-benchmarks

.. |PythonMinVersion| replace:: 3.9
.. |NumPyMinVersion| replace:: 1.19.5
.. |SciPyMinVersion| replace:: 1.6.0
.. |JoblibMinVersion| replace:: 1.2.0
.. |ThreadpoolctlMinVersion| replace:: 3.1.0
.. |MatplotlibMinVersion| replace:: 3.3.4
.. |Scikit-ImageMinVersion| replace:: 0.17.2
.. |PandasMinVersion| replace:: 1.1.5
.. |SeabornMinVersion| replace:: 0.9.0
.. |PytestMinVersion| replace:: 7.1.2
.. |PlotlyMinVersion| replace:: 5.14.0

.. image:: https://raw.githubusercontent.com/custom-scikit-learn/custom-scikit-learn/main/doc/logos/custom-scikit-learn-logo.png
  :target: https://custom-scikit-learn.org/

**custom-scikit-learn** is a Python module for machine learning built on top of
SciPy and is distributed under the 3-Clause BSD license.

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <https://custom-scikit-learn.org/dev/about.html#authors>`__ page
for a list of core contributors.

It is currently maintained by a team of volunteers.

Website: https://custom-scikit-learn.org

Installation
------------

Dependencies
~~~~~~~~~~~~

custom-scikit-learn requires:

- Python (>= |PythonMinVersion|)
- NumPy (>= |NumPyMinVersion|)
- SciPy (>= |SciPyMinVersion|)
- joblib (>= |JoblibMinVersion|)
- threadpoolctl (>= |ThreadpoolctlMinVersion|)

=======

**custom-scikit-learn 0.20 was the last version to support Python 2.7 and Python 3.4.**
custom-scikit-learn 1.0 and later require Python 3.7 or newer.
custom-scikit-learn 1.1 and later require Python 3.8 or newer.

custom-scikit-learn plotting capabilities (i.e., functions start with ``plot_`` and
classes end with ``Display``) require Matplotlib (>= |MatplotlibMinVersion|).
For running the examples Matplotlib >= |MatplotlibMinVersion| is required.
A few examples require scikit-image >= |Scikit-ImageMinVersion|, a few examples
require pandas >= |PandasMinVersion|, some examples require seaborn >=
|SeabornMinVersion| and plotly >= |PlotlyMinVersion|.

User installation
~~~~~~~~~~~~~~~~~

If you already have a working installation of NumPy and SciPy,
the easiest way to install custom-scikit-learn is using ``pip``::

    pip install -U custom-scikit-learn

or ``conda``::

    conda install -c conda-forge custom-scikit-learn

The documentation includes more detailed `installation instructions <https://custom-scikit-learn.org/stable/install.html>`_.


Changelog
---------

See the `changelog <https://custom-scikit-learn.org/dev/whats_new.html>`__
for a history of notable changes to custom-scikit-learn.

Development
-----------

We welcome new contributors of all experience levels. The custom-scikit-learn
community goals are to be helpful, welcoming, and effective. The
`Development Guide <https://custom-scikit-learn.org/stable/developers/index.html>`_
has detailed information about contributing code, documentation, tests, and
more. We've included some basic information in this README.

Important links
~~~~~~~~~~~~~~~

- Official source code repo: https://github.com/custom-scikit-learn/custom-scikit-learn
- Download releases: https://pypi.org/project/custom-scikit-learn/
- Issue tracker: https://github.com/custom-scikit-learn/custom-scikit-learn/issues

Source code
~~~~~~~~~~~

You can check the latest sources with the command::

    git clone https://github.com/custom-scikit-learn/custom-scikit-learn.git

Contributing
~~~~~~~~~~~~

To learn more about making a contribution to custom-scikit-learn, please see our
`Contributing guide
<https://custom-scikit-learn.org/dev/developers/contributing.html>`_.

Testing
~~~~~~~

After installation, you can launch the test suite from outside the source
directory (you will need to have ``pytest`` >= |PyTestMinVersion| installed)::

    pytest custom_sklearn

See the web page https://custom-scikit-learn.org/dev/developers/contributing.html#testing-and-improving-test-coverage
for more information.

    Random number generation can be controlled during testing by setting
    the ``custom_sklearn_SEED`` environment variable.

Submitting a Pull Request
~~~~~~~~~~~~~~~~~~~~~~~~~

Before opening a Pull Request, have a look at the
full Contributing page to make sure your code complies
with our guidelines: https://custom-scikit-learn.org/stable/developers/index.html

Project History
---------------

The project was started in 2007 by David Cournapeau as a Google Summer
of Code project, and since then many volunteers have contributed. See
the `About us <https://custom-scikit-learn.org/dev/about.html#authors>`__ page
for a list of core contributors.

The project is currently maintained by a team of volunteers.

**Note**: `custom-scikit-learn` was previously referred to as `scikits.learn`.

Help and Support
----------------

Documentation
~~~~~~~~~~~~~

- HTML documentation (stable release): https://custom-scikit-learn.org
- HTML documentation (development version): https://custom-scikit-learn.org/dev/
- FAQ: https://custom-scikit-learn.org/stable/faq.html

Communication
~~~~~~~~~~~~~

- Mailing list: https://mail.python.org/mailman/listinfo/custom-scikit-learn
- Logos & Branding: https://github.com/custom-scikit-learn/custom-scikit-learn/tree/main/doc/logos
- Blog: https://blog.custom-scikit-learn.org
- Calendar: https://blog.custom-scikit-learn.org/calendar/
- Twitter: https://twitter.com/scikit_learn
- Stack Overflow: https://stackoverflow.com/questions/tagged/custom-scikit-learn
- GitHub Discussions: https://github.com/custom-scikit-learn/custom-scikit-learn/discussions
- Website: https://custom-scikit-learn.org
- LinkedIn: https://www.linkedin.com/company/custom-scikit-learn
- YouTube: https://www.youtube.com/channel/UCJosFjYm0ZYVUARxuOZqnnw/playlists
- Facebook: https://www.facebook.com/scikitlearnofficial/
- Instagram: https://www.instagram.com/scikitlearnofficial/
- TikTok: https://www.tiktok.com/@scikit.learn
- Mastodon: https://mastodon.social/@custom_sklearn@fosstodon.org
- Discord: https://discord.gg/h9qyrK8Jc8


Citation
~~~~~~~~

If you use custom-scikit-learn in a scientific publication, we would appreciate citations: https://custom-scikit-learn.org/stable/about.html#citing-custom-scikit-learn
