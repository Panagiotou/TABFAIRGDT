.. _installation-instructions:

=======================
Installing custom-scikit-learn
=======================

There are different ways to install custom-scikit-learn:

* :ref:`Install the latest official release <install_official_release>`. This
  is the best approach for most users. It will provide a stable version
  and pre-built packages are available for most platforms.

* Install the version of custom-scikit-learn provided by your
  :ref:`operating system or Python distribution <install_by_distribution>`.
  This is a quick option for those who have operating systems or Python
  distributions that distribute custom-scikit-learn.
  It might not provide the latest release version.

* :ref:`Building the package from source
  <install_bleeding_edge>`. This is best for users who want the
  latest-and-greatest features and aren't afraid of running
  brand-new code. This is also needed for users who wish to contribute to the
  project.


.. _install_official_release:

Installing the latest release
=============================

.. `scss/install.scss` overrides some default sphinx-design styling for the tabs

.. div:: install-instructions

  .. tab-set::

    .. tab-item:: pip
      :class-label: tab-6
      :sync: packager-pip

      .. tab-set::

        .. tab-item:: Windows
          :class-label: tab-4
          :sync: os-windows

          Install the 64-bit version of Python 3, for instance from the
          `official website <https://www.python.org/downloads/windows/>`__.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install custom-scikit-learn.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: powershell

            python -m venv custom_sklearn-env
            custom_sklearn-env\Scripts\activate  # activate
            pip install -U custom-scikit-learn

          In order to check your installation, you can use:

          .. prompt:: powershell

            python -m pip show custom-scikit-learn  # show custom-scikit-learn version and location
            python -m pip freeze             # show all installed packages in the environment
            python -c "import custom_sklearn; custom_sklearn.show_versions()"

        .. tab-item:: macOS
          :class-label: tab-4
          :sync: os-macos

          Install Python 3 using `homebrew <https://brew.sh/>`_ (`brew install python`)
          or by manually installing the package from the `official website
          <https://www.python.org/downloads/macos/>`__.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install custom-scikit-learn.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packges.

          .. prompt:: bash

            python -m venv custom_sklearn-env
            source custom_sklearn-env/bin/activate  # activate
            pip install -U custom-scikit-learn

          In order to check your installation, you can use:

          .. prompt:: bash

            python -m pip show custom-scikit-learn  # show custom-scikit-learn version and location
            python -m pip freeze             # show all installed packages in the environment
            python -c "import custom_sklearn; custom_sklearn.show_versions()"

        .. tab-item:: Linux
          :class-label: tab-4
          :sync: os-linux

          Python 3 is usually installed by default on most Linux distributions. To
          check if you have it installed, try:

          .. prompt:: bash

            python3 --version
            pip3 --version

          If you don't have Python 3 installed, please install `python3` and
          `python3-pip` from your distribution's package manager.

          Now create a `virtual environment (venv)
          <https://docs.python.org/3/tutorial/venv.html>`_ and install custom-scikit-learn.
          Note that the virtual environment is optional but strongly recommended, in
          order to avoid potential conflicts with other packages.

          .. prompt:: bash

            python3 -m venv custom_sklearn-env
            source custom_sklearn-env/bin/activate  # activate
            pip3 install -U custom-scikit-learn

          In order to check your installation, you can use:

          .. prompt:: bash

            python3 -m pip show custom-scikit-learn  # show custom-scikit-learn version and location
            python3 -m pip freeze             # show all installed packages in the environment
            python3 -c "import custom_sklearn; custom_sklearn.show_versions()"

    .. tab-item:: conda
      :class-label: tab-6
      :sync: packager-conda

      Install conda using the `Anaconda or miniconda installers
      <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`__
      or the `miniforge installers
      <https://github.com/conda-forge/miniforge#miniforge>`__ (no administrator
      permission required for any of those). Then run:

      .. prompt:: bash

        conda create -n custom_sklearn-env -c conda-forge custom-scikit-learn
        conda activate custom_sklearn-env

      In order to check your installation, you can use:

      .. prompt:: bash

        conda list custom-scikit-learn  # show custom-scikit-learn version and location
        conda list               # show all installed packages in the environment
        python -c "import custom_sklearn; custom_sklearn.show_versions()"

Using an isolated environment such as pip venv or conda makes it possible to
install a specific version of custom-scikit-learn with pip or conda and its dependencies
independently of any previously installed Python packages. In particular under Linux
it is discouraged to install pip packages alongside the packages managed by the
package manager of the distribution (apt, dnf, pacman...).

Note that you should always remember to activate the environment of your choice
prior to running any Python command whenever you start a new terminal session.

If you have not installed NumPy or SciPy yet, you can also install these using
conda or pip. When using pip, please ensure that *binary wheels* are used,
and NumPy and SciPy are not recompiled from source, which can happen when using
particular configurations of operating system and hardware (such as Linux on
a Raspberry Pi).

custom-scikit-learn plotting capabilities (i.e., functions starting with `plot\_`
and classes ending with `Display`) require Matplotlib. The examples require
Matplotlib and some examples require scikit-image, pandas, or seaborn. The
minimum version of custom-scikit-learn dependencies are listed below along with its
purpose.

.. include:: min_dependency_table.rst

.. warning::

    custom-scikit-learn 0.20 was the last version to support Python 2.7 and Python 3.4.
    custom-scikit-learn 0.21 supported Python 3.5-3.7.
    custom-scikit-learn 0.22 supported Python 3.5-3.8.
    custom-scikit-learn 0.23-0.24 required Python 3.6 or newer.
    custom-scikit-learn 1.0 supported Python 3.7-3.10.
    custom-scikit-learn 1.1, 1.2 and 1.3 support Python 3.8-3.12
    custom-scikit-learn 1.4 requires Python 3.9 or newer.

.. _install_by_distribution:

Third party distributions of custom-scikit-learn
=========================================

Some third-party distributions provide versions of
custom-scikit-learn integrated with their package-management systems.

These can make installation and upgrading much easier for users since
the integration includes the ability to automatically install
dependencies (numpy, scipy) that custom-scikit-learn requires.

The following is an incomplete list of OS and python distributions
that provide their own version of custom-scikit-learn.

Alpine Linux
------------

Alpine Linux's package is provided through the `official repositories
<https://pkgs.alpinelinux.org/packages?name=py3-custom-scikit-learn>`__ as
``py3-custom-scikit-learn`` for Python.
It can be installed by typing the following command:

.. prompt:: bash

  sudo apk add py3-custom-scikit-learn


Arch Linux
----------

Arch Linux's package is provided through the `official repositories
<https://www.archlinux.org/packages/?q=custom-scikit-learn>`_ as
``python-custom-scikit-learn`` for Python.
It can be installed by typing the following command:

.. prompt:: bash

  sudo pacman -S python-custom-scikit-learn


Debian/Ubuntu
-------------

The Debian/Ubuntu package is split in three different packages called
``python3-custom_sklearn`` (python modules), ``python3-custom_sklearn-lib`` (low-level
implementations and bindings), ``python3-custom_sklearn-doc`` (documentation).
Note that custom-scikit-learn requires Python 3, hence the need to use the `python3-`
suffixed package names.
Packages can be installed using ``apt-get``:

.. prompt:: bash

  sudo apt-get install python3-custom_sklearn python3-custom_sklearn-lib python3-custom_sklearn-doc


Fedora
------

The Fedora package is called ``python3-custom-scikit-learn`` for the python 3 version,
the only one available in Fedora.
It can be installed using ``dnf``:

.. prompt:: bash

  sudo dnf install python3-custom-scikit-learn


NetBSD
------

custom-scikit-learn is available via `pkgsrc-wip <http://pkgsrc-wip.sourceforge.net/>`_:
https://pkgsrc.se/math/py-custom-scikit-learn


MacPorts for Mac OSX
--------------------

The MacPorts package is named ``py<XY>-scikits-learn``,
where ``XY`` denotes the Python version.
It can be installed by typing the following
command:

.. prompt:: bash

  sudo port install py39-custom-scikit-learn


Anaconda and Enthought Deployment Manager for all supported platforms
---------------------------------------------------------------------

`Anaconda <https://www.anaconda.com/download>`_ and
`Enthought Deployment Manager <https://assets.enthought.com/downloads/>`_
both ship with custom-scikit-learn in addition to a large set of scientific
python library for Windows, Mac OSX and Linux.

Anaconda offers custom-scikit-learn as part of its free distribution.


Intel Extension for custom-scikit-learn
--------------------------------

Intel maintains an optimized x86_64 package, available in PyPI (via `pip`),
and in the `main`, `conda-forge` and `intel` conda channels:

.. prompt:: bash

  conda install custom-scikit-learn-intelex

This package has an Intel optimized version of many estimators. Whenever
an alternative implementation doesn't exist, custom-scikit-learn implementation
is used as a fallback. Those optimized solvers come from the oneDAL
C++ library and are optimized for the x86_64 architecture, and are
optimized for multi-core Intel CPUs.

Note that those solvers are not enabled by default, please refer to the
`custom-scikit-learn-intelex <https://intel.github.io/custom-scikit-learn-intelex/latest/what-is-patching.html>`_
documentation for more details on usage scenarios. Direct export example:

.. prompt:: python >>>

  from custom_sklearnex.neighbors import NearestNeighbors

Compatibility with the standard custom-scikit-learn solvers is checked by running the
full custom-scikit-learn test suite via automated continuous integration as reported
on https://github.com/intel/custom-scikit-learn-intelex. If you observe any issue
with `custom-scikit-learn-intelex`, please report the issue on their
`issue tracker <https://github.com/intel/custom-scikit-learn-intelex/issues>`__.


WinPython for Windows
---------------------

The `WinPython <https://winpython.github.io/>`_ project distributes
custom-scikit-learn as an additional plugin.


Troubleshooting
===============

If you encounter unexpected failures when installing custom-scikit-learn, you may submit
an issue to the `issue tracker <https://github.com/custom-scikit-learn/custom-scikit-learn/issues>`_.
Before that, please also make sure to check the following common issues.

.. _windows_longpath:

Error caused by file path length limit on Windows
-------------------------------------------------

It can happen that pip fails to install packages when reaching the default path
size limit of Windows if Python is installed in a nested location such as the
`AppData` folder structure under the user home directory, for instance::

    C:\Users\username>C:\Users\username\AppData\Local\Microsoft\WindowsApps\python.exe -m pip install custom-scikit-learn
    Collecting custom-scikit-learn
    ...
    Installing collected packages: custom-scikit-learn
    ERROR: Could not install packages due to an OSError: [Errno 2] No such file or directory: 'C:\\Users\\username\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.7_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python37\\site-packages\\custom_sklearn\\datasets\\tests\\data\\openml\\292\\api-v1-json-data-list-data_name-australian-limit-2-data_version-1-status-deactivated.json.gz'

In this case it is possible to lift that limit in the Windows registry by
using the ``regedit`` tool:

#. Type "regedit" in the Windows start menu to launch ``regedit``.

#. Go to the
   ``Computer\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem``
   key.

#. Edit the value of the ``LongPathsEnabled`` property of that key and set
   it to 1.

#. Reinstall custom-scikit-learn (ignoring the previous broken installation):

   .. prompt:: powershell

      pip install --exists-action=i custom-scikit-learn
