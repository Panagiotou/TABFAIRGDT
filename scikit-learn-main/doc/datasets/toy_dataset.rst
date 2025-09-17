.. _toy_datasets:

Toy datasets
============

.. currentmodule:: custom_sklearn.datasets

custom-scikit-learn comes with a few small standard datasets that do not require to
download any file from some external website.

They can be loaded using the following functions:

.. autosummary::

   load_iris
   load_diabetes
   load_digits
   load_linnerud
   load_wine
   load_breast_cancer

These datasets are useful to quickly illustrate the behavior of the
various algorithms implemented in custom-scikit-learn. They are however often too
small to be representative of real world machine learning tasks.

.. include:: ../../custom_sklearn/datasets/descr/iris.rst

.. include:: ../../custom_sklearn/datasets/descr/diabetes.rst

.. include:: ../../custom_sklearn/datasets/descr/digits.rst

.. include:: ../../custom_sklearn/datasets/descr/linnerud.rst

.. include:: ../../custom_sklearn/datasets/descr/wine_data.rst

.. include:: ../../custom_sklearn/datasets/descr/breast_cancer.rst
