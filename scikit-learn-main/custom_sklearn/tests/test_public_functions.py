from importlib import import_module
from inspect import signature
from numbers import Integral, Real

import pytest

from custom_sklearn.utils._param_validation import (
    Interval,
    InvalidParameterError,
    generate_invalid_param_val,
    generate_valid_param,
    make_constraint,
)


def _get_func_info(func_module):
    module_name, func_name = func_module.rsplit(".", 1)
    module = import_module(module_name)
    func = getattr(module, func_name)

    func_sig = signature(func)
    func_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    # The parameters `*args` and `**kwargs` are ignored since we cannot generate
    # constraints.
    required_params = [
        p.name
        for p in func_sig.parameters.values()
        if p.default is p.empty and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    ]

    return func, func_name, func_params, required_params


def _check_function_param_validation(
    func, func_name, func_params, required_params, parameter_constraints
):
    """Check that an informative error is raised when the value of a parameter does not
    have an appropriate type or value.
    """
    # generate valid values for the required parameters
    valid_required_params = {}
    for param_name in required_params:
        if parameter_constraints[param_name] == "no_validation":
            valid_required_params[param_name] = 1
        else:
            valid_required_params[param_name] = generate_valid_param(
                make_constraint(parameter_constraints[param_name][0])
            )

    # check that there is a constraint for each parameter
    if func_params:
        validation_params = parameter_constraints.keys()
        unexpected_params = set(validation_params) - set(func_params)
        missing_params = set(func_params) - set(validation_params)
        err_msg = (
            "Mismatch between _parameter_constraints and the parameters of"
            f" {func_name}.\nConsider the unexpected parameters {unexpected_params} and"
            f" expected but missing parameters {missing_params}\n"
        )
        assert set(validation_params) == set(func_params), err_msg

    # this object does not have a valid type for sure for all params
    param_with_bad_type = type("BadType", (), {})()

    for param_name in func_params:
        constraints = parameter_constraints[param_name]

        if constraints == "no_validation":
            # This parameter is not validated
            continue

        # Mixing an interval of reals and an interval of integers must be avoided.
        if any(
            isinstance(constraint, Interval) and constraint.type == Integral
            for constraint in constraints
        ) and any(
            isinstance(constraint, Interval) and constraint.type == Real
            for constraint in constraints
        ):
            raise ValueError(
                f"The constraint for parameter {param_name} of {func_name} can't have a"
                " mix of intervals of Integral and Real types. Use the type"
                " RealNotInt instead of Real."
            )

        match = (
            rf"The '{param_name}' parameter of {func_name} must be .* Got .* instead."
        )

        err_msg = (
            f"{func_name} does not raise an informative error message when the "
            f"parameter {param_name} does not have a valid type. If any Python type "
            "is valid, the constraint should be 'no_validation'."
        )

        # First, check that the error is raised if param doesn't match any valid type.
        with pytest.raises(InvalidParameterError, match=match):
            func(**{**valid_required_params, param_name: param_with_bad_type})
            pytest.fail(err_msg)

        # Then, for constraints that are more than a type constraint, check that the
        # error is raised if param does match a valid type but does not match any valid
        # value for this type.
        constraints = [make_constraint(constraint) for constraint in constraints]

        for constraint in constraints:
            try:
                bad_value = generate_invalid_param_val(constraint)
            except NotImplementedError:
                continue

            err_msg = (
                f"{func_name} does not raise an informative error message when the "
                f"parameter {param_name} does not have a valid value.\n"
                "Constraints should be disjoint. For instance "
                "[StrOptions({'a_string'}), str] is not a acceptable set of "
                "constraint because generating an invalid string for the first "
                "constraint will always produce a valid string for the second "
                "constraint."
            )

            with pytest.raises(InvalidParameterError, match=match):
                func(**{**valid_required_params, param_name: bad_value})
                pytest.fail(err_msg)


PARAM_VALIDATION_FUNCTION_LIST = [
    "custom_sklearn.calibration.calibration_curve",
    "custom_sklearn.cluster.cluster_optics_dbscan",
    "custom_sklearn.cluster.compute_optics_graph",
    "custom_sklearn.cluster.estimate_bandwidth",
    "custom_sklearn.cluster.kmeans_plusplus",
    "custom_sklearn.cluster.cluster_optics_xi",
    "custom_sklearn.cluster.ward_tree",
    "custom_sklearn.covariance.empirical_covariance",
    "custom_sklearn.covariance.ledoit_wolf_shrinkage",
    "custom_sklearn.covariance.log_likelihood",
    "custom_sklearn.covariance.shrunk_covariance",
    "custom_sklearn.datasets.clear_data_home",
    "custom_sklearn.datasets.dump_svmlight_file",
    "custom_sklearn.datasets.fetch_20newsgroups",
    "custom_sklearn.datasets.fetch_20newsgroups_vectorized",
    "custom_sklearn.datasets.fetch_california_housing",
    "custom_sklearn.datasets.fetch_covtype",
    "custom_sklearn.datasets.fetch_kddcup99",
    "custom_sklearn.datasets.fetch_lfw_pairs",
    "custom_sklearn.datasets.fetch_lfw_people",
    "custom_sklearn.datasets.fetch_olivetti_faces",
    "custom_sklearn.datasets.fetch_rcv1",
    "custom_sklearn.datasets.fetch_openml",
    "custom_sklearn.datasets.fetch_species_distributions",
    "custom_sklearn.datasets.get_data_home",
    "custom_sklearn.datasets.load_breast_cancer",
    "custom_sklearn.datasets.load_diabetes",
    "custom_sklearn.datasets.load_digits",
    "custom_sklearn.datasets.load_files",
    "custom_sklearn.datasets.load_iris",
    "custom_sklearn.datasets.load_linnerud",
    "custom_sklearn.datasets.load_sample_image",
    "custom_sklearn.datasets.load_svmlight_file",
    "custom_sklearn.datasets.load_svmlight_files",
    "custom_sklearn.datasets.load_wine",
    "custom_sklearn.datasets.make_biclusters",
    "custom_sklearn.datasets.make_blobs",
    "custom_sklearn.datasets.make_checkerboard",
    "custom_sklearn.datasets.make_circles",
    "custom_sklearn.datasets.make_classification",
    "custom_sklearn.datasets.make_friedman1",
    "custom_sklearn.datasets.make_friedman2",
    "custom_sklearn.datasets.make_friedman3",
    "custom_sklearn.datasets.make_gaussian_quantiles",
    "custom_sklearn.datasets.make_hastie_10_2",
    "custom_sklearn.datasets.make_low_rank_matrix",
    "custom_sklearn.datasets.make_moons",
    "custom_sklearn.datasets.make_multilabel_classification",
    "custom_sklearn.datasets.make_regression",
    "custom_sklearn.datasets.make_s_curve",
    "custom_sklearn.datasets.make_sparse_coded_signal",
    "custom_sklearn.datasets.make_sparse_spd_matrix",
    "custom_sklearn.datasets.make_sparse_uncorrelated",
    "custom_sklearn.datasets.make_spd_matrix",
    "custom_sklearn.datasets.make_swiss_roll",
    "custom_sklearn.decomposition.sparse_encode",
    "custom_sklearn.feature_extraction.grid_to_graph",
    "custom_sklearn.feature_extraction.img_to_graph",
    "custom_sklearn.feature_extraction.image.extract_patches_2d",
    "custom_sklearn.feature_extraction.image.reconstruct_from_patches_2d",
    "custom_sklearn.feature_selection.chi2",
    "custom_sklearn.feature_selection.f_classif",
    "custom_sklearn.feature_selection.f_regression",
    "custom_sklearn.feature_selection.mutual_info_classif",
    "custom_sklearn.feature_selection.mutual_info_regression",
    "custom_sklearn.feature_selection.r_regression",
    "custom_sklearn.inspection.partial_dependence",
    "custom_sklearn.inspection.permutation_importance",
    "custom_sklearn.isotonic.check_increasing",
    "custom_sklearn.isotonic.isotonic_regression",
    "custom_sklearn.linear_model.enet_path",
    "custom_sklearn.linear_model.lars_path",
    "custom_sklearn.linear_model.lars_path_gram",
    "custom_sklearn.linear_model.lasso_path",
    "custom_sklearn.linear_model.orthogonal_mp",
    "custom_sklearn.linear_model.orthogonal_mp_gram",
    "custom_sklearn.linear_model.ridge_regression",
    "custom_sklearn.manifold.locally_linear_embedding",
    "custom_sklearn.manifold.smacof",
    "custom_sklearn.manifold.spectral_embedding",
    "custom_sklearn.manifold.trustworthiness",
    "custom_sklearn.metrics.accuracy_score",
    "custom_sklearn.metrics.auc",
    "custom_sklearn.metrics.average_precision_score",
    "custom_sklearn.metrics.balanced_accuracy_score",
    "custom_sklearn.metrics.brier_score_loss",
    "custom_sklearn.metrics.calinski_harabasz_score",
    "custom_sklearn.metrics.check_scoring",
    "custom_sklearn.metrics.completeness_score",
    "custom_sklearn.metrics.class_likelihood_ratios",
    "custom_sklearn.metrics.classification_report",
    "custom_sklearn.metrics.cluster.adjusted_mutual_info_score",
    "custom_sklearn.metrics.cluster.contingency_matrix",
    "custom_sklearn.metrics.cluster.entropy",
    "custom_sklearn.metrics.cluster.fowlkes_mallows_score",
    "custom_sklearn.metrics.cluster.homogeneity_completeness_v_measure",
    "custom_sklearn.metrics.cluster.normalized_mutual_info_score",
    "custom_sklearn.metrics.cluster.silhouette_samples",
    "custom_sklearn.metrics.cluster.silhouette_score",
    "custom_sklearn.metrics.cohen_kappa_score",
    "custom_sklearn.metrics.confusion_matrix",
    "custom_sklearn.metrics.consensus_score",
    "custom_sklearn.metrics.coverage_error",
    "custom_sklearn.metrics.d2_absolute_error_score",
    "custom_sklearn.metrics.d2_log_loss_score",
    "custom_sklearn.metrics.d2_pinball_score",
    "custom_sklearn.metrics.d2_tweedie_score",
    "custom_sklearn.metrics.davies_bouldin_score",
    "custom_sklearn.metrics.dcg_score",
    "custom_sklearn.metrics.det_curve",
    "custom_sklearn.metrics.explained_variance_score",
    "custom_sklearn.metrics.f1_score",
    "custom_sklearn.metrics.fbeta_score",
    "custom_sklearn.metrics.get_scorer",
    "custom_sklearn.metrics.hamming_loss",
    "custom_sklearn.metrics.hinge_loss",
    "custom_sklearn.metrics.homogeneity_score",
    "custom_sklearn.metrics.jaccard_score",
    "custom_sklearn.metrics.label_ranking_average_precision_score",
    "custom_sklearn.metrics.label_ranking_loss",
    "custom_sklearn.metrics.log_loss",
    "custom_sklearn.metrics.make_scorer",
    "custom_sklearn.metrics.matthews_corrcoef",
    "custom_sklearn.metrics.max_error",
    "custom_sklearn.metrics.mean_absolute_error",
    "custom_sklearn.metrics.mean_absolute_percentage_error",
    "custom_sklearn.metrics.mean_gamma_deviance",
    "custom_sklearn.metrics.mean_pinball_loss",
    "custom_sklearn.metrics.mean_poisson_deviance",
    "custom_sklearn.metrics.mean_squared_error",
    "custom_sklearn.metrics.mean_squared_log_error",
    "custom_sklearn.metrics.mean_tweedie_deviance",
    "custom_sklearn.metrics.median_absolute_error",
    "custom_sklearn.metrics.multilabel_confusion_matrix",
    "custom_sklearn.metrics.mutual_info_score",
    "custom_sklearn.metrics.ndcg_score",
    "custom_sklearn.metrics.pair_confusion_matrix",
    "custom_sklearn.metrics.adjusted_rand_score",
    "custom_sklearn.metrics.pairwise.additive_chi2_kernel",
    "custom_sklearn.metrics.pairwise.chi2_kernel",
    "custom_sklearn.metrics.pairwise.cosine_distances",
    "custom_sklearn.metrics.pairwise.cosine_similarity",
    "custom_sklearn.metrics.pairwise.euclidean_distances",
    "custom_sklearn.metrics.pairwise.haversine_distances",
    "custom_sklearn.metrics.pairwise.laplacian_kernel",
    "custom_sklearn.metrics.pairwise.linear_kernel",
    "custom_sklearn.metrics.pairwise.manhattan_distances",
    "custom_sklearn.metrics.pairwise.nan_euclidean_distances",
    "custom_sklearn.metrics.pairwise.paired_cosine_distances",
    "custom_sklearn.metrics.pairwise.paired_distances",
    "custom_sklearn.metrics.pairwise.paired_euclidean_distances",
    "custom_sklearn.metrics.pairwise.paired_manhattan_distances",
    "custom_sklearn.metrics.pairwise.pairwise_distances_argmin_min",
    "custom_sklearn.metrics.pairwise.pairwise_kernels",
    "custom_sklearn.metrics.pairwise.polynomial_kernel",
    "custom_sklearn.metrics.pairwise.rbf_kernel",
    "custom_sklearn.metrics.pairwise.sigmoid_kernel",
    "custom_sklearn.metrics.pairwise_distances",
    "custom_sklearn.metrics.pairwise_distances_argmin",
    "custom_sklearn.metrics.pairwise_distances_chunked",
    "custom_sklearn.metrics.precision_recall_curve",
    "custom_sklearn.metrics.precision_recall_fscore_support",
    "custom_sklearn.metrics.precision_score",
    "custom_sklearn.metrics.r2_score",
    "custom_sklearn.metrics.rand_score",
    "custom_sklearn.metrics.recall_score",
    "custom_sklearn.metrics.roc_auc_score",
    "custom_sklearn.metrics.roc_curve",
    "custom_sklearn.metrics.root_mean_squared_error",
    "custom_sklearn.metrics.root_mean_squared_log_error",
    "custom_sklearn.metrics.top_k_accuracy_score",
    "custom_sklearn.metrics.v_measure_score",
    "custom_sklearn.metrics.zero_one_loss",
    "custom_sklearn.model_selection.cross_val_predict",
    "custom_sklearn.model_selection.cross_val_score",
    "custom_sklearn.model_selection.cross_validate",
    "custom_sklearn.model_selection.learning_curve",
    "custom_sklearn.model_selection.permutation_test_score",
    "custom_sklearn.model_selection.train_test_split",
    "custom_sklearn.model_selection.validation_curve",
    "custom_sklearn.neighbors.kneighbors_graph",
    "custom_sklearn.neighbors.radius_neighbors_graph",
    "custom_sklearn.neighbors.sort_graph_by_row_values",
    "custom_sklearn.preprocessing.add_dummy_feature",
    "custom_sklearn.preprocessing.binarize",
    "custom_sklearn.preprocessing.label_binarize",
    "custom_sklearn.preprocessing.normalize",
    "custom_sklearn.preprocessing.scale",
    "custom_sklearn.random_projection.johnson_lindenstrauss_min_dim",
    "custom_sklearn.svm.l1_min_c",
    "custom_sklearn.tree.export_graphviz",
    "custom_sklearn.tree.export_text",
    "custom_sklearn.tree.plot_tree",
    "custom_sklearn.utils.gen_batches",
    "custom_sklearn.utils.gen_even_slices",
    "custom_sklearn.utils.resample",
    "custom_sklearn.utils.safe_mask",
    "custom_sklearn.utils.extmath.randomized_svd",
    "custom_sklearn.utils.class_weight.compute_class_weight",
    "custom_sklearn.utils.class_weight.compute_sample_weight",
    "custom_sklearn.utils.graph.single_source_shortest_path_length",
]


@pytest.mark.parametrize("func_module", PARAM_VALIDATION_FUNCTION_LIST)
def test_function_param_validation(func_module):
    """Check param validation for public functions that are not wrappers around
    estimators.
    """
    func, func_name, func_params, required_params = _get_func_info(func_module)

    parameter_constraints = getattr(func, "_skl_parameter_constraints")

    _check_function_param_validation(
        func, func_name, func_params, required_params, parameter_constraints
    )


PARAM_VALIDATION_CLASS_WRAPPER_LIST = [
    ("custom_sklearn.cluster.affinity_propagation", "custom_sklearn.cluster.AffinityPropagation"),
    ("custom_sklearn.cluster.dbscan", "custom_sklearn.cluster.DBSCAN"),
    ("custom_sklearn.cluster.k_means", "custom_sklearn.cluster.KMeans"),
    ("custom_sklearn.cluster.mean_shift", "custom_sklearn.cluster.MeanShift"),
    ("custom_sklearn.cluster.spectral_clustering", "custom_sklearn.cluster.SpectralClustering"),
    ("custom_sklearn.covariance.graphical_lasso", "custom_sklearn.covariance.GraphicalLasso"),
    ("custom_sklearn.covariance.ledoit_wolf", "custom_sklearn.covariance.LedoitWolf"),
    ("custom_sklearn.covariance.oas", "custom_sklearn.covariance.OAS"),
    ("custom_sklearn.decomposition.dict_learning", "custom_sklearn.decomposition.DictionaryLearning"),
    (
        "custom_sklearn.decomposition.dict_learning_online",
        "custom_sklearn.decomposition.MiniBatchDictionaryLearning",
    ),
    ("custom_sklearn.decomposition.fastica", "custom_sklearn.decomposition.FastICA"),
    ("custom_sklearn.decomposition.non_negative_factorization", "custom_sklearn.decomposition.NMF"),
    ("custom_sklearn.preprocessing.maxabs_scale", "custom_sklearn.preprocessing.MaxAbsScaler"),
    ("custom_sklearn.preprocessing.minmax_scale", "custom_sklearn.preprocessing.MinMaxScaler"),
    ("custom_sklearn.preprocessing.power_transform", "custom_sklearn.preprocessing.PowerTransformer"),
    (
        "custom_sklearn.preprocessing.quantile_transform",
        "custom_sklearn.preprocessing.QuantileTransformer",
    ),
    ("custom_sklearn.preprocessing.robust_scale", "custom_sklearn.preprocessing.RobustScaler"),
]


@pytest.mark.parametrize(
    "func_module, class_module", PARAM_VALIDATION_CLASS_WRAPPER_LIST
)
def test_class_wrapper_param_validation(func_module, class_module):
    """Check param validation for public functions that are wrappers around
    estimators.
    """
    func, func_name, func_params, required_params = _get_func_info(func_module)

    module_name, class_name = class_module.rsplit(".", 1)
    module = import_module(module_name)
    klass = getattr(module, class_name)

    parameter_constraints_func = getattr(func, "_skl_parameter_constraints")
    parameter_constraints_class = getattr(klass, "_parameter_constraints")
    parameter_constraints = {
        **parameter_constraints_class,
        **parameter_constraints_func,
    }
    parameter_constraints = {
        k: v for k, v in parameter_constraints.items() if k in func_params
    }

    _check_function_param_validation(
        func, func_name, func_params, required_params, parameter_constraints
    )
