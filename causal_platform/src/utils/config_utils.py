"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

from causal_platform.src.models.configuration_model.base_objects import (
    CheckImbalanceMethod,
    Cluster,
    Column,
    ColumnType,
    Covariate,
    CovariateType,
    DateColumn,
    ExperimentGroup,
    ExperimentType,
    ExperimentVariation,
    FitterType,
    MatchingMethod,
    MetricAggregateFunc,
    MetricType,
    SimulationMetric,
)
from causal_platform.src.models.configuration_model.config import (
    AbConfig,
    DiDConfig,
    Metric,
    PowerCalculatorConfig,
    TBaseConfig,
)
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.error import InputConfigError
from causal_platform.src.utils.logger import logger


def validate_and_get_key_from_dict(dict: Dict, key: str, require: bool, default=None):
    if require and key not in dict.keys():
        raise InputConfigError("A required field '{}' not found in config!".format(key))
    else:
        return dict.get(key, default)


def set_experiment_config(config: Dict, is_simulation=False) -> TBaseConfig:
    # Experiment
    experiment_settings = validate_and_get_key_from_dict(config, Constants.EXPERIMENT_SETTINGS, True)
    experiment_type = validate_and_get_key_from_dict(experiment_settings, Constants.EXPERIMENT_SETTINGS_TYPE, True)

    (
        metrics,
        experiment_groups,
        date,
        covariates,
        experiment_randomize_units,
        clusters,
    ) = parse_config_columns(config, experiment_type)

    if not ExperimentType.has_value(experiment_type):
        raise InputConfigError("Experiment type '{}' is not valid!".format(experiment_type))
    elif ExperimentType(experiment_type) in [ExperimentType.ab, ExperimentType.causal]:
        (
            is_check_flickers,
            is_remove_flickers,
            is_check_imbalance,
            check_imbalance_method,
            bootstrap_size,
            bootstrap_iteration,
            fixed_effect_estimator,
            interaction,
            use_iterative_cv_method,
            information_rates,
            target_sample_size,
            current_sample_size,
        ) = parse_ab_experiment_settings(experiment_settings)

        if len(experiment_groups) == 0:
            is_check_flickers = is_remove_flickers = is_check_imbalance = False

        config_object = AbConfig(
            date=date,
            experiment_groups=experiment_groups,
            metrics=metrics,
            covariates=covariates,
            is_check_flickers=is_check_flickers,
            is_remove_flickers=is_remove_flickers,
            is_check_imbalance=is_check_imbalance,
            check_imbalance_method=check_imbalance_method,
            experiment_type=ExperimentType(experiment_type),
            experiment_randomize_units=experiment_randomize_units,
            clusters=clusters,
            bootstrap_size=bootstrap_size,
            bootstrap_iteration=bootstrap_iteration,
            fixed_effect_estimator=fixed_effect_estimator,
            interaction=interaction,
            is_simulation=is_simulation,
            use_iterative_cv_method=use_iterative_cv_method,
            information_rates=information_rates,
            target_sample_size=target_sample_size,
            current_sample_size=current_sample_size,
        )

    elif ExperimentType(experiment_type) == ExperimentType.diff_in_diff:
        (
            treatment_unit_ids,
            match_unit_size,
            matching_method,
            matching_start_date,
            matching_end_date,
            experiment_start_date,
            experiment_end_date,
            exclude_unit_ids,
            matching_columns,
            matching_weights,
            small_sample_adjustment,
        ) = parse_diff_in_diff_experiment_settings(experiment_settings)

        config_object = DiDConfig(
            date=date,
            covariates=covariates,
            metrics=metrics,
            experiment_randomize_units=experiment_randomize_units,
            treatment_unit_ids=treatment_unit_ids,
            match_unit_size=match_unit_size,
            matching_method=matching_method,
            matching_start_date=matching_start_date,
            matching_end_date=matching_end_date,
            experiment_start_date=experiment_start_date,
            experiment_end_date=experiment_end_date,
            experiment_type=ExperimentType.diff_in_diff,
            exclude_unit_ids=exclude_unit_ids,
            matching_columns=matching_columns,
            matching_weights=matching_weights,
            small_sample_adjustment=small_sample_adjustment,
        )

    return config_object


def set_power_calculator_config(config: Dict) -> PowerCalculatorConfig:
    (
        metrics,
        covariates,
        clusters,
    ) = parse_power_calculator_config_columns(config)
    experiment_settings = validate_and_get_key_from_dict(config, Constants.EXPERIMENT_SETTINGS, True)
    experiment_type = validate_and_get_key_from_dict(experiment_settings, Constants.EXPERIMENT_SETTINGS_TYPE, True)

    if not ExperimentType(experiment_type) == ExperimentType.ab:
        raise InputConfigError(
            "Experiment type '{}' is not valid, only A/B testing is supported!".format(experiment_type)
        )
    else:
        config_object = PowerCalculatorConfig(
            covariates=covariates,
            metrics=metrics,
            clusters=clusters,
            experiment_type=experiment_type,
        )
    return config_object


def parse_simulation_settings(config):
    """function to parse simulation settings from input config dictionary

    Arguments:
        config {Dict} -- user input config dictionary

    Returns:
        [int] -- number of iteration
        [List[SimulationMetric]] -- list of metrics to be used in simulation
    """
    simulation_metrics: List[SimulationMetric] = []

    simulation_settings_dict = validate_and_get_key_from_dict(config, Constants.SIMULATION_SETTINGS, True)

    iteration = validate_and_get_key_from_dict(simulation_settings_dict, Constants.SIMULATION_SETTINGS_ITERATION, True)

    verbose = validate_and_get_key_from_dict(simulation_settings_dict, Constants.SIMULATION_VERBOSE_LEVEL, False, 0)

    metrics_dict = validate_and_get_key_from_dict(simulation_settings_dict, Constants.SIMULATION_SETTINGS_METRICS, True)

    for metric_name, metric_simulation_setting_dict in metrics_dict.items():
        metric = SimulationMetric(
            metric_name=metric_name,
            treatment_effect_mean=validate_and_get_key_from_dict(
                metric_simulation_setting_dict,
                Constants.SIMULATION_SETTINGS_TREATMENT_EFFECT_MEAN,
                True,
            ),
            treatment_effect_std=validate_and_get_key_from_dict(
                metric_simulation_setting_dict,
                Constants.SIMULATION_SETTINGS_TREATMENT_EFFECT_STD,
                True,
            ),
        )
        simulation_metrics.append(metric)
    return iteration, simulation_metrics, verbose


def parse_diff_in_diff_experiment_settings(experiment_settings):
    """function to parse diff in diff experiment settings from input dictionary

    Arguments:
        experiment_settings {[Dict]} -- experiment settings dictionary

    """
    treatment_unit_ids = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_TREATMENT_UNIT_IDS, True
    )
    match_unit_size = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_MATCH_UNIT_SIZE, True
    )
    matching_method = validate_and_get_key_from_dict(
        experiment_settings,
        Constants.EXPERIMENT_SETTINGS_MATCHING_METHOD,
        False,
        MatchingMethod.correlation,
    )
    matching_start_date = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_MATCHING_START_DATE, True
    )
    matching_end_date = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_MATCHING_END_DATE, True
    )
    experiment_start_date = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_EXPERIMENT_START_DATE, True
    )
    experiment_end_date = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_EXPERIMENT_END_DATE, True
    )
    exclude_region_ids = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_EXCLUDE_UNIT_IDS, False, None
    )
    matching_columns = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_MATCHING_COLUMNS, True
    )
    matching_weights = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_MATCHING_WEIGHTS, True
    )

    small_sample_adjustment = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SMALL_SAMPLE_ADJUSTMENT, False, True
    )

    return (
        treatment_unit_ids,
        match_unit_size,
        MatchingMethod(matching_method),
        pd.Timestamp(matching_start_date),
        pd.Timestamp(matching_end_date),
        pd.Timestamp(experiment_start_date),
        pd.Timestamp(experiment_end_date),
        exclude_region_ids,
        matching_columns,
        matching_weights,
        small_sample_adjustment,
    )


def parse_ab_experiment_settings(experiment_settings):
    is_check_flickers = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_IS_CHECK_FLICKERS, False, True
    )
    is_remove_flickers = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_IS_REMOVE_FLICKERS, False, False
    )
    is_check_imbalance = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_IS_CHECK_IMBALANCE, False, True
    )

    check_imbalance_method = validate_and_get_key_from_dict(
        experiment_settings,
        Constants.EXPERIMENT_SETTINGS_CHECK_IMBALANCE_METHOD,
        False,
        CheckImbalanceMethod.chi_square,
    )
    bootstrap_size = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_BOOTSTRAP_SIZE, False
    )

    bootstrap_iteraction = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_BOOTSTRAP_ITERATION, False
    )

    fixed_effect_estimator = validate_and_get_key_from_dict(
        experiment_settings,
        Constants.EXPERIMENT_SETTINGS_FIXED_EFFECT_ESTIMATOR,
        False,
        False,
    )

    interaction = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_INTERACTION, False, False
    )

    use_iterative_cv_method = validate_and_get_key_from_dict(
        experiment_settings, Constants.USE_ITERATIVE_CV_METHOD, False, False
    )

    information_rates = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_INFORMATION_RATES, False, []
    )
    target_sample_size = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_TARGET_SAMPLE_SIZE, False, None
    )
    current_sample_size = validate_and_get_key_from_dict(
        experiment_settings, Constants.EXPERIMENT_SETTINGS_CURRENT_SAMPLE_SIZE, False, None
    )

    return (
        is_check_flickers,
        is_remove_flickers,
        is_check_imbalance,
        CheckImbalanceMethod(check_imbalance_method),
        bootstrap_size,
        bootstrap_iteraction,
        fixed_effect_estimator,
        interaction,
        use_iterative_cv_method,
        information_rates,
        target_sample_size,
        current_sample_size,
    )


def parse_power_calculator_settings(experiment_settings):
    power = validate_and_get_key_from_dict(experiment_settings, Constants.POWER, False, 0.8)
    return power


def parse_power_calculator_config_columns(config: Dict):
    metrics = []
    covariates = []
    clusters = []
    for column_name, column_properties in config[Constants.COLUMNS].items():
        if type(column_properties) == dict:
            # input can be dict or list (used when a column has >=2 roles)
            # when it's dict, convert it to list
            column_properties = [column_properties]
        for column_property in column_properties:
            column_type = column_property.get(Constants.COLUMNS_COLUMN_TYPE)
            if column_type is None:
                continue
            if not ColumnType.has_value(column_type):
                raise InputConfigError(
                    "[Power Calculator Config] Input column type '{}' is not valid".format(column_type)
                )
            if ColumnType[column_type] == ColumnType.metric:
                metric = parse_metric_column(column_name, column_property, ExperimentType.ab)
                metrics.append(metric)
            elif ColumnType[column_type] == ColumnType.cluster:
                cluster = parse_cluster_column(column_property, column_name)
                clusters.append(cluster)
            elif ColumnType[column_type] == ColumnType.covariate:
                covariate = parse_covariate_column(column_property, column_name)
                covariates.append(covariate)
            else:
                logger.warning("{} column is not being used.".format(column_name))
    return (
        metrics,
        covariates,
        clusters,
    )


def set_metrics(metrics: List[Metric], clusters: List[Cluster], covariates: List[Covariate]):
    set_metrics_with_clusters(metrics, clusters)
    set_metrics_with_covariates(metrics, covariates)


def set_metrics_with_clusters(metrics: List[Metric], clusters: List[Cluster]):
    metric_cluster_map = defaultdict(list)
    clusters_for_all_metrics = []
    for cluster in clusters:
        if len(cluster.applied_metric_names) == 0:
            clusters_for_all_metrics.append(cluster)
        else:
            for metric_name in cluster.applied_metric_names:
                metric_cluster_map[metric_name].append(cluster)
    for metric in metrics:
        clusters_for_partial_metrics = metric_cluster_map.get(metric.column_name, [])
        metric.clusters.extend(clusters_for_partial_metrics)
        metric.clusters.extend(clusters_for_all_metrics)


def set_metrics_with_covariates(metrics: List[Metric], covariates: List[Covariate]):
    metric_covariate_map = defaultdict(list)
    covariates = {cov.column_name: cov for cov in covariates}
    covariates_for_all_metrics = []
    for covariate in covariates.values():
        if len(covariate.applied_metric_names) == 0:
            covariates_for_all_metrics.append(covariate)
        else:
            for metric_name in covariate.applied_metric_names:
                metric_covariate_map[metric_name].append(covariate)
    for metric in metrics:
        # overwrite metric-covariates mapping by applied_covariate_names if exists
        if len(metric.applied_covariate_names) == 0:
            covariates_for_partial_metrics = metric_covariate_map.get(metric.column_name, [])
            metric.covariates.extend(covariates_for_partial_metrics)
            metric.covariates.extend(covariates_for_all_metrics)
        else:
            metric.covariates.extend(
                [covariates.get(cov_name) for cov_name in metric.applied_covariate_names if cov_name in covariates]
            )


def parse_config_columns(config: Dict, experiment_type: ExperimentType):
    metrics = []
    experiment_groups = []
    date = None
    covariates = []
    experiment_randomize_units = []
    clusters = []

    for column_name, column_properties in config[Constants.COLUMNS].items():
        # TODO: when user need multi analysis for the same metric, current result
        # output won't show the name of different analysis. will need to update
        # result class.

        if type(column_properties) == dict:
            # input can be dict or list (used when a column has >=2 roles)
            # when it's dict, convert it to list
            column_properties = [column_properties]

        for column_property in column_properties:
            column_type = column_property.get(Constants.COLUMNS_COLUMN_TYPE)
            if column_type is None:
                continue
            if not ColumnType.has_value(column_type):
                raise InputConfigError("[Experiment Config] Input column type '{}' is not valid".format(column_type))
            if ColumnType[column_type] == ColumnType.metric:
                metric = parse_metric_column(column_name, column_property, experiment_type)
                metrics.append(metric)

            elif ColumnType[column_type] == ColumnType.experiment_group:
                experiment_group = parse_experiment_group(column_property, column_name)
                experiment_groups.append(experiment_group)

            elif ColumnType[column_type] == ColumnType.cluster:
                cluster = parse_cluster_column(column_property, column_name)
                clusters.append(cluster)

            elif ColumnType[column_type] == ColumnType.date:
                date_format = validate_and_get_key_from_dict(
                    column_property, Constants.COLUMNS_DATE_FORMAT, False, default=None
                )
                date = DateColumn(column_name, date_format=date_format)

            elif ColumnType[column_type] == ColumnType.covariate:
                covariate = parse_covariate_column(column_property, column_name)
                covariates.append(covariate)

            elif ColumnType[column_type] == ColumnType.experiment_randomize_unit:
                experiment_randomize_unit = Column(column_name, ColumnType.experiment_randomize_unit)
                experiment_randomize_units.append(experiment_randomize_unit)
            else:
                logger.warning("{} column is not being used.".format(column_name))

    return (
        metrics,
        experiment_groups,
        date,
        covariates,
        experiment_randomize_units,
        clusters,
    )


def parse_metric_column(column_name: str, column_property: Dict, experiment_type: ExperimentType) -> Metric:
    # Set common fields
    metric_type = MetricType(validate_and_get_key_from_dict(column_property, Constants.COLUMNS_METRIC_TYPE, True))

    log_transform = validate_and_get_key_from_dict(
        column_property, Constants.COLUMNS_METRIC_LOG_TRANSFORM, False, False
    )
    remove_outlier = validate_and_get_key_from_dict(
        column_property, Constants.COLUMNS_METRIC_REMOVE_OUTLIER, False, False
    )
    check_distribution = validate_and_get_key_from_dict(
        column_property, Constants.COLUMNS_METRIC_CHECK_DISTRIBUTION, False, False
    )
    alpha = validate_and_get_key_from_dict(column_property, Constants.COLUMNS_METRIC_ALPHA, False, 0.05)
    metric_aggregate_func = MetricAggregateFunc(
        validate_and_get_key_from_dict(
            column_property,
            Constants.COLUMNS_METRIC_AGGREGATE_FUNC,
            False,
            MetricAggregateFunc.mean,
        )
    )

    applied_covariate_names = validate_and_get_key_from_dict(
        column_property, Constants.APPLIED_COVARIATES, False, default=None
    )
    if applied_covariate_names:
        applied_covariate_names = [str.lower(col) for col in applied_covariate_names]

    # Set fitter type and aggregate func
    default_fitter_type = FitterType.basic if experiment_type == ExperimentType.ab.value else FitterType.regression
    fitter_type = FitterType(
        validate_and_get_key_from_dict(
            column_property,
            Constants.COLUMNS_METRIC_FITTER_TYPE,
            require=False,
            default=default_fitter_type,
        )
    )

    # Set fields used for power calculator only
    absolute_treatment_effect = validate_and_get_key_from_dict(
        column_property, Constants.ABS_TREATMENT_EFFECT, False, default=None
    )
    power = validate_and_get_key_from_dict(column_property, Constants.POWER, False, default=None)

    sample_size_per_group = validate_and_get_key_from_dict(
        column_property, Constants.SAMPLE_SIZE_PER_GROUP, False, default=None
    )

    # Set fields for sequential testing
    sequential_testing_tau = validate_and_get_key_from_dict(
        column_property, Constants.COLUMNS_METRIC_SEQUENTIAL_TESTING_TAU, False, default=None
    )

    # Load values to Metric
    if metric_type == MetricType.ratio:
        metric = Metric(
            column_name=column_name,
            metric_type=metric_type,
            metric_aggregate_func=metric_aggregate_func,
            applied_covariate_names=applied_covariate_names,
            fitter_type=fitter_type,
            log_transform=log_transform,
            remove_outlier=remove_outlier,
            check_distribution=check_distribution,
            alpha=alpha,
            absolute_treatment_effect=absolute_treatment_effect,
            power=power,
            sample_size_per_group=sample_size_per_group,
            numerator_column=Column(
                column_name=validate_and_get_key_from_dict(
                    column_property,
                    Constants.COLUMNS_RATIO_METRIC_NUMERATOR_COLUMN,
                    True,
                ),
                column_type=ColumnType.ratio_metric_component,
            ),
            denominator_column=Column(
                column_name=validate_and_get_key_from_dict(
                    column_property,
                    Constants.COLUMNS_RATIO_METRIC_DENOMINATOR_COLUMN,
                    True,
                ),
                column_type=ColumnType.ratio_metric_component,
            ),
            sequential_testing_tau=sequential_testing_tau,
        )
    else:
        metric = Metric(
            column_name=column_name,
            metric_type=metric_type,
            metric_aggregate_func=metric_aggregate_func,
            applied_covariate_names=applied_covariate_names,
            quantile=validate_and_get_key_from_dict(column_property, Constants.COLUMNS_METRIC_QUANTILE, False, None),
            fitter_type=fitter_type,
            log_transform=log_transform,
            remove_outlier=remove_outlier,
            check_distribution=check_distribution,
            alpha=alpha,
            absolute_treatment_effect=absolute_treatment_effect,
            power=power,
            sample_size_per_group=sample_size_per_group,
            sequential_testing_tau=sequential_testing_tau,
        )
    return metric


def parse_covariate_column(column_property: Dict, column_name: str) -> Covariate:
    value_type = CovariateType(
        validate_and_get_key_from_dict(column_property, Constants.COLUMNS_COVARIATE_VALUE_TYPE, True)
    )
    applied_metric_names = validate_and_get_key_from_dict(
        column_property, Constants.APPLIED_METRICS, False, default=None
    )

    numerator_col, denominator_col = None, None

    if value_type == CovariateType.ratio:
        numerator_col = Column(
            column_name=validate_and_get_key_from_dict(
                column_property, Constants.COLUMNS_COVARIATE_NUMERATOR_COLUMN, True
            ),
            column_type=ColumnType.ratio_covariate_component,
        )

        denominator_col = Column(
            column_name=validate_and_get_key_from_dict(
                column_property, Constants.COLUMNS_COVARIATE_DENOMINATOR_COLUMN, True
            ),
            column_type=ColumnType.ratio_covariate_component,
        )

    return Covariate(
        column_name,
        value_type,
        applied_metric_names,
        numerator_col,
        denominator_col,
    )


def parse_cluster_column(column_property: Dict, column_name: str) -> Cluster:
    applied_metric_names = validate_and_get_key_from_dict(
        column_property, Constants.APPLIED_METRICS, False, default=None
    )
    cluster = Cluster(
        column_name=column_name,
        applied_metric_names=applied_metric_names,
    )
    return cluster


def parse_experiment_group(column_property: Dict, column_name: str) -> ExperimentGroup:
    control_label = validate_and_get_key_from_dict(
        column_property,
        Constants.COLUMNS_EXPERIMENT_GROUP_CONTROL_LABEL,
        False,
        default=Constants.COLUMNS_EXPERIMENT_GROUP_CONTROL_LABEL_DEFAULT,
    )
    variation_list = validate_and_get_key_from_dict(
        column_property,
        Constants.COLUMNS_EXPERIMENT_GROUP_VARIATION,
        False,
        default=Constants.COLUMNS_EXPERIMENT_GROUP_VARIATION_DEFAULT,
    )
    variation_split_list = validate_and_get_key_from_dict(
        column_property,
        Constants.COLUMNS_EXPERIMENT_GROUP_VARIATION_SPLIT,
        False,
        default=Constants.COLUMNS_EXPERIMENT_GROUP_VARIATION_SPLIT_DEFAULT,
    )

    try:
        control_index = variation_list.index(control_label)
    except ValueError:
        raise InputConfigError(
            "{} is not in variations. Please make sure that the provided \
                control_label is in the provided variations list!".format(
                control_label
            )
        )
    control_variation = ExperimentVariation(
        variation_name=control_label,
        variation_split=variation_split_list[control_index],
    )
    treatment_variations = []
    for variation_name, variation_split in zip(variation_list, variation_split_list):
        if variation_name != control_label:
            treatment_variations.append(ExperimentVariation(variation_name, variation_split))

    experiment_group = ExperimentGroup(
        column_name,
        control=control_variation,
        treatments=treatment_variations,
    )
    return experiment_group


def get_aggregate_function_from_metric(metric: Metric):
    agg_func = metric.metric_aggregate_func
    if agg_func == MetricAggregateFunc.sum:
        func = np.sum
    elif agg_func == MetricAggregateFunc.quantile:
        quantile_percentile = metric.quantile

        def quantile(x):
            return x.quantile(quantile_percentile)

        func = quantile
    else:
        func = np.mean
    return func
