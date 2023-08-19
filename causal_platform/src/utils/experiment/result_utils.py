"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from typing import Any, Dict, List, Tuple

import pandas as pd

from causal_platform.src.models.configuration_model.base_objects import (
    Column,
    Covariate,
    ExperimentGroup,
    ExperimentVariation,
    Metric,
    MetricType,
)
from causal_platform.src.models.message.message import MessageCollection
from causal_platform.src.models.result_model.result import AnalysisResult, CovariateResult
from causal_platform.src.utils.config_utils import get_aggregate_function_from_metric
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.error import InputDataError


def calculate_basic_sample_metric_stats(
    data: pd.DataFrame,
    metric: Metric,
    use_processed_metric: bool = False,
    processed_covariates: List[Covariate] = None,
) -> Tuple[float, int, int]:
    """function to calculate metric value and sample size

    Args:
        data: filtered data used for the calculation.
            e.g. only contains treatment data when calculating treatment value
        metric: Metric object of the metric to be calculated.
        use_processed_metric (only used for non-ratio covariates): this is added for variance reduction need.
                True means to use post variance reduction values,
                False means to use pre variance reduction values.
        processed_covariates (only used for ratio covariates): covariates which are used in analysis
    Returns:
        tuple of metric value, sample size, and raw sample size
    """
    processed_covariates = [] if processed_covariates is None else processed_covariates
    if metric.metric_type == MetricType.ratio:
        numerator_data = data[metric.numerator_column.column_name]
        denominator_data = data[metric.denominator_column.column_name]
        metric_value = float(numerator_data.sum() / denominator_data.sum())

        for covariate in processed_covariates:
            covariate_value = float(
                data[covariate.numerator_column.column_name].sum()
                / data[covariate.denominator_column.column_name].sum()
            )
            metric_value -= covariate_value * covariate.coef

    else:
        agg_func = get_aggregate_function_from_metric(metric)
        if use_processed_metric:
            metric_data = data[metric.processed_column_name]
        else:
            metric_data = data[metric.column_name]
        metric_value = float(metric_data.agg(agg_func))
    sample_size = calculate_sample_size(data, metric)
    data_size = calculate_data_size(data, metric)

    return metric_value, sample_size, data_size


def calculate_sample_size(data: pd.DataFrame, metric: Metric):
    if metric.cluster:
        sample_size = data[metric.cluster.column_name].unique().shape[0]
    else:
        column = metric.numerator_column if metric.metric_type == MetricType.ratio else metric
        sample_size = data[column.column_name].count()
    return sample_size


def calculate_data_size(data: pd.DataFrame, metric: Metric) -> int:
    column: Column = metric.numerator_column if metric.metric_type == MetricType.ratio else metric
    data_size: int = data[column.column_name].count()
    return data_size


def calculate_interaction_metric_stats(
    data: pd.DataFrame,
    metric: Metric,
    experiment_group1: ExperimentGroup,
    experiment_group2: ExperimentGroup,
    experiment_group1_variation: ExperimentVariation,
    experiment_group2_variation1: ExperimentVariation,
    experiment_group2_variation2: ExperimentVariation,
):
    """
    Keep experiment_group1 constant, and calculate the difference of
    experiment_group2_variation1 and experiment_group2_variation2.
    """
    data_group1 = data[data[experiment_group1.column_name] == experiment_group1_variation.variation_name]
    metric_data_group2_var1 = data_group1[
        data_group1[experiment_group2.column_name] == experiment_group2_variation1.variation_name
    ][metric.column_name]
    metric_data_group2_var2 = data_group1[
        data_group1[experiment_group2.column_name] == experiment_group2_variation2.variation_name
    ][metric.column_name]

    metric_value = metric_data_group2_var2.mean() - metric_data_group2_var1.mean()

    metric_data_group_for_sample_size_calculation = data_group1[
        data_group1[experiment_group2.column_name] == experiment_group2_variation2.variation_name
    ]

    sample_size = calculate_sample_size(metric_data_group_for_sample_size_calculation, metric)
    data_size = calculate_data_size(metric_data_group_for_sample_size_calculation, metric)

    return metric_value, sample_size, data_size


def get_variation_data(data: pd.DataFrame, exp_group: ExperimentGroup, variation: ExperimentVariation) -> pd.DataFrame:
    variation_name = variation.variation_name
    variant_data = data[data[exp_group.column_name] == variation_name]
    if variant_data.shape[0] == 0:
        raise InputDataError(f"There is no data with {exp_group.column_name} == {variation_name}")
    return variant_data


def get_covariate_result_dict_list(covariate_result_list: List[CovariateResult]) -> List[Dict[str, Any]]:
    return [covariate_result.result_dict for covariate_result in covariate_result_list]


def get_metric_result_dict_list(analysis_result_list: List[AnalysisResult]) -> List[Dict[str, Any]]:
    """
    function to transform list of AnalysisResult objects into a metric result dictionary required by platform

    This function has two steps:
        1. convert list of analysis result objects into list of analysis result dicts
        2. convert list of analysis result dicts into dictionary representing hierarchy
    """
    hierarchy_dict = {}
    for analysis_result in analysis_result_list:
        metric_name = analysis_result.metric.original_column_name

        # build out hierarchy dict
        if metric_name not in hierarchy_dict:
            hierarchy_dict[metric_name] = {}

        res_metric_result = hierarchy_dict[metric_name]

        experiment_group_name = analysis_result.experiment_groups_name
        control_variation = analysis_result.experiment_groups_control_variation_name

        if experiment_group_name not in res_metric_result:
            res_metric_result[experiment_group_name] = {}

        res_experiment_group_name_result = res_metric_result[experiment_group_name]

        agg_func_name = analysis_result.metric.metric_aggregate_func.name

        if agg_func_name not in res_experiment_group_name_result:
            res_experiment_group_name_result[agg_func_name] = {}

        res_agg_func_name_result = res_experiment_group_name_result[agg_func_name]

        # skip adding control result if they are interaction result
        if not analysis_result.is_interaction_result:
            if Constants.CONTROL_RESULTS not in res_agg_func_name_result:
                res_agg_func_name_result[Constants.CONTROL_RESULTS] = {}

            # handle control if required
            if control_variation not in res_agg_func_name_result[Constants.CONTROL_RESULTS]:
                res_agg_func_name_result[Constants.CONTROL_RESULTS][
                    control_variation
                ] = analysis_result.control_result_dict

        # handle treatment variation
        if Constants.TREATMENT_RESULTS not in res_agg_func_name_result:
            res_agg_func_name_result[Constants.TREATMENT_RESULTS] = {}
        variation = analysis_result.experiment_groups_treatment_variation_name

        res_agg_func_name_result[Constants.TREATMENT_RESULTS][variation] = analysis_result.treatment_result_dict

    return transform_hierarchy_dict_to_dict(hierarchy_dict)


def transform_hierarchy_dict_to_dict(analysis_result_dict: Dict[str, Dict]) -> List[Dict[str, Any]]:
    """function to transform list of analysis result dict into a dictionary (json)
    for the experiment platform

    Arguments:
        analysis_result_dict {Dict} -- list of result from analyser

    Returns:
        analysis_result_json_dict_list [List[Dict]] -- Result of list of analysis results
        in json format required by the experiment platform
    """
    metric_results = []
    output_keys = {
        Constants.VARIATION,
        Constants.AVERAGE_TREATMENT_EFFECT,
        Constants.RELATIVE_AVERAGE_TREATMENT_EFFECT,
        Constants.P_VALUE,
        Constants.METRIC_VALUE,
        Constants.ABSOLUTE_CONFIDENCE_INTERVAL,
        Constants.RELATIVE_CONFIDENCE_INTERVAL,
        Constants.SAMPLE_SIZE,
        Constants.DATA_SIZE,
        Constants.SE,
        Constants.SEQUENTIAL_P_VALUE,
        Constants.SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL,
        Constants.SEQUENTIAL_RELATIVE_CONFIDENCE_INTERVAL,
        Constants.SEQUENTIAL_RESULT_TYPE,
    }
    for metric_name, metric_result in analysis_result_dict.items():
        output_experiment_group_results = []
        for experiment_group_name, experiment_group_result in metric_result.items():
            agg_func_results = []
            for agg_func_name, agg_func_result in experiment_group_result.items():
                treatment_results, control_results = [], []
                for _, variation_result in agg_func_result.get(Constants.CONTROL_RESULTS, {}).items():
                    control_results.append({k: v for k, v in variation_result.items() if k in output_keys})

                for _, variation_result in agg_func_result[Constants.TREATMENT_RESULTS].items():
                    treatment_results.append({k: v for k, v in variation_result.items() if k in output_keys})

                output_agg_func_result = {
                    Constants.AGG_FUNC_NAME: agg_func_name,
                    Constants.CONTROL_RESULTS: control_results,
                    Constants.TREATMENT_RESULTS: treatment_results,
                }
                agg_func_results.append(output_agg_func_result)
            output_experiment_group_result = {
                Constants.EXPERIMENT_GROUP_NAME: experiment_group_name,
                Constants.AGG_FUNC_RESULTS: agg_func_results,
            }
            output_experiment_group_results.append(output_experiment_group_result)
        output_metric_result = {
            Constants.METRIC_NAME: metric_name,
            Constants.EXPERIMENT_GROUP_RESULTS: output_experiment_group_results,
        }
        metric_results.append(output_metric_result)
    return metric_results


def filter_analysis_results(analysis_results: List[AnalysisResult]):
    message_collection = MessageCollection()
    validated_analysis_results: List[AnalysisResult] = []

    # Check that each analysis has valid output (Example: p-value is a number)
    for analysis_result in analysis_results:
        is_valid = analysis_result.validate()
        if is_valid:
            validated_analysis_results.append(analysis_result)

        message_collection.combine(analysis_result.message_collection)
    return validated_analysis_results, message_collection
