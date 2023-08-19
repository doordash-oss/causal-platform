"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from causal_platform.src.models.configuration_model.base_objects import Column, ExperimentGroup, Metric
from causal_platform.src.utils.error import InputDataError
from causal_platform.src.utils.logger import logger


# bootstrap sample
def get_bootstrap_sample(
    data: pd.DataFrame,
    size: int = None,
    replace: bool = True,
    cluster: Optional[Column] = None,
) -> pd.DataFrame:
    """function to get a dataframe of bootstrap samples, including cluster bootstrap
    and normal bootstrap

    Arguments:
        size {int} -- number of sample to return. If size is not None, then it's \
            number of cluster. Otherwise, it's number of data point.
        replace {bool} -- whether to bootstrap with replacement
    """
    if cluster:
        bootstrap_data = cluster_bootstrap_sample(data, cluster, size, replace)
    else:
        bootstrap_data = bootstrap_sample(data, size, replace)
    return bootstrap_data


def cluster_bootstrap_sample(
    data: pd.DataFrame, cluster: Column, size: int = None, replace: bool = True
) -> pd.DataFrame:
    """cluster bootstrap function to get bootstrap sample. This function will bootstrap by the
    the provided cluster column. This is being used by `get_bootstrap_sample`.
    If you want to get bootstrap function, you should use function `get_bootstrap_sample`

    Arguments:
        size {int} -- number of cluster to return
        cluster {Column} -- the column of cluster
        replace {bool} -- is bootstrap with replacement or not
    """
    if size is None:
        size = data[cluster.column_name].unique().shape[0]

    TEMP_COLUMN_NAME = "new_unit_id"

    unique_units = data[cluster.column_name].unique()

    random_selected_units = pd.DataFrame(
        np.random.choice(unique_units, size=size, replace=replace),
        columns=[cluster.column_name],
    )

    # create a new column as the new unit id for the cluster
    random_selected_units = random_selected_units.rename_axis(TEMP_COLUMN_NAME).reset_index()

    bootstrap_data = pd.merge(
        random_selected_units,
        data,
        left_on=cluster.column_name,
        right_on=cluster.column_name,
        how="left",
    )

    # use the new unit id to replace old unit id
    bootstrap_data.drop(cluster.column_name, axis=1, inplace=True)
    bootstrap_data.rename(columns={TEMP_COLUMN_NAME: cluster.column_name}, inplace=True)
    return bootstrap_data


def bootstrap_sample(data: pd.DataFrame, size: int = None, replace: bool = True) -> pd.DataFrame:
    """function to get bootstrap sample. This function bootstrap by row index.
    If you want to get bootstrap function, you should use function `get_bootstrap_sample`

    Arguments:
        data {pd.DataFrame}
        size {int} -- number of sample to return
        replace {bool} -- is bootstrap with replacement or not

    Returns:
        pd.DataFrame -- bootstrap sample data
    """
    if size is None:
        size = data.shape[0]

    random_selected_indexes = np.random.choice(data.shape[0], size=size, replace=replace)
    bootstrap_sample = data.copy().reset_index(drop=True).iloc[random_selected_indexes]
    bootstrap_sample = bootstrap_sample.reset_index(drop=True)
    return bootstrap_sample


def bootstrap_statistics_list(
    data: pd.DataFrame,
    size: int,
    replace: bool,
    statistics_calculate_func: Callable,
    iteration: int,
    cluster: Optional[Column] = None,
    statistics_calculate_func_kwargs: Optional[Dict] = None,
) -> List[float]:
    """function to bootstrap a list of statistics

    Arguments:
        data {pd.DataFrame} -- data
        size {int} -- number of sample to bootstrap at one iteration
        replace {bool} -- is bootstrap with replacement or not
        statistics_calculate_func {Callable} -- function used to calculate statistics
        iteration {int} -- how many iteration
        cluster {Optional[Column]} -- if it's provided, then perform cluster bootstrap,
                otherwise, normal bootstrap.

    Returns:
        List[float] -- a list of bootstraped statistics with length = iteration
    """
    bootstrap_statistics_list = []
    count_errors = 0
    exception = None
    for i in range(iteration):
        if cluster is None:
            samples = bootstrap_sample(data, size, replace)
        else:
            samples = cluster_bootstrap_sample(data=data, size=size, cluster=cluster, replace=replace)

        try:
            metric = statistics_calculate_func(samples, **statistics_calculate_func_kwargs)
            bootstrap_statistics_list.append(metric)
        except Exception as error:
            exception = error
            count_errors += 1
            if count_errors > 0.25 * iteration:
                print(exception)
                raise InputDataError(
                    "More than 25% of the statistics bootstrap failed. \
                    Please check your metric (i.e. zero value in denominator of ratio metric)"
                )

    if len(bootstrap_statistics_list) < iteration:
        logger.warning(
            "{} iterations are executed at bootstrapping statistics list, but only {} succeeded!".format(
                iteration, len(bootstrap_statistics_list)
            )
        )
        print(exception)

    return bootstrap_statistics_list


def bootstrap_standard_error(
    data: pd.DataFrame,
    size: int,
    replace: bool,
    statistics_calculate_func: Callable,
    iteration: int,
    cluster: Optional[Column] = None,
    statistics_calculate_func_kwargs: Optional[Dict] = None,
) -> float:
    """function to get bootstraped standard error.

    Arguments:
        data {pd.DataFrame} -- data
        size {int} -- number of sample to bootstrap at one iteration
        replace {bool} -- is bootstrap with replacement or not
        statistics_calculate_func {Callable} -- function used to calculate statistics
        iteration {int} -- how many iteration
        cluster {Optional[Column]} -- if it's provided, then perform cluster bootstrap,
                otherwise, normal bootstrap.

    Returns:
        float -- bootstraped standard error
    """
    bootstrapped_statistics_list = bootstrap_statistics_list(
        data=data,
        size=size,
        replace=replace,
        statistics_calculate_func=statistics_calculate_func,
        iteration=iteration,
        cluster=cluster,
        statistics_calculate_func_kwargs=statistics_calculate_func_kwargs,
    )
    standard_error = np.std(np.array(bootstrapped_statistics_list), ddof=1)
    return standard_error


def bootstrap_confidence_interval(
    data: pd.DataFrame,
    size: int,
    replace: bool,
    statistics_calculate_func: Callable,
    iteration: int,
    cluster: Optional[Column] = None,
    alpha: float = 0.05,
    statistics_calculate_func_kwargs: Optional[Dict] = None,
) -> Tuple[float, float]:
    """function to get bootstrap confidence interval

    Arguments:
        data {pd.DataFrame} -- data
        size {int} -- number of sample to bootstrap at one iteration
        replace {bool} -- is bootstrap with replacement or not
        statistics_calculate_func {Callable} -- function used to calculate statistics
        iteration {int} -- how many iteration
        cluster {Optional[Column]} -- if it's provided, then perform cluster bootstrap,
                otherwise, normal bootstrap.
        alpha {float} -- type 1 error

    Returns:
        Tuple[float, float] -- bootstrapped confidence interval
    """
    bootstrapped_statistics_list = bootstrap_statistics_list(
        data=data,
        size=size,
        replace=replace,
        statistics_calculate_func=statistics_calculate_func,
        iteration=iteration,
        cluster=cluster,
        statistics_calculate_func_kwargs=statistics_calculate_func_kwargs,
    )

    confidence_interval = get_confidence_interval_from_unsorted_list(bootstrapped_statistics_list, alpha)
    return confidence_interval


# point estimate
def calculate_point_estimate(
    data: pd.DataFrame,
    statistics_calculate_func: Callable,
    statistics_calculate_func_kwargs: Optional[Dict] = None,
):
    (
        point_estimate,
        treatment_value,
        control_value,
        treatment_size,
        control_size,
        treatment_data_size,
        control_data_size,
    ) = statistics_calculate_func(data=data, **statistics_calculate_func_kwargs)
    return (
        point_estimate,
        treatment_value,
        control_value,
        treatment_size,
        control_size,
        treatment_data_size,
        control_data_size,
    )


# calculate critical value
def calculate_critical_value_from_t_distribution(alpha: float, degree_of_freedom: float) -> Tuple[float, float]:
    critical_value = np.abs(stats.t.ppf(alpha / 2, degree_of_freedom))
    return -critical_value, critical_value


def calculate_critical_values_from_empirical_distribution(samples: List, alpha: float) -> Tuple[float, float]:
    critical_value_left = np.quantile(samples, alpha / 2)
    critical_value_right = np.quantile(samples, 1 - (alpha / 2))
    return critical_value_left, critical_value_right


# calculate confidence interval
def calculate_confidence_interval_from_standard_error(
    point_estimate: float, standard_error: float, critical_values: Tuple[float, float]
) -> Tuple[float, float]:
    return (
        point_estimate + critical_values[0] * standard_error,
        point_estimate + critical_values[1] * standard_error,
    )


# calculate t statistics
def calculate_t_statistics(point_estimate: float, standard_error: float, population_mean: float = 0) -> float:
    return (point_estimate - population_mean) / standard_error


def get_confidence_interval_from_unsorted_list(value_list: List[float], alpha: float) -> Tuple[float, float]:
    value_array = np.array(value_list)
    confidence_interval_left = np.quantile(value_array, alpha / 2)
    confidence_interval_right = np.quantile(value_array, 1 - (alpha / 2))
    return confidence_interval_left, confidence_interval_right


# calculate p value
def calculate_p_value_from_t_distribution(t_statistics: float, degree_of_freedom: float) -> float:
    return stats.t.sf(np.abs(t_statistics), degree_of_freedom) * 2


def calculate_p_value_from_distribution(sample: List[float], point_estimate: float) -> float:
    p_value = sum(w >= abs(point_estimate) or w <= -abs(point_estimate) for w in sample)
    return p_value


# statistics calculate functions
def calculate_quantile_statistics(
    data: pd.DataFrame,
    quantile: float,
    control_label: str,
    treatment_label: str,
    metric: Metric,
    experiment_group: ExperimentGroup,
    cluster: Optional[Column] = None,
):
    treatment = data[data[experiment_group.column_name] == treatment_label][metric.column_name]
    control = data[data[experiment_group.column_name] == control_label][metric.column_name]
    treatment_quantile_value = np.quantile(treatment, quantile)
    control_quantile_value = np.quantile(control, quantile)
    difference_of_quantile = treatment_quantile_value - control_quantile_value
    treatment_data_size = treatment.count()
    control_data_size = control.count()

    if not cluster:
        treatment_size = treatment_data_size
        control_size = control_data_size
    else:
        treatment_size = (
            data[data[experiment_group.column_name] == treatment_label][cluster.column_name].unique().shape[0]
        )
        control_size = data[data[experiment_group.column_name] == control_label][cluster.column_name].unique().shape[0]

    return (
        difference_of_quantile,
        treatment_quantile_value,
        control_quantile_value,
        treatment_size,
        control_size,
        treatment_data_size,
        control_data_size,
    )


def calculate_mean_statistics(
    data: pd.DataFrame,
    control_label: str,
    treatment_label: str,
    metric: Metric,
    experiment_group: ExperimentGroup,
):
    treatment = data[data[experiment_group.column_name] == treatment_label][metric.column_name]
    control = data[data[experiment_group.column_name] == control_label][metric.column_name]
    treatment_mean_value = np.mean(treatment)
    control_mean_value = np.mean(control)
    difference_of_mean = treatment_mean_value - control_mean_value
    treatment_data_size = treatment_size = treatment.count()
    control_data_size = control_size = control.count()
    return (
        difference_of_mean,
        treatment_mean_value,
        control_mean_value,
        treatment_size,
        control_size,
        treatment_data_size,
        control_data_size,
    )


def calculate_ratio_statistics(
    data: pd.DataFrame,
    control_label: str,
    treatment_label: str,
    experiment_group: ExperimentGroup,
    numerator_column: Column,
    denominator_column: Column,
):
    treatment = data[data[experiment_group.column_name] == treatment_label]
    control = data[data[experiment_group.column_name] == control_label]

    treatment_numerator = np.sum(treatment[numerator_column.column_name])
    control_numerator = np.sum(control[numerator_column.column_name])
    treatment_denominator = np.sum(treatment[denominator_column.column_name])
    control_denominator = np.sum(control[denominator_column.column_name])

    if treatment_denominator == 0:
        raise InputDataError(
            "Unable to calculate ratio metric for treatment group! Sum of {} at treatment group is 0!".format(
                denominator_column.column_name
            )
        )

    if control_denominator == 0:
        raise InputDataError(
            "Unable to calculate ratio metric for control group! Sum of {} at control group is 0!".format(
                denominator_column.column_name
            )
        )

    treatment_ratio_value = treatment_numerator / treatment_denominator
    control_ratio_value = control_numerator / control_denominator
    difference_of_ratio = treatment_ratio_value - control_ratio_value
    treatment_data_size = treatment_size = treatment[numerator_column.column_name].count()
    control_data_size = control_size = control[numerator_column.column_name].count()

    return (
        difference_of_ratio,
        treatment_ratio_value,
        control_ratio_value,
        treatment_size,
        control_size,
        treatment_data_size,
        control_data_size,
    )
