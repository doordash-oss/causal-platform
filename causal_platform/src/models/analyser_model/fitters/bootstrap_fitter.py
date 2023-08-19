"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import copy
from typing import Callable, Dict, List, Optional

import pandas as pd

from causal_platform.src.models.analyser_model.fitters.fitter import Fitter
from causal_platform.src.models.configuration_model.base_objects import Column, ExperimentGroup, Metric
from causal_platform.src.models.message.message import MessageCollection
from causal_platform.src.models.result_model.result import AnalysisResult
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.error import InputConfigError, InputDataError
from causal_platform.src.utils.experiment.bootstrap import (
    bootstrap_standard_error,
    calculate_confidence_interval_from_standard_error,
    calculate_critical_value_from_t_distribution,
    calculate_critical_values_from_empirical_distribution,
    calculate_p_value_from_distribution,
    calculate_p_value_from_t_distribution,
    calculate_point_estimate,
    calculate_quantile_statistics,
    calculate_ratio_statistics,
    calculate_t_statistics,
    get_bootstrap_sample,
)
from causal_platform.src.utils.experiment.fitter_utils import process_data_for_fitter
from causal_platform.src.utils.logger import logger


class BootstrapFitter(Fitter):
    def __init__(
        self,
        data: pd.DataFrame,
        metric: Metric,
        experiment_groups: List[ExperimentGroup],
        statistics_calculate_func: Callable,
        statistics_calculate_func_kwargs: Dict,
        cluster: Optional[Column] = None,
        iteration: Optional[int] = 400,
        bootstrap_size: Optional[int] = None,
        replace: bool = True,
        method: str = Constants.BOOTSTRAP_SE,
    ):
        """
        Arguments:
            statistics_calculate_func: function that calculate the difference between treatment and control.
                The function needs to return <diff of treatment and control>, <treatment value>, <control value>,
                <treatment size>, <control size>
            statistics_calculate_func_kwargs: kwargs for statistics_calculate_func.
        """
        self.message_collection = MessageCollection()
        self.metric = copy.deepcopy(metric)
        self.experiment_groups = copy.deepcopy(experiment_groups)
        self.cluster = copy.deepcopy(cluster)
        self.iteration = iteration if iteration else 400
        if bootstrap_size is None:
            if cluster is None:
                self.bootstrap_size = data.shape[0]
            else:
                self.bootstrap_size = data[cluster.column_name].unique().shape[0]
        else:
            self.bootstrap_size = bootstrap_size
        self.statistics_calculate_func = statistics_calculate_func
        self.replace = replace
        self.method = method
        self.statistics_calculate_func_kwargs = statistics_calculate_func_kwargs
        self.analysis_results = []

        self.data = process_data_for_fitter([self.metric] + self.experiment_groups + [self.cluster], data)

    def fit(self):
        if self.cluster:
            degree_of_freedom = self.data[self.cluster.column_name].unique().shape[0] - 1
        else:
            degree_of_freedom = self.data.shape[0] - 1

        for experiment_group in self.experiment_groups:
            control_label = experiment_group.control.variation_name
            self.statistics_calculate_func_kwargs[
                Constants.STATISTICS_CALCULATE_FUNC_EXPERIMENT_GROUP
            ] = experiment_group
            self.statistics_calculate_func_kwargs[Constants.STATISTICS_CALCULATE_FUNC_CONTROL_LABEL] = control_label
            for treatment in experiment_group.treatments:
                treatment_label = treatment.variation_name
                self.statistics_calculate_func_kwargs[
                    Constants.STATISTICS_CALCULATE_FUNC_TREATMENT_LABEL
                ] = treatment_label
                if self.method == Constants.BOOTSTRAP_SE:
                    (
                        point_estimate,
                        p_value,
                        confidence_interval,
                        se,
                        treatment_value,
                        control_value,
                        treatment_size,
                        control_size,
                        treatment_data_size,
                        control_data_size,
                    ) = self.bootstrap_se(
                        bootstrap_size=self.bootstrap_size,
                        replace=self.replace,
                        statistics_calculate_func=self.statistics_calculate_func,
                        iteration=self.iteration,
                        degree_of_freedom=degree_of_freedom,
                        cluster=self.cluster,
                        statistics_calculate_func_kwargs=self.statistics_calculate_func_kwargs,
                    )
                elif self.method == Constants.BOOTSTRAP_T:
                    (
                        point_estimate,
                        p_value,
                        confidence_interval,
                        se,
                        treatment_value,
                        control_value,
                        treatment_size,
                        control_size,
                        treatment_data_size,
                        control_data_size,
                    ) = self.bootstrap_t(
                        bootstrap_size=self.bootstrap_size,
                        replace=self.replace,
                        statistics_calculate_func=self.statistics_calculate_func,
                        iteration=self.iteration,
                        cluster=self.cluster,
                        statistics_calculate_func_kwargs=self.statistics_calculate_func_kwargs,
                    )
                else:
                    raise InputConfigError(
                        """Unknown method '{}'! Only 'bootstrap_t' and 'bootstrap_se'
                            are supported!
                        """.format(
                            self.method
                        )
                    )

                result = AnalysisResult(
                    metric=self.metric,
                    estimated_treatment_effect=point_estimate,
                    p_value=p_value,
                    confidence_interval_left=confidence_interval[0],
                    confidence_interval_right=confidence_interval[1],
                    experiment_group=experiment_group,
                    experiment_group_variation=treatment,
                    se=se,
                    metric_treatment_value=treatment_value,
                    metric_control_value=control_value,
                    metric_treatment_sample_size=treatment_size,
                    metric_control_sample_size=control_size,
                    metric_treatment_data_size=treatment_data_size,
                    metric_control_data_size=control_data_size,
                    is_sequential_result_valid=False,
                )

                self.analysis_results.append(result)

    def get_analysis_results(self) -> List[AnalysisResult]:
        return self.analysis_results

    def bootstrap_se(
        self,
        bootstrap_size: int,
        replace: bool,
        statistics_calculate_func: Callable,
        iteration: int,
        degree_of_freedom: int,
        cluster: Optional[Column] = None,
        statistics_calculate_func_kwargs: Dict = None,
    ):

        standard_error = bootstrap_standard_error(
            data=self.data,
            size=bootstrap_size,
            replace=replace,
            statistics_calculate_func=statistics_calculate_func,
            iteration=iteration,
            cluster=cluster,
            statistics_calculate_func_kwargs=statistics_calculate_func_kwargs,
        )
        (
            point_estimate,
            treatment_value,
            control_value,
            treatment_size,
            control_size,
            treatment_data_size,
            control_data_size,
        ) = calculate_point_estimate(
            data=self.data,
            statistics_calculate_func=statistics_calculate_func,
            statistics_calculate_func_kwargs=statistics_calculate_func_kwargs,
        )

        critical_value = calculate_critical_value_from_t_distribution(self.metric.alpha, degree_of_freedom)
        confidence_interval = calculate_confidence_interval_from_standard_error(
            point_estimate, standard_error, critical_value
        )
        t_statistics = calculate_t_statistics(point_estimate, standard_error)
        p_value = calculate_p_value_from_t_distribution(t_statistics, degree_of_freedom)

        return (
            point_estimate,
            p_value,
            confidence_interval,
            standard_error,
            treatment_value,
            control_value,
            treatment_size,
            control_size,
            treatment_data_size,
            control_data_size,
        )

    def bootstrap_t(
        self,
        bootstrap_size: int,
        replace: bool,
        statistics_calculate_func: Callable,
        iteration: int,
        cluster: Optional[Column] = None,
        statistics_calculate_func_kwargs: Optional[Dict] = None,
    ):

        (
            beta_0,
            treatment_value,
            control_value,
            treatment_size,
            control_size,
            treatment_data_size,
            control_data_size,
        ) = calculate_point_estimate(
            data=self.data,
            statistics_calculate_func=statistics_calculate_func,
            statistics_calculate_func_kwargs=statistics_calculate_func_kwargs,
        )

        # TODO(caixia): this is too slow now. will need to use other method
        standard_error_0 = bootstrap_standard_error(
            self.data,
            bootstrap_size,
            replace,
            statistics_calculate_func,
            iteration,
            cluster,
            statistics_calculate_func_kwargs,
        )

        wald_statistics = []
        exception = None
        count_errors = 0
        for i in range(iteration):
            try:
                bootstraped_sample = get_bootstrap_sample(
                    self.data, size=bootstrap_size, replace=replace, cluster=cluster
                )

                beta, _, _, _, _, _, _ = calculate_point_estimate(
                    data=bootstraped_sample,
                    statistics_calculate_func=statistics_calculate_func,
                    statistics_calculate_func_kwargs=statistics_calculate_func_kwargs,
                )
                # TODO(caixia): use standard_error_0 as the standard error of bootstraped sample
                # for now to avoid too much computation.
                t = calculate_t_statistics(beta, standard_error_0, beta_0)
                wald_statistics.append(t)
            except Exception as error:
                exception = error
                count_errors += 1
                if count_errors > 0.25 * iteration:
                    print(exception)
                    raise InputDataError(
                        "More than 25% of the wald statistics bootstrap failed. \
                        Please check your metric (i.e. zero value in denominator of ratio metric)"
                    )

        if len(wald_statistics) < iteration:
            logger.warning(
                "{} bootstrap-t iterations is executed, but only {} succeeded!".format(iteration, len(wald_statistics))
            )
            print(exception)

        critical_values = calculate_critical_values_from_empirical_distribution(wald_statistics, self.metric.alpha)

        confidence_interval = calculate_confidence_interval_from_standard_error(
            beta_0, standard_error_0, critical_values
        )

        p_value = calculate_p_value_from_distribution(wald_statistics, beta_0)

        return (
            beta_0,
            p_value,
            confidence_interval,
            standard_error_0,
            treatment_value,
            control_value,
            treatment_size,
            control_size,
            treatment_data_size,
            control_data_size,
        )


class QuantileBootstrapFitter(BootstrapFitter):
    def __init__(
        self,
        data: pd.DataFrame,
        metric: Metric,
        experiment_groups: List[ExperimentGroup],
        cluster: Optional[Column] = None,
        iteration: Optional[int] = 400,
        bootstrap_size: Optional[int] = None,
        replace: bool = True,
        method: str = Constants.BOOTSTRAP_SE,
    ):
        super().__init__(
            data=data,
            metric=metric,
            experiment_groups=experiment_groups,
            cluster=cluster,
            iteration=iteration,
            bootstrap_size=bootstrap_size,
            replace=replace,
            method=method,
            statistics_calculate_func=calculate_quantile_statistics,
            statistics_calculate_func_kwargs={
                Constants.STATISTICS_CALCULATE_FUNC_QUANTILE: metric.quantile,
                Constants.STATISTICS_CALCULATE_FUNC_METRIC: metric,
                Constants.STATISTICS_CALCULATE_FUNC_CLUSTER: cluster,
            },
        )


class RatioBootstrapFitter(BootstrapFitter):
    def __init__(
        self,
        data: pd.DataFrame,
        metric: Metric,
        experiment_groups: List[ExperimentGroup],
        cluster: Optional[Column] = None,
        iteration: Optional[int] = 400,
        bootstrap_size: Optional[int] = None,
        replace: bool = True,
        method: str = Constants.BOOTSTRAP_SE,
    ):
        super().__init__(
            data=data,
            metric=metric,
            experiment_groups=experiment_groups,
            cluster=cluster,
            iteration=iteration,
            bootstrap_size=bootstrap_size,
            replace=replace,
            method=method,
            statistics_calculate_func=calculate_ratio_statistics,
            statistics_calculate_func_kwargs={
                Constants.STATISTICS_CALCULATE_FUNC_NUMERATOR_COLUMN: metric.numerator_column,
                Constants.STATISTICS_CALCULATE_FUNC_DENOMINATOR_COLUMN: metric.denominator_column,
            },
        )
