"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import copy
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.stats import norm

from causal_platform.src.models.analyser_model.fitters.fitter import Fitter
from causal_platform.src.models.configuration_model.base_objects import (
    Column,
    Covariate,
    ExperimentGroup,
    Metric,
    MetricAggregateFunc,
    MetricType,
    SequentialResultType,
)
from causal_platform.src.models.data_model import DataLoader
from causal_platform.src.models.message.enums import Source, Status
from causal_platform.src.models.message.message import Message, MessageCollection
from causal_platform.src.models.result_model.result import AnalysisResult, CovariateResult
from causal_platform.src.utils.dashab_timer import time_profiler
from causal_platform.src.utils.delta_method import (
    calculate_ratio_covariate_coefficients,
    calculate_ratio_covariate_coefficients_noniterative,
    get_adjusted_ratio_metric_variance,
    get_delta_ratio_variance,
)
from causal_platform.src.utils.error import InputConfigError
from causal_platform.src.utils.experiment.fitter_utils import (
    get_covariates_by_type,
    get_demean_column_name,
    get_resid_column_name,
    process_data_for_fitter,
)
from causal_platform.src.utils.experiment.group_sequential import GroupSequentialTest
from causal_platform.src.utils.experiment.result_utils import (
    calculate_basic_sample_metric_stats,
    get_variation_data,
    calculate_data_size,
)
from causal_platform.src.utils.experiment.stats_utils import (
    calculate_default_sequential_tau,
    fixed_compute_p_value_and_conf_int,
    get_treatment_effect_standard_error,
    seq_compute_p_value_and_conf_int,
)


class BasicFitter(Fitter):
    def __init__(
        self,
        data: Union[pd.DataFrame, DataLoader],
        metric: Metric,
        experiment_groups: List[ExperimentGroup],
        cluster: Optional[Column] = None,
        covariates: Optional[List[Covariate]] = None,
        interaction: bool = False,
        use_iterative_cv_method: bool = True,
        information_rates: Optional[List[float]] = None,
        target_sample_size: Optional[float] = None,
        current_sample_size: Optional[float] = None,
    ) -> None:
        self.message_collection = MessageCollection()
        self.metric = copy.deepcopy(metric)
        self.experiment_groups = copy.deepcopy(experiment_groups)
        self.cluster = copy.deepcopy(cluster)
        self.interaction = interaction
        self.covariates = [] if covariates is None else copy.deepcopy(covariates)
        self.processed_ratio_covariates = []
        if self.interaction and len(self.experiment_groups) > 2:
            raise InputConfigError(
                """causal-platform only support two way interaction of two experiment groups!\
                There are more than two experiment_group in the config."""
            )
        # when using low memory mode, each metric in analyzer is fit by reading from disk only once,
        # because processing also reads relevant columns to ram

        self.data = process_data_for_fitter(
            self.experiment_groups + self.covariates + [self.cluster] + [self.metric], data
        )
        self.use_iterative_cv_method = use_iterative_cv_method
        self.information_rates = information_rates
        self.target_sample_size = target_sample_size
        self.current_sample_size = current_sample_size
        self.group_sequential: Optional[GroupSequentialTest] = None
        self.sequential_result_type: Optional[SequentialResultType] = None
        self.covariates_var = None

    def ddof(self):
        return len(self.processed_ratio_covariates) + 1

    def fit(self, *args, **kwargs):
        """Fit variance reduction if covariates are provided."""
        if len(self.covariates) > 0:
            self._variance_reduction()

    def get_covariate_results(self) -> List[CovariateResult]:
        results = []

        # variance of epsilon i.e. sigma^2
        resid_var = get_adjusted_ratio_metric_variance(self.data, self.metric, self.processed_ratio_covariates) * len(
            self.data
        )
        # (X'X)-1
        inverse_matrix = inv(self.covariates_var * len(self.data) ** 2)
        # covariates' coefficients' variance (X'X)-1 * sigma^2
        beta_hat_variance = inverse_matrix.diagonal() * resid_var
        # ddof adjustment
        ddof_adjust = (len(self.data) - 1) / (len(self.data) - self.ddof())
        beta_hat_variance = beta_hat_variance * ddof_adjust

        for idx, cov in enumerate(self.processed_ratio_covariates):
            point_estimate = cov.coef
            se = beta_hat_variance[idx] ** 0.5
            p_value, conf_int_left, conf_int_right = fixed_compute_p_value_and_conf_int(
                point_estimate, se, self.metric.alpha
            )
            data_size = calculate_data_size(self.data, self.metric)
            results.append(
                CovariateResult(
                    covariate=cov,
                    metric=self.metric,
                    estimated_coefficient=point_estimate,
                    p_value=p_value,
                    confidence_interval_left=conf_int_left,
                    confidence_interval_right=conf_int_right,
                    data_size=data_size,
                    se=se,
                )
            )
        return results

    def get_analysis_results(self) -> List[AnalysisResult]:
        """Get analysis results object."""
        results = []
        if self.interaction and len(self.experiment_groups) == 2:
            exp_group_1st = self.experiment_groups[0]
            exp_group_2nd = self.experiment_groups[1]
            analysis_results = self._get_interaction_analysis_result(self.data, exp_group_1st, exp_group_2nd)
            results.extend(analysis_results)
        else:
            for experiment_group in self.experiment_groups:
                analysis_results_list = self._get_analysis_result(self.data, experiment_group)
                results.extend(analysis_results_list)

        return results

    def _get_analysis_result(self, data: pd.DataFrame, exp_group: ExperimentGroup) -> List[AnalysisResult]:
        results = []
        control_data = get_variation_data(data, exp_group, exp_group.control)

        unadjusted_control_value, _, _ = calculate_basic_sample_metric_stats(
            control_data, self.metric, use_processed_metric=False
        )

        control_value, control_size, control_data_size = calculate_basic_sample_metric_stats(
            control_data,
            self.metric,
            use_processed_metric=True,
            processed_covariates=self.processed_ratio_covariates,
        )

        for variation in exp_group.treatments:
            treatment_data = get_variation_data(data, exp_group, variation)

            unadjusted_treatment_value, _, _ = calculate_basic_sample_metric_stats(
                treatment_data, self.metric, use_processed_metric=False
            )

            (treatment_value, treatment_size, treatment_data_size,) = calculate_basic_sample_metric_stats(
                treatment_data,
                self.metric,
                use_processed_metric=True,
                processed_covariates=self.processed_ratio_covariates,
            )

            estimated_treatment_effect = treatment_value - control_value
            if self.current_sample_size is None:
                self.current_sample_size = treatment_size + control_size

            standard_error = self._compute_standard_error(treatment_data, control_data)
            # calculate fixed horizon results
            p_value, conf_int_left, conf_int_right = self._compute_p_value_and_conf_int(
                estimated_treatment_effect, standard_error, is_fixed_horizon=True
            )
            # calculate sequential testing results
            (
                sequential_p_value,
                sequential_conf_int_left,
                sequential_conf_int_right,
            ) = self._compute_p_value_and_conf_int(
                estimated_treatment_effect,
                standard_error,
                is_fixed_horizon=False,
                current_sample_size=self.current_sample_size,
                target_sample_size=self.target_sample_size,
                information_rates=self.information_rates,
            )

            analysis_result = AnalysisResult(
                metric=self.metric,
                estimated_treatment_effect=estimated_treatment_effect,
                p_value=p_value,
                confidence_interval_left=conf_int_left,
                confidence_interval_right=conf_int_right,
                experiment_group=exp_group,
                experiment_group_variation=variation,
                se=standard_error,
                metric_treatment_value=unadjusted_treatment_value,
                metric_control_value=unadjusted_control_value,
                metric_treatment_sample_size=treatment_size,
                metric_control_sample_size=control_size,
                metric_treatment_data_size=treatment_data_size,
                metric_control_data_size=control_data_size,
                is_sequential_result_valid=True,
                sequential_p_value=sequential_p_value,
                sequential_confidence_interval_left=sequential_conf_int_left,
                sequential_confidence_interval_right=sequential_conf_int_right,
                sequential_result_type=self.sequential_result_type,
            )
            results.append(analysis_result)
        return results

    def _get_interaction_analysis_result(
        self,
        data: pd.DataFrame,
        exp_group_1st: ExperimentGroup,
        exp_group_2nd: ExperimentGroup,
    ) -> List[AnalysisResult]:
        results = []
        # calculate result for first exp_group
        results_1st = self._get_analysis_result(
            get_variation_data(data, exp_group_2nd, exp_group_2nd.control),
            exp_group_1st,
        )
        results.extend(results_1st)
        # calculate result for second exp_group
        results_2nd = self._get_analysis_result(
            get_variation_data(data, exp_group_1st, exp_group_1st.control),
            exp_group_2nd,
        )
        results.extend(results_2nd)

        # calculate interaction
        for treatment_1st in exp_group_1st.treatments:
            partial_data = get_variation_data(data, exp_group_1st, treatment_1st)
            results_2nd_under_1st_trt = self._get_analysis_result(partial_data, exp_group_2nd)
            for treatment_2nd, result_2nd_under_1st_trt, result_2nd_under_1st_ctl in zip(
                exp_group_2nd.treatments, results_2nd_under_1st_trt, results_2nd
            ):
                interaction_effect = (
                    result_2nd_under_1st_trt.estimated_treatment_effect
                    - result_2nd_under_1st_ctl.estimated_treatment_effect
                )
                se = np.sqrt(result_2nd_under_1st_trt.se**2 + result_2nd_under_1st_ctl.se**2)
                p_val, conf_int_left, conf_int_right = self._compute_p_value_and_conf_int(
                    interaction_effect, se, is_fixed_horizon=True
                )
                (
                    sequential_p_value,
                    sequential_conf_int_left,
                    sequential_conf_int_right,
                ) = self._compute_p_value_and_conf_int(interaction_effect, se, is_fixed_horizon=False)

                interaction_result = AnalysisResult(
                    metric=result_2nd_under_1st_trt.metric,
                    estimated_treatment_effect=interaction_effect,
                    p_value=p_val,
                    confidence_interval_left=conf_int_left,
                    confidence_interval_right=conf_int_right,
                    experiment_group=(exp_group_1st, exp_group_2nd),
                    experiment_group_variation=(treatment_1st, treatment_2nd),
                    se=se,
                    metric_treatment_value=result_2nd_under_1st_trt.estimated_treatment_effect,
                    metric_control_value=result_2nd_under_1st_ctl.estimated_treatment_effect,
                    metric_treatment_sample_size=result_2nd_under_1st_trt.metric_treatment_sample_size,
                    metric_control_sample_size=result_2nd_under_1st_ctl.metric_control_sample_size,
                    metric_treatment_data_size=result_2nd_under_1st_trt.metric_treatment_data_size,
                    metric_control_data_size=result_2nd_under_1st_ctl.metric_control_data_size,
                    is_sequential_result_valid=True,
                    sequential_p_value=sequential_p_value,
                    sequential_confidence_interval_left=sequential_conf_int_left,
                    sequential_confidence_interval_right=sequential_conf_int_right,
                    is_interaction_result=True,
                )
                results.append(interaction_result)
        return results

    def _compute_standard_error(self, treatment_data: pd.DataFrame, control_data: pd.DataFrame) -> float:
        # calculate standard error
        var_treatment = self.compute_sample_statistics_variance(treatment_data)
        var_control = self.compute_sample_statistics_variance(control_data)
        return get_treatment_effect_standard_error(var_treatment, var_control)

    def compute_sample_statistics_variance(self, data: pd.DataFrame):
        """Calculate variance of sample by metric type."""
        var = None
        if (
            self.metric.metric_type == MetricType.continuous
            and self.metric.metric_aggregate_func == MetricAggregateFunc.mean
        ):
            var = self._compute_mean_variance(data)
        elif self.metric.metric_type == MetricType.proportional:
            var = self._compute_mean_variance(data)
        elif self.metric.metric_type == MetricType.ratio:
            var = get_adjusted_ratio_metric_variance(data, self.metric, self.processed_ratio_covariates)
        elif (
            self.metric.metric_type == MetricType.continuous
            and self.metric.metric_aggregate_func == MetricAggregateFunc.quantile
        ):
            var = self._compute_quantile_variance(data)
        return var

    @time_profiler(process_name="sequential-test")
    def _compute_p_value_and_conf_int(
        self,
        point_estimate: float,
        standard_error: float,
        tau: Optional[float] = None,
        is_fixed_horizon: bool = True,
        information_rates: Optional[List[float]] = None,
        target_sample_size: Optional[int] = None,
        current_sample_size: Optional[int] = None,
    ):
        if is_fixed_horizon:
            return fixed_compute_p_value_and_conf_int(point_estimate, standard_error, self.metric.alpha)
        elif all([information_rates, target_sample_size, current_sample_size]):
            self.sequential_result_type = SequentialResultType.group.value
            return self.group_sequential.get_group_sequential_result(
                point_estimate=point_estimate,
                standard_error=standard_error,
                information_rates=information_rates,
                target_sample_size=target_sample_size,
                current_sample_size=current_sample_size,
                alpha=self.metric.alpha,
            )
        else:
            self.sequential_result_type = SequentialResultType.always_valid.value
            if tau is None:
                tau = calculate_default_sequential_tau(standard_error)
            return seq_compute_p_value_and_conf_int(point_estimate, standard_error, tau, self.metric.alpha)

    def _compute_mean_variance(self, data: pd.DataFrame) -> float:
        if self.cluster is not None:
            d = data.groupby(self.cluster.column_name).size().to_numpy()
            n = data.groupby(self.cluster.column_name)[self.metric.processed_column_name].sum().to_numpy()
            variance = get_delta_ratio_variance(n, d)
        else:
            n = data.shape[0]

            variance = np.std(data[self.metric.processed_column_name], ddof=1) ** 2 / n
        return variance

    def _compute_quantile_variance(self, data: pd.DataFrame) -> float:
        """paper: https://arxiv.org/pdf/1803.06336.pdf."""
        quantile = self.metric.quantile
        metric_series = data[self.metric.processed_column_name]
        n = metric_series.count()
        t = norm.ppf(1 - self.metric.alpha / 2)

        quantile_radius = t * np.sqrt(quantile * (1 - quantile) / n)
        quantile_ci = [
            max(quantile - quantile_radius, 0),
            quantile,
            min(quantile + quantile_radius, 1),
        ]
        quantile_value_ci = np.nanquantile(metric_series, quantile_ci, method="lower")
        if self.cluster:
            # if there is cluster, correct the SE with ratio delta method
            correction = self._compute_quantile_cluster_correction(data)
            quantile_value_ci = (quantile_value_ci - quantile_value_ci[1]) * correction + quantile_value_ci[1]
        se = (quantile_value_ci[2] - quantile_value_ci[0]) / (2 * t)
        return se**2

    def _compute_quantile_cluster_correction(self, data: pd.DataFrame) -> float:
        # convert the dataframe to numpy array in shape (count_cluster, cluster_size)
        # 1. Each cluster has different size, so padding NAN to make every cluster equal size
        size_by_cluster = data.groupby(self.cluster.column_name).size()
        max_size = size_by_cluster.max()
        pad_shape = (max_size - size_by_cluster).to_frame("pad_rows").reset_index()
        # get rows to add as a series of lists
        rows_to_pad = pad_shape.apply(
            lambda x: [[x[self.cluster.column_name], np.nan]] * int(x["pad_rows"]),
            axis=1,
        )
        pad_df = pd.DataFrame(
            [row for item in rows_to_pad.values.tolist() for row in item],
            columns=[self.cluster.column_name, self.metric.processed_column_name],
        )
        data = pd.concat([data, pad_df], ignore_index=True)
        # 2. convert the dataframe into numpy array in shape (num_cluster, cluster_size)
        array = np.stack(
            [x.to_numpy() for _, x in data.groupby(self.cluster.column_name)[self.metric.processed_column_name]]
        )

        # 3. calculate quantile and count for each cluster
        quantile = self.metric.quantile
        quantile_value = np.nanquantile(array, quantile, method="lower")
        values = np.sum(np.less_equal(array, quantile_value), axis=1)
        counts = np.sum(~np.isnan(array), axis=1)
        n = np.sum(counts)

        # 4. calculate correction factor
        sigma = np.sqrt(get_delta_ratio_variance(values, counts)) * np.sqrt(n)
        correction = sigma / np.sqrt(quantile * (1 - quantile))
        return correction

    def _variance_reduction(self):
        if self.metric.metric_aggregate_func == MetricAggregateFunc.quantile:
            self.message_collection.add_metric_message(
                metric=self.metric.column_name,
                message=Message(
                    source=Source.analysis,
                    title="Basic Fitter",
                    description=f"""{self.metric.column_name} is using quantile aggregation,\
                causal-platform can't apply variance reduction on quantile aggregation,\
                continuing the analysis without applying covariate!""",
                    status=Status.warn,
                ),
            )
            return
        (
            categorical_covariates,
            numerical_covariates,
            ratio_covariates,
        ) = get_covariates_by_type(self.covariates)

        categorical_covariate_names = [cat_cov.column_name for cat_cov in categorical_covariates]
        numerical_covariate_names = [num_cov.column_name for num_cov in numerical_covariates]

        if self.metric.metric_type == MetricType.ratio:
            if self.use_iterative_cv_method:
                self.processed_ratio_covariates = calculate_ratio_covariate_coefficients(
                    data=self.data, metric=self.metric, covariates=ratio_covariates
                )
            else:
                (
                    self.processed_ratio_covariates,
                    self.covariates_var,
                ) = calculate_ratio_covariate_coefficients_noniterative(
                    data=self.data, metric=self.metric, covariates=ratio_covariates
                )
            unprocessed_covaraites = set(cov.column_name for cov in ratio_covariates) - set(
                cov.column_name for cov in self.processed_ratio_covariates
            )
            if len(unprocessed_covaraites) != 0:
                self.message_collection.add_metric_message(
                    metric=self.metric.column_name,
                    message=Message(
                        source=Source.analysis,
                        title="Basic Fitter",
                        description=f"""Failed to process covariates: {unprocessed_covaraites} for metric {self.metric.column_name}""",
                        status=Status.warn,
                    ),
                )

        else:  # non-ratio case
            # initialize demeaned numerical covariates and metric
            demeaned_numerical_covariates = (
                self.data[numerical_covariate_names] if len(numerical_covariate_names) > 0 else None
            )  # df of the extracted numerical

            # not ratio, just extract values
            processed_metric_name = self.metric.column_name
            processed_metric_value = self.data[processed_metric_name]

            # process categorical covariates
            if len(categorical_covariates) > 0:
                for fe in categorical_covariate_names:
                    data_grouped_by_fe = self.data.groupby(fe)
                    processed_metric_value = processed_metric_value - data_grouped_by_fe[
                        processed_metric_name
                    ].transform("mean")
                    processed_metric_name = get_demean_column_name(processed_metric_name)
                    if demeaned_numerical_covariates is not None:
                        demeaned_numerical_covariates = demeaned_numerical_covariates - data_grouped_by_fe[
                            numerical_covariate_names
                        ].transform("mean")

            # process numerical covariates
            if demeaned_numerical_covariates is not None:
                feature_shape = demeaned_numerical_covariates.shape
                # append the intercept column
                X = np.ones((feature_shape[0], feature_shape[1] + 1))
                X[:, :-1] = demeaned_numerical_covariates.to_numpy()
                Y = processed_metric_value.to_numpy()
                a = np.linalg.pinv(np.dot(np.transpose(X), X))
                b = np.dot(np.transpose(X), Y)
                beta = np.dot(a, b)
                processed_metric_value = Y - np.dot(X, beta)
                processed_metric_name = get_resid_column_name(processed_metric_name)

            # set processed metric name and value
            self.metric.set_processed_column_name(processed_metric_name)
            self.data[processed_metric_name] = processed_metric_value

        # drop covariate columns to save memory in analysis part
        self.data.drop(categorical_covariate_names + numerical_covariate_names, axis=1, inplace=True)
