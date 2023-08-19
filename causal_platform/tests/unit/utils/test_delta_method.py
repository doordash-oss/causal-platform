"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import numpy as np
import pytest
from numpy.linalg import inv

from causal_platform.src.models.configuration_model.base_objects import (
    Column,
    ColumnType,
    Covariate,
    CovariateType,
    Metric,
    MetricAggregateFunc,
    MetricType,
)
from causal_platform.src.utils.delta_method import (
    calculate_adjusted_ratio_metric_covariate_covariance,
    calculate_ratio_covariate_coefficients,
    calculate_ratio_covariate_coefficients_noniterative,
    get_adjusted_ratio_metric_variance,
    get_delta_ratio_covariance,
)
from causal_platform.tests.data import data_generator


class TestDeltaMethod:
    @pytest.fixture
    def multi_cov_data(self):
        """
        DGP:
        num_rows = 100
        Numerator Variance Matrix = [[10, 5, 4], [5, 10, 6], [4, 6, 10]]
        Coefficient = [5, 8, 6]
        noise_sd = 10
        """
        df = data_generator.get_multi_ratio_cov_data()
        return df

    def test_get_delta_ratio_covariance(self):
        # when x/y and u/v are uncorrelated, the covariance should be zero
        x = np.array([1, 1, -2, -2])
        y = np.array([6, 6, 4, 4])
        u = np.array([2, -7, 2, -7])
        v = np.array([3, -1, 3, -1])
        assert get_delta_ratio_covariance(x, y, u, v) == 0

        # when denominators are trivial, the covariance should be equal to "nominal" covariance
        x = np.array([2, 3, 5, 7])
        y = np.array([1, 1, 1, 1])
        u = np.array([11, 13, 17, 19])
        v = np.array([1, 1, 1, 1])
        assert get_delta_ratio_covariance(x, y, u, v) == np.cov(x, u)[0, 1] / len(x)

        # when the denominator mean is zero, taylor expansion is no longer applicable and should return nan
        x = np.array([23, 29, 31, 37])
        y = np.array([1, 1, -1, -1])
        u = np.array([41, 43, 47, 53])
        v = np.array([1, 1, 1, 1])
        assert np.isnan(get_delta_ratio_covariance(x, y, u, v))

    def test_calculate_adjusted_ratio_metric_covariate_covariance(self, multi_cov_data):

        metric = Metric(
            "ratio_metric",
            MetricType.ratio,
            MetricAggregateFunc.mean,
            numerator_column=Column("metric_n", ColumnType.ratio_metric_component),
            denominator_column=Column("metric_d", ColumnType.ratio_metric_component),
        )
        cov1 = Covariate(
            "cov1",
            CovariateType.ratio,
            numerator_column=Column("cov1_n", ColumnType.ratio_covariate_component),
            denominator_column=Column("cov1_d", ColumnType.ratio_covariate_component),
        )
        cov2 = Covariate(
            "cov2",
            CovariateType.ratio,
            numerator_column=Column("cov2_n", ColumnType.ratio_covariate_component),
            denominator_column=Column("cov2_d", ColumnType.ratio_covariate_component),
        )

        # when there is no processed covariates, output the covariance
        result = calculate_adjusted_ratio_metric_covariate_covariance(multi_cov_data, metric, [], cov1)
        expected_result = (
            np.cov(
                multi_cov_data[metric.numerator_column.column_name],
                multi_cov_data[cov1.numerator_column.column_name],
            )[0, 1]
            / multi_cov_data.shape[0]
        )
        assert np.abs(result - expected_result) <= 1e-4

        # when there is a covariate, output the adjusted covariance
        cov1.coef = 7
        processed_covaraites = [cov1]
        result = calculate_adjusted_ratio_metric_covariate_covariance(
            multi_cov_data, metric, processed_covaraites, cov2
        )
        expected_result = (
            np.cov(
                multi_cov_data[metric.numerator_column.column_name]
                - cov1.coef * multi_cov_data[cov1.numerator_column.column_name],
                multi_cov_data[cov2.numerator_column.column_name],
            )[0, 1]
            / multi_cov_data.shape[0]
        )
        assert np.abs(result - expected_result) <= 1e-4

    def test_calculate_ratio_covaraite_coefficients(self, multi_cov_data):

        metric = Metric(
            "ratio_metric",
            MetricType.ratio,
            MetricAggregateFunc.mean,
            numerator_column=Column("metric_n", ColumnType.ratio_metric_component),
            denominator_column=Column("metric_d", ColumnType.ratio_metric_component),
        )
        cov1 = Covariate(
            "cov1",
            CovariateType.ratio,
            numerator_column=Column("cov1_n", ColumnType.ratio_covariate_component),
            denominator_column=Column("cov1_d", ColumnType.ratio_covariate_component),
        )
        cov2 = Covariate(
            "cov2",
            CovariateType.ratio,
            numerator_column=Column("cov2_n", ColumnType.ratio_covariate_component),
            denominator_column=Column("cov2_d", ColumnType.ratio_covariate_component),
        )

        processed_covariates = calculate_ratio_covariate_coefficients(multi_cov_data, metric, [cov1, cov2])
        beta1_expected = np.cov(
            multi_cov_data[metric.numerator_column.column_name],
            multi_cov_data[cov1.numerator_column.column_name],
        )[0, 1] / np.var(multi_cov_data[cov1.numerator_column.column_name], ddof=1)
        beta2_expected = np.cov(
            multi_cov_data[metric.numerator_column.column_name]
            - beta1_expected * multi_cov_data[cov1.numerator_column.column_name],
            multi_cov_data[cov2.numerator_column.column_name],
        )[0, 1] / np.var(multi_cov_data[cov2.numerator_column.column_name], ddof=1)
        assert np.abs(processed_covariates[0].coef - beta1_expected) <= 1e-4
        assert np.abs(processed_covariates[1].coef - beta2_expected) <= 1e-4

    def test_calculate_ratio_covaraite_coefficients_noniterative(self, multi_cov_data):

        metric = Metric(
            "ratio_metric",
            MetricType.ratio,
            MetricAggregateFunc.mean,
            numerator_column=Column("metric_n", ColumnType.ratio_metric_component),
            denominator_column=Column("metric_d", ColumnType.ratio_metric_component),
        )
        cov1 = Covariate(
            "cov1",
            CovariateType.ratio,
            numerator_column=Column("cov1_n", ColumnType.ratio_covariate_component),
            denominator_column=Column("cov1_d", ColumnType.ratio_covariate_component),
        )
        cov2 = Covariate(
            "cov2",
            CovariateType.ratio,
            numerator_column=Column("cov2_n", ColumnType.ratio_covariate_component),
            denominator_column=Column("cov2_d", ColumnType.ratio_covariate_component),
        )

        processed_covariates, _ = calculate_ratio_covariate_coefficients_noniterative(
            multi_cov_data, metric, [cov1, cov2]
        )
        beta_expected = np.matmul(
            inv(
                np.cov(
                    [
                        multi_cov_data[cov1.numerator_column.column_name],
                        multi_cov_data[cov2.numerator_column.column_name],
                    ]
                )
            ),
            [
                np.cov(
                    multi_cov_data[metric.numerator_column.column_name],
                    multi_cov_data[cov1.numerator_column.column_name],
                )[0][1],
                np.cov(
                    multi_cov_data[metric.numerator_column.column_name],
                    multi_cov_data[cov2.numerator_column.column_name],
                )[0][1],
            ],
        )
        beta1_expected = beta_expected[0]
        beta2_expected = beta_expected[1]
        assert np.abs(processed_covariates[0].coef - beta1_expected) <= 1e-4
        assert np.abs(processed_covariates[1].coef - beta2_expected) <= 1e-4

    def test_get_adjusted_ratio_metric_variance(self, multi_cov_data):
        dgp_coef = [5, 8, 6]
        metric = Metric(
            "ratio_metric",
            MetricType.ratio,
            MetricAggregateFunc.mean,
            numerator_column=Column("metric_n", ColumnType.ratio_metric_component),
            denominator_column=Column("metric_d", ColumnType.ratio_metric_component),
        )
        cov1 = Covariate(
            "cov1",
            CovariateType.ratio,
            numerator_column=Column("cov1_n", ColumnType.ratio_covariate_component),
            denominator_column=Column("cov1_d", ColumnType.ratio_covariate_component),
            coef=dgp_coef[0],
        )
        cov2 = Covariate(
            "cov2",
            CovariateType.ratio,
            numerator_column=Column("cov2_n", ColumnType.ratio_covariate_component),
            denominator_column=Column("cov2_d", ColumnType.ratio_covariate_component),
            coef=dgp_coef[1],
        )
        var1 = get_adjusted_ratio_metric_variance(multi_cov_data, metric, [cov1, cov2])
        expected_variance = np.var(
            multi_cov_data[metric.numerator_column.column_name]
            - cov1.coef * multi_cov_data[cov1.numerator_column.column_name]
            - cov2.coef * multi_cov_data[cov2.numerator_column.column_name],
            ddof=1,
        ) / len(multi_cov_data)

        cov3 = Covariate(
            "cov3",
            CovariateType.ratio,
            numerator_column=Column("cov3_n", ColumnType.ratio_covariate_component),
            denominator_column=Column("cov3_d", ColumnType.ratio_covariate_component),
            coef=dgp_coef[2],
        )
        var2 = get_adjusted_ratio_metric_variance(multi_cov_data, metric, [cov1, cov2, cov3])
        assert np.abs(var1 - expected_variance) <= 1e-4
        assert var2 <= var1
        assert var1 <= np.var(multi_cov_data[metric.numerator_column.column_name])
