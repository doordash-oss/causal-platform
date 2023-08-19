"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import pandas as pd

from causal_platform.src.models.configuration_model.base_objects import (
    Cluster,
    Column,
    ColumnType,
    ExperimentGroup,
    Metric,
    MetricAggregateFunc,
    MetricType,
)
from causal_platform.src.utils.config_utils import set_experiment_config
from causal_platform.src.utils.experiment.bootstrap import (
    bootstrap_confidence_interval,
    bootstrap_sample,
    bootstrap_standard_error,
    bootstrap_statistics_list,
    calculate_confidence_interval_from_standard_error,
    calculate_critical_value_from_t_distribution,
    calculate_critical_values_from_empirical_distribution,
    calculate_mean_statistics,
    calculate_point_estimate,
    calculate_quantile_statistics,
    calculate_ratio_statistics,
    cluster_bootstrap_sample,
)
from causal_platform.tests.unit.data import get_ratio_test_input, get_test_input


class TestBootstrap:
    def test_cluster_bootstrap_sample(self):
        data, config = get_test_input()
        cluster = Column("cluster", ColumnType.covariate)
        # without replacement
        bootstrap_data = cluster_bootstrap_sample(data, cluster, 2, False)
        assert bootstrap_data.shape[0] >= 2
        assert bootstrap_data[cluster.column_name].unique().shape[0] == 2
        # with replacement
        bootstrap_data = cluster_bootstrap_sample(data, cluster, 5, replace=True)
        assert bootstrap_data.shape[0] >= 5
        assert bootstrap_data[cluster.column_name].unique().shape[0] == 5

    def test_bootstrap_sample(self):
        data, config = get_test_input()
        # without replacement
        bootstrap_data = bootstrap_sample(data, 5, replace=False)
        assert bootstrap_data.shape[0] == 5
        # with replacement
        bootstrap_data = bootstrap_sample(data, 5, replace=True)
        assert bootstrap_data.shape[0] == 5

    def test_calculate_quantile_metric(self):
        data, config = get_test_input()
        metric = Metric(
            column_name="metric2",
            metric_type=MetricType.continuous,
            metric_aggregate_func=MetricAggregateFunc.mean,
            log_transform=False,
            remove_outlier=False,
            check_distribution=False,
        )
        experiment_group = ExperimentGroup("group")
        (
            difference_of_quantile,
            treatment_quantile_value,
            control_quantile_value,
            treatment_size,
            control_size,
            treatment_data_size,
            control_data_size,
        ) = calculate_quantile_statistics(
            data,
            quantile=0.9,
            control_label="control",
            treatment_label="treatment",
            metric=metric,
            experiment_group=experiment_group,
        )
        assert difference_of_quantile > 0
        assert treatment_size <= treatment_data_size
        assert control_size <= control_data_size

    def test_calculate_ratio_statistics(self):
        data, config_dict = get_ratio_test_input()
        config = set_experiment_config(config_dict)
        (
            ratio,
            treatment_ratio_value,
            control_ratio_value,
            treatment_size,
            control_size,
            treatment_data_size,
            control_data_size,
        ) = calculate_ratio_statistics(
            data=data,
            control_label="control",
            treatment_label="treatment",
            experiment_group=config.experiment_groups[0],
            numerator_column=config.metrics[0].numerator_column,
            denominator_column=config.metrics[0].denominator_column,
        )
        assert round(ratio, 2) == 1.86
        assert treatment_size == treatment_data_size
        assert control_size == control_data_size

    def test_bootstrap_statistics_list(self):
        data, config = get_test_input()
        metric = Metric(
            column_name="metric2",
            metric_type=MetricType.continuous,
            metric_aggregate_func=MetricAggregateFunc.mean,
            log_transform=False,
            remove_outlier=False,
            check_distribution=False,
        )
        experiment_group = ExperimentGroup("group")
        bootstrap_metric_list = bootstrap_statistics_list(
            data,
            100,
            True,
            calculate_quantile_statistics,
            100,
            statistics_calculate_func_kwargs={
                "quantile": 0.95,
                "control_label": "control",
                "treatment_label": "treatment",
                "metric": metric,
                "experiment_group": experiment_group,
            },
        )

        assert len(bootstrap_metric_list) == 100

    def test_bootstrap_standard_error(self):
        data, config = get_test_input()
        metric = Metric(
            column_name="metric2",
            metric_type=MetricType.continuous,
            metric_aggregate_func=MetricAggregateFunc.mean,
            log_transform=False,
            remove_outlier=False,
            check_distribution=False,
        )
        experiment_group = ExperimentGroup("group")
        se = bootstrap_standard_error(
            data,
            100,
            True,
            calculate_quantile_statistics,
            100,
            statistics_calculate_func_kwargs={
                "quantile": 0.95,
                "control_label": "control",
                "treatment_label": "treatment",
                "metric": metric,
                "experiment_group": experiment_group,
            },
        )
        assert isinstance(se, float)

    def test_bootstrap_confidence_interval(self):
        data, config = get_test_input()
        metric = Metric(
            column_name="metric2",
            metric_type=MetricType.continuous,
            metric_aggregate_func=MetricAggregateFunc.mean,
            log_transform=False,
            remove_outlier=False,
            check_distribution=False,
        )
        experiment_group = ExperimentGroup("group")
        confidence_interval = bootstrap_confidence_interval(
            data=data,
            size=100,
            replace=True,
            statistics_calculate_func=calculate_quantile_statistics,
            iteration=100,
            statistics_calculate_func_kwargs={
                "quantile": 0.95,
                "control_label": "control",
                "treatment_label": "treatment",
                "metric": metric,
                "experiment_group": experiment_group,
            },
        )
        assert len(confidence_interval) == 2

    def test_calculate_point_estimate(self):
        data = pd.DataFrame([["treatment", 100], ["control", 50]], columns=["group", "metric"])

        metric = Metric(
            column_name="metric",
            metric_type=MetricType.continuous,
            metric_aggregate_func=MetricAggregateFunc.mean,
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
            data=data,
            statistics_calculate_func=calculate_mean_statistics,
            statistics_calculate_func_kwargs={
                "control_label": "control",
                "treatment_label": "treatment",
                "metric": metric,
                "experiment_group": ExperimentGroup("group"),
            },
        )
        assert point_estimate == 50
        assert treatment_value == 100
        assert control_value == 50
        assert treatment_size == 1
        assert control_size == 1
        assert treatment_data_size == 1
        assert control_data_size == 1

    def test_calculate_quantile_statistics_with_cluster_provided(self):
        data = pd.DataFrame(
            [
                ["treatment", 100, 1],
                ["treatment", 100, 1],
                ["treatment", 100, 2],
                ["control", 50, 2],
                ["control", 50, 2],
                ["control", 50, 2],
            ],
            columns=["group", "metric", "cluster"],
        )

        metric = Metric(
            column_name="metric",
            metric_type=MetricType.continuous,
            metric_aggregate_func=MetricAggregateFunc.quantile,
            quantile=0.95,
        )

        (
            difference_of_quantile,
            treatment_value,
            control_value,
            treatment_size,
            control_size,
            treatment_data_size,
            control_data_size,
        ) = calculate_point_estimate(
            data=data,
            statistics_calculate_func=calculate_quantile_statistics,
            statistics_calculate_func_kwargs={
                "quantile": 0.95,
                "control_label": "control",
                "treatment_label": "treatment",
                "metric": metric,
                "experiment_group": ExperimentGroup("group"),
                "cluster": Cluster(column_name="cluster"),
            },
        )
        assert difference_of_quantile > 0
        assert treatment_size == 2
        assert control_size == 1
        assert treatment_data_size == 3
        assert control_data_size == 3

    def test_calculate_critical_value_from_t_distribution(self):
        t_critical_values = calculate_critical_value_from_t_distribution(0.05, 10)
        assert round(t_critical_values[1], 3) == 2.228

    def test_calculate_critical_value_from_distribution(self):
        sample = [i for i in range(10)]
        critical_values = calculate_critical_values_from_empirical_distribution(sample, 0.05)
        assert critical_values[1] == 8.775
        assert critical_values[0] == 0.225

    def test_calculate_confidence_interval_from_standard_error(self):
        (left_val, right_val) = calculate_confidence_interval_from_standard_error(
            point_estimate=12, standard_error=0.1, critical_values=(-2, 2)
        )
        assert left_val == 12 - 0.1 * 2
        assert right_val == 12 + 0.1 * 2

        # asymmetric critical value
        (left_val, right_val) = calculate_confidence_interval_from_standard_error(
            point_estimate=12, standard_error=0.1, critical_values=(-3, 2)
        )
        assert left_val == 12 - 0.1 * 3
        assert right_val == 12 + 0.1 * 2
