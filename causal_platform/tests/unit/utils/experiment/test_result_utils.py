"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import pandas as pd
import pytest

from causal_platform.src.models.configuration_model.base_objects import (
    Cluster,
    ExperimentGroup,
    ExperimentVariation,
    Metric,
    MetricAggregateFunc,
    MetricType,
)
from causal_platform.src.utils.experiment.result_utils import (
    calculate_data_size,
    calculate_interaction_metric_stats,
    calculate_sample_size,
)


class TestResultUtils:
    @pytest.fixture
    def sample_metric(self) -> Metric:
        return Metric(
            column_name="c1",
            metric_type=MetricType.continuous,
            metric_aggregate_func=MetricAggregateFunc.mean,
            log_transform=False,
            remove_outlier=False,
            check_distribution=False,
        )

    @pytest.fixture
    def sample_metric_interaction(self) -> Metric:
        return Metric(
            column_name="column",
            metric_type=MetricType.continuous,
            metric_aggregate_func=MetricAggregateFunc.mean,
            log_transform=False,
            remove_outlier=False,
            check_distribution=False,
            clusters=[Cluster(column_name="cluster")],
        )

    @pytest.fixture
    def sample_metric_with_clusters(self) -> Metric:
        return Metric(
            column_name="c1",
            metric_type=MetricType.continuous,
            metric_aggregate_func=MetricAggregateFunc.mean,
            log_transform=False,
            remove_outlier=False,
            check_distribution=False,
            clusters=[Cluster(column_name="c2"), Cluster(column_name="c3")],
        )

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        return pd.DataFrame([[1, 2, 7], [2, 2, 8], [3, 2, 9]], columns=["c1", "c2", "c3"])

    @pytest.fixture
    def sample_data_interaction(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                [1, 1, "variation0", "variation1"],
                [2, 1, "variation0", "variation1"],
                [3, 1, "variation0", "variation1"],
                [4, 2, "variation0", "variation2"],
                [5, 2, "variation0", "variation2"],
                [6, 3, "variation0", "variation2"],
                [7, 3, "variation1", "variation1"],
                [8, 3, "variation1", "variation1"],
                [9, 3, "variation1", "variation1"],
            ],
            columns=["column", "cluster", "group1", "group2"],
        )

    def test_calculate_sample_size_with_no_cluster(self, sample_data, sample_metric):
        assert calculate_data_size(sample_data, sample_metric) == 3

    def test_calculate_sample_size_with_cluster(self, sample_data, sample_metric_with_clusters):
        assert calculate_sample_size(sample_data, sample_metric_with_clusters) == 1

    def test_calculate_data_size(self, sample_data, sample_metric):
        assert calculate_data_size(sample_data, sample_metric) == 3

    def test_calculate_interaction_value_and_sample_size(self, sample_data_interaction, sample_metric_interaction):
        (metric_value, sample_size, data_size) = calculate_interaction_metric_stats(
            sample_data_interaction,
            sample_metric_interaction,
            ExperimentGroup("group1"),
            ExperimentGroup("group2"),
            ExperimentVariation("variation0", 0.5),
            ExperimentVariation("variation1", 0.5),
            ExperimentVariation("variation2", 0.5),
        )
        assert metric_value == 3.0
        assert sample_size == 2
        assert data_size == 3
