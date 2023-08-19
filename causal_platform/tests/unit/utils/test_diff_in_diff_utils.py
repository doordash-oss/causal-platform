"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import numpy as np
import pandas as pd
import pytest

from causal_platform.src.utils.config_utils import set_experiment_config
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.diff_in_diff.calculation import (
    calculate_correlation,
    calculate_euclidean_distance,
    calculate_standardize_metric,
    calculate_weighted_sum,
    standardize_and_calculate_weighted_sum,
)
from causal_platform.src.utils.diff_in_diff.plotting import (
    plot_matching_parallel_lines,
    prep_matching_plot_data,
)
from causal_platform.src.utils.diff_in_diff.prep_data import (
    aggregate_metric_by_date,
    get_aggregate_metric_in_unit_ids,
    get_data_between_start_end_date,
    get_data_filtered_by_unit_ids,
    get_unit_candidates,
    prep_data_for_diff_in_diff,
)
from causal_platform.tests.unit.data import get_diff_in_diff_input, get_real_diff_in_diff_input


class TestDiDUtils:
    @pytest.fixture
    def sample_data(self):
        data, config = get_diff_in_diff_input()
        return data

    @pytest.fixture
    def config(self):
        data, config = get_diff_in_diff_input()
        return config

    def test_get_region_candidates(self, sample_data):
        candidates = get_unit_candidates(sample_data, "market", [1], [2])
        assert candidates == [3]

    def test_get_data_filtered_by_unit_ids(self, sample_data):
        filtered_data = get_data_filtered_by_unit_ids(sample_data, "market", [1])
        assert filtered_data.shape[0] == 12

    def test_get_data_between_start_end_date(self, sample_data):
        filtered_data = get_data_between_start_end_date(
            sample_data, "date", pd.Timestamp("2019-01-01"), pd.Timestamp("2019-01-03")
        )
        assert filtered_data.shape[0] == 9

    def test_aggregate_metric_by_date(self, sample_data):
        aggregate_metric = aggregate_metric_by_date(sample_data, "date", "applicant", np.mean)
        assert type(aggregate_metric) == pd.Series
        assert aggregate_metric.shape[0] == 12

    def test_aggregate_metric_in_region_ids(self, sample_data):
        aggregate_metric = get_aggregate_metric_in_unit_ids(sample_data, "date", "market", [1], "applicant", np.mean)
        assert type(aggregate_metric) == pd.Series
        assert aggregate_metric.shape[0] == 12

    # calculation utils
    def test_calculate_correlation(self, sample_data):
        assert round(calculate_correlation(sample_data.applicant, sample_data.cvr), 2) == -0.21

    def test_calculate_euclidean_distance(self, sample_data):
        series1 = sample_data[sample_data["market"] == 1]["applicant"].to_numpy()
        series2 = sample_data[sample_data["market"] == 2]["applicant"].to_numpy()
        distance = calculate_euclidean_distance(series1, series2)
        assert round(distance, 2) == 289.31

    def test_calculate_weighted_sum(self, sample_data):
        weighted_sum = calculate_weighted_sum(sample_data[["applicant", "cvr"]].to_numpy(), np.array([0.5, 0.5]))
        assert weighted_sum.shape == (36,)

    def test_standardize_metric(self, sample_data):
        standardize_metric = calculate_standardize_metric(sample_data["applicant"].to_numpy())
        assert standardize_metric.shape == (36,)
        assert standardize_metric.max() == 1
        assert standardize_metric.min() == 0

    def test_prep_data_for_diff_in_diff(self):
        data, config_dict = get_real_diff_in_diff_input()
        config = set_experiment_config(config_dict)
        diff_in_diff_data = prep_data_for_diff_in_diff(
            data,
            config.treatment_unit_ids,
            [4, 10, 32],
            config.experiment_randomize_units[0].column_name,
            config.date.column_name,
            config.experiment_start_date,
        )
        assert diff_in_diff_data[config.date.column_name].min() == config.matching_start_date
        assert diff_in_diff_data[config.date.column_name].max() == config.experiment_end_date
        assert diff_in_diff_data.loc[0, "treatment"] == 1
        assert diff_in_diff_data.loc[0, "time"] == 0

    def test_prep_matching_plot_data(self):
        data, config_dict = get_real_diff_in_diff_input()
        config = set_experiment_config(config_dict)
        data[Constants.WEIGHTED_SUM_COLUMN_NAME] = standardize_and_calculate_weighted_sum(
            data[config.matching_columns].to_numpy(), config
        )
        treatment_series, control_series = prep_matching_plot_data(data, config, control_unit_ids=[2, 7, 8, 9])

        assert treatment_series.index.name == config.date.column_name
        assert control_series.index.name == config.date.column_name
        assert treatment_series.shape[0] == 6
        assert control_series.shape[0] == 6


def test_plot_matching_parallel_lines():
    data, config_dict = get_real_diff_in_diff_input()
    config = set_experiment_config(config_dict)
    data[Constants.WEIGHTED_SUM_COLUMN_NAME] = standardize_and_calculate_weighted_sum(
        data[config.matching_columns].to_numpy(), config
    )
    treatment_series, control_series = prep_matching_plot_data(data, config, control_unit_ids=[2, 7, 8, 9])
    plot_matching_parallel_lines(
        treatment_series,
        control_series,
        control_unit_ids=[2, 7, 8, 9],
        treatment_unit_ids=config.treatment_unit_ids,
    )
