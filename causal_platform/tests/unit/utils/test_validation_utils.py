"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from causal_platform.src.models.configuration_model.base_objects import (
    Column,
    ColumnType,
    DateColumn,
    MetricType,
)
from causal_platform.src.pipeline.experiment_pipelines.ab_pipeline import ABPipeline
from causal_platform.src.utils.validation_utils import check_column_in_data, check_column_is_type


class TestValidationUtils:
    @pytest.fixture
    def raw_data(self):

        data_length = 15

        data_columns = ["int_col", "proportion_col", "ratio_col", "str_col", "date_col"]

        data = pd.DataFrame(
            np.random.randint(0, 100, size=(data_length, len(data_columns))),
            columns=data_columns,
        )

        data["ratio_col"] = data["ratio_col"] / 100
        data["proportion_col"] = np.random.randint(0, 2, size=15)
        data["date_col"] = datetime.now()
        data["date_col"] += pd.to_timedelta(np.random.randint(0, 365, size=data_length), unit="d")
        data["str_col"] = "A"
        data["date_str_col"] = data["date_col"].dt.strftime("%Y-%m-%d")
        return data

    @pytest.fixture
    def data_with_None(self):
        data = pd.DataFrame(
            {
                "continuous_col": np.random.normal(10, 1, 100),
                "str_col": np.random.choice(["yes", "no"], p=[0.5, 0.5], size=100),
                "proportional_col": np.random.choice([1, 0], p=[0.5, 0.5], size=100),
                "exp_group": np.random.choice([1, 0], p=[0.5, 0.5], size=100),
            }
        )
        data = pd.concat(
            [
                data,
                pd.Series(
                    {
                        "continuous_col": None,
                        "str_col": None,
                        "proportional_col": None,
                        "exp_group": None,
                    }
                ),
            ],
            ignore_index=True,
        )
        data["continuous_col"] = data["continuous_col"].astype("O")
        data["proportional_col"] = data["proportional_col"].astype("O")
        return data

    @pytest.fixture
    def int_column_name(self):
        return "int_col"

    @pytest.fixture
    def proportion_column_name(self):
        return "proportion_col"

    @pytest.fixture
    def ratio_column_name(self):
        return "ratio_col"

    @pytest.fixture
    def str_column_name(self):
        return "str_col"

    @pytest.fixture
    def date_str_column_name(self):
        return "date_str_col"

    @pytest.fixture
    def date_column_name(self):
        return "date_col"

    @pytest.fixture
    def non_existing_column_name(self):
        return "does_not_exist"

    def test_check_column_in_data(self, raw_data):
        assert check_column_in_data(raw_data, "str_col") is True
        assert check_column_in_data(raw_data, "does_not_exist") is False

    def test_check_column_is_type(
        self,
        raw_data,
        int_column_name,
        proportion_column_name,
        ratio_column_name,
        str_column_name,
        date_column_name,
        date_str_column_name,
    ):

        int_column = Column(int_column_name, ColumnType.metric)
        proportion_column = Column(proportion_column_name, ColumnType.metric)
        ratio_column = Column(ratio_column_name, ColumnType.metric)
        str_column = Column(str_column_name, ColumnType.metric)
        date_column = DateColumn(date_column_name)
        date_str_column = DateColumn(date_str_column_name)

        # 1. continuous check
        assert check_column_is_type(raw_data, int_column, MetricType.continuous) is True
        assert check_column_is_type(raw_data, proportion_column, MetricType.continuous) is True
        assert check_column_is_type(raw_data, ratio_column, MetricType.continuous) is True
        assert check_column_is_type(raw_data, str_column, MetricType.continuous) is False
        assert check_column_is_type(raw_data, date_column, MetricType.continuous) is False
        assert check_column_is_type(raw_data, date_str_column, MetricType.continuous) is False

        # 2. proportion check
        assert check_column_is_type(raw_data, int_column, MetricType.proportional) is False
        assert check_column_is_type(raw_data, proportion_column, MetricType.proportional) is True
        assert check_column_is_type(raw_data, ratio_column, MetricType.proportional) is False
        assert check_column_is_type(raw_data, str_column, MetricType.proportional) is False
        assert check_column_is_type(raw_data, date_column, MetricType.proportional) is False
        assert check_column_is_type(raw_data, date_str_column, MetricType.proportional) is False

        # 4. date check
        assert check_column_is_type(raw_data, int_column, ColumnType.date) is False
        assert check_column_is_type(raw_data, proportion_column, ColumnType.date) is False
        assert check_column_is_type(raw_data, ratio_column, ColumnType.date) is False
        assert check_column_is_type(raw_data, str_column, ColumnType.date) is False
        assert check_column_is_type(raw_data, date_column, ColumnType.date) is True
        assert check_column_is_type(raw_data, date_str_column, ColumnType.date) is False

    def test_convert_dtype(self, data_with_None):
        config = {
            "columns": {
                "continuous_col": {"column_type": "metric", "metric_type": "continuous"},
                "proportional_col": {"column_type": "metric", "metric_type": "proportional"},
                "exp_group": {
                    "column_type": "experiment_group",
                    "variations": [0, 1],
                    "control_label": 0,
                },
            },
            "experiment_settings": {"type": "ab"},
        }
        pipeline = ABPipeline(data_with_None, config)
        pipeline.run()

        config = {
            "columns": {
                "str_col": {"column_type": "metric", "metric_type": "continuous"},
                "exp_group": {
                    "column_type": "experiment_group",
                    "variations": [0, 1],
                    "control_label": 0,
                },
            },
            "experiment_settings": {"type": "ab"},
        }
        pipeline = ABPipeline(data_with_None, config)
        res = pipeline.run(output_format="dict")
        assert "Metric column str_col must be type MetricType.continuous" in res["log_messages"]["errors"][0]
