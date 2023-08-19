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

from causal_platform.src.models.configuration_model.base_objects import (
    ColumnType,
    ExperimentType,
    MatchingMethod,
    MetricAggregateFunc,
    MetricType,
)
from causal_platform.src.utils.config_utils import (
    parse_ab_experiment_settings,
    parse_config_columns,
    parse_diff_in_diff_experiment_settings,
    parse_experiment_group,
    set_experiment_config,
)
from causal_platform.src.utils.constants import Constants
from causal_platform.tests.unit.data import (
    get_diff_in_diff_input,
)


class TestConfigUtils:
    @pytest.fixture
    def raw_data(self):
        return pd.DataFrame(np.random.randint(0, 100, size=(15, 5)), columns=list("ABCDE"))

    @pytest.fixture
    def normal_config(self):
        config = {
            "columns": {
                "bucket": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["treatment", "control"],
                    "variations_split": [0.5, 0.5],
                },
                "created_at": {"column_type": "date"},
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "log_transform": True,
                    "remove_outlier": True,
                    "check_distribution": False,
                },
                "dat": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "log_transform": True,
                    "remove_outlier": True,
                    "check_distribution": False,
                },
            },
            "experiment_settings": {
                "is_check_imbalance": True,
                "is_check_flickers": True,
                "is_check_metric_type": True,
                "fixed_effect_estimator": True,
                "type": "ab",
            },
        }
        return config

    @pytest.fixture
    def ratio_metric_config(self):
        config = {
            "columns": {
                "group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "date": {"column_type": "date"},
                "mto": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "delivery",
                    "denominator_column": "ticket",
                },
            },
            "experiment_settings": {
                "is_check_imbalance": False,
                "is_check_flickers": False,
                "is_check_metric_type": False,
                "type": "ab",
            },
        }
        return config

    @pytest.fixture
    def simple_config(self):
        config = {
            "columns": {
                "bucket": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["treatment", "control"],
                    "variations_split": [0.5, 0.5],
                },
                "dat": {"column_type": "metric", "metric_type": "continuous"},
            },
            "experiment_settings": {"type": "ab"},
        }
        return config

    @pytest.fixture
    def multiple_dates_config(self):
        config = {
            "columns": {
                "bucket": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["treatment", "control"],
                    "variations_split": [0.5, 0.5],
                },
                "created_at": {"column_type": "date"},
                "updated_at": {"column_type": "date"},
                "asap": {"column_type": "metric", "metric_type": "continuous"},
                "dat": {"column_type": "metric", "metric_type": "continuous"},
            },
            "experiment_settings": {
                "is_check_imbalance": True,
                "is_check_flickers": True,
                "is_check_metric_type": True,
                "type": "ab",
            },
        }
        return config

    @pytest.fixture
    def covar_config(self):
        config = {
            "columns": {
                "bucket": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["treatment", "control"],
                    "variations_split": [0.5, 0.5],
                },
                "created_at": {"column_type": "date"},
                "updated_at": {"column_type": "date"},
                "asap": {"column_type": "metric", "metric_type": "continuous"},
                "dat": {"column_type": "metric", "metric_type": "continuous"},
                "alt": {"column_type": "covariate", "value_type": "numerical"},
                "t2p": {
                    "column_type": "covariate",
                    "value_type": "numerical",
                    "applied_metrics": ["asap"],
                },
                "cluster1": {"column_type": "cluster", "applied_metrics": ["asap"]},
                "cluster2": {"column_type": "cluster"},
            },
            "experiment_settings": {
                "is_check_imbalance": True,
                "is_check_flickers": True,
                "is_check_metric_type": True,
                "type": "ab",
            },
        }
        return config

    @pytest.fixture
    def multi_config(self):
        config = {
            "columns": {
                "asap": [
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "quantile",
                        "quantile": 0.95,
                    },
                ],
                "unit": [
                    {"column_type": "experiment_randomize_unit"},
                    {"column_type": "cluster"},
                ],
                "group": {"column_type": "experiment_group"},
            },
            "experiment_settings": {
                "is_check_imbalance": True,
                "is_check_flickers": True,
                "is_check_metric_type": True,
                "type": "ab",
            },
        }
        return config

    @pytest.fixture
    def configs(
        self,
        normal_config,
        simple_config,
        multiple_dates_config,
        covar_config,
        multi_config,
        ratio_metric_config,
    ):
        return {
            "normal_config": normal_config,
            "simple_config": simple_config,
            "multiple_dates_config": multiple_dates_config,
            "covar_config": covar_config,
            "multi_config": multi_config,
            "ratio_metric_config": ratio_metric_config,
        }

    def test_config_utils(self, raw_data, configs):

        # 1. normal config
        normal_config = configs["normal_config"]
        config_result = set_experiment_config(normal_config)

        # dates
        assert (config_result.date.column_name, config_result.date.column_type) == (
            "created_at",
            ColumnType.date,
        )
        assert config_result.fixed_effect_estimator is True
        # experiment_groups
        assert (
            config_result.experiment_groups[0].column_name,
            config_result.experiment_groups[0].column_type,
        ) == ("bucket", ColumnType.experiment_group)
        # covariates
        assert len(config_result.covariates) == 0
        # metrics
        assert len(config_result.metrics) == 2
        assert (config_result.metrics[1].column_name, config_result.metrics[1].column_type,) == (
            "dat",
            ColumnType.metric,
        )
        assert config_result.metrics[1].metric_aggregate_func == MetricAggregateFunc.mean
        assert config_result.metrics[1].metric_type == MetricType.continuous
        # experiment type
        assert config_result.experiment_type == ExperimentType.ab

        # required columns
        assert config_result.get_required_columns_names() == {
            "asap",
            "created_at",
            "dat",
            "bucket",
        }

        # 2. simple config with many fields unspecified in raw config
        simple_config = configs["simple_config"]
        config_result = set_experiment_config(simple_config)
        # dates
        assert config_result.date is None
        # experiment_groups
        assert (
            config_result.experiment_groups[0].column_name,
            config_result.experiment_groups[0].column_type,
        ) == ("bucket", ColumnType.experiment_group)
        # covariates
        assert len(config_result.covariates) == 0
        # metrics
        assert len(config_result.metrics) == 1
        assert (config_result.metrics[0].column_name, config_result.metrics[0].column_type,) == (
            "dat",
            ColumnType.metric,
        )
        assert config_result.metrics[0].metric_aggregate_func == MetricAggregateFunc.mean
        assert config_result.metrics[0].metric_type == MetricType.continuous
        # experiment type
        assert config_result.experiment_type == ExperimentType.ab

        # 3. multiple_dates_config config
        multiple_dates_config = configs["multiple_dates_config"]
        config_result = set_experiment_config(multiple_dates_config)

        # dates
        assert (config_result.date.column_name, config_result.date.column_type) == (
            "updated_at",
            ColumnType.date,
        )
        # experiment_groups
        assert (
            config_result.experiment_groups[0].column_name,
            config_result.experiment_groups[0].column_type,
        ) == ("bucket", ColumnType.experiment_group)
        # covariates
        assert len(config_result.covariates) == 0
        # metrics
        assert len(config_result.metrics) == 2
        assert (config_result.metrics[1].column_name, config_result.metrics[1].column_type,) == (
            "dat",
            ColumnType.metric,
        )
        assert config_result.metrics[1].metric_aggregate_func == MetricAggregateFunc.mean
        assert config_result.metrics[1].metric_type == MetricType.continuous
        # experiment type
        assert config_result.experiment_type == ExperimentType.ab

        # 4. multiple_dates_config config
        covar_config = configs["covar_config"]
        config_result = set_experiment_config(covar_config)

        # dates
        assert (config_result.date.column_name, config_result.date.column_type) == (
            "updated_at",
            ColumnType.date,
        )
        # experiment_groups
        assert (
            config_result.experiment_groups[0].column_name,
            config_result.experiment_groups[0].column_type,
        ) == ("bucket", ColumnType.experiment_group)
        # covariates
        assert len(config_result.covariates) == 2

        assert (
            config_result.covariates[0].column_name,
            config_result.covariates[0].column_type,
        ) == ("alt", ColumnType.covariate)
        assert len(config_result.metrics[0].covariates) == 2
        assert len(config_result.metrics[1].covariates) == 1
        assert config_result.metrics[0].covariates[0].column_name == "t2p"
        assert len(config_result.all_distinct_covariates) == 2
        assert config_result.all_distinct_covariates[0].column_name in ["alt", "t2p"]

        # clusters
        assert len(config_result.clusters) == 2
        assert len(config_result.metrics[0].clusters) == 2
        assert len(config_result.metrics[1].clusters) == 1
        assert config_result.metrics[0].cluster.column_name == "cluster1"
        assert config_result.metrics[1].cluster.column_name == "cluster2"
        assert len(config_result.all_distinct_clusters) == 2
        assert config_result.all_distinct_clusters[0].column_name in [
            "cluster1",
            "cluster2",
        ]
        assert config_result.all_distinct_clusters[1].column_name in [
            "cluster1",
            "cluster2",
        ]
        assert config_result.cluster.column_name == "cluster1"
        # metrics
        assert len(config_result.metrics) == 2
        assert (config_result.metrics[0].column_name, config_result.metrics[0].column_type,) == (
            "asap",
            ColumnType.metric,
        )
        # experiment type
        assert config_result.experiment_type == ExperimentType.ab
        # variations
        assert config_result.experiment_groups[0].control.variation_name == "control"
        assert config_result.experiment_groups[0].treatments[0].variation_name == "treatment"
        assert config_result.experiment_groups[0].control.variation_split == 0.5
        assert config_result.experiment_groups[0].treatments[0].variation_split == 0.5

        # 5. ratio_metric_config config
        ratio_metric_config = configs["ratio_metric_config"]
        config_result = set_experiment_config(ratio_metric_config)
        assert config_result.get_required_columns_names() == {
            "ticket",
            "delivery",
            "date",
            "group",
        }

    def test_diff_in_diff_config(self, raw_data, configs):
        data, did_config = get_diff_in_diff_input()
        config_result = set_experiment_config(did_config)

        self.assertion_columns(
            config_result,
            0,
            1,
            "applicant",
            ColumnType.metric,
            MetricAggregateFunc.mean,
            MetricType.continuous,
        )

        self.assertion_diff_in_diff_experiment_settings(
            config_result,
            1,
            5,
            MatchingMethod.correlation,
            "2019-01-01",
            "2019-01-04",
            "2019-01-05",
            "2019-01-12",
        )
        assert config_result.get_required_columns_names() == {"date", "applicant", "cvr", "market"}

    def test_parse_config_columns(self, raw_data, configs):
        data, did_config = get_diff_in_diff_input()
        (
            metrics,
            experiment_groups,
            date,
            covariates,
            diff_in_diff_regions,
            clusters,
        ) = parse_config_columns(did_config, ExperimentType.diff_in_diff)

        assert len(metrics) == 1
        assert experiment_groups == []
        assert date.column_name == "date"
        assert len(covariates) == 0
        assert diff_in_diff_regions[0].column_name == "market"
        assert clusters == []

    def test_parse_ab_experiment_settings(self, raw_data, configs):
        ab_config = configs["normal_config"]
        is_check_flickers = parse_ab_experiment_settings(ab_config[Constants.EXPERIMENT_SETTINGS])
        assert is_check_flickers

    def test_parse_diff_in_diff_experiment_settings(self, raw_data, configs):
        data, did_config = get_diff_in_diff_input()
        (
            treatment_unit_ids,
            match_unit_size,
            matching_method,
            matching_start_date,
            matching_end_date,
            experiment_start_date,
            experiment_end_date,
            exclude_region_ids,
            matching_columns,
            matching_weights,
            small_sample_adjustment,
        ) = parse_diff_in_diff_experiment_settings(did_config[Constants.EXPERIMENT_SETTINGS])
        assert len(treatment_unit_ids) == 1
        assert treatment_unit_ids[0] == 1
        assert match_unit_size == 5
        assert matching_method == MatchingMethod.correlation
        assert matching_start_date == pd.Timestamp("2019-01-01")
        assert matching_end_date == pd.Timestamp("2019-01-04")
        assert experiment_start_date == pd.Timestamp("2019-01-05")
        assert experiment_end_date == pd.Timestamp("2019-01-12")
        assert exclude_region_ids == [3]
        assert len(matching_columns) == 2
        assert len(matching_weights) == 2
        assert small_sample_adjustment is True

    def assertion_ab_experiment_settings(
        self,
        config_result,
        is_check_imbalance,
        is_check_flickers,
        is_check_metric_type,
        num_experiment_groups,
    ):
        assert config_result.is_check_imbalance == is_check_imbalance
        assert config_result.is_check_flickers == is_check_flickers
        assert config_result.is_check_metric_type == is_check_metric_type
        assert len(config_result.experiment_groups) == num_experiment_groups

    def assertion_columns(
        self,
        config_result,
        num_covariate,
        num_metric,
        first_metric_name,
        first_metric_type,
        first_metric_agg_func,
        first_metric_metric_type,
    ):
        # covariates
        assert len(config_result.covariates) == num_covariate
        # metrics
        assert len(config_result.metrics) == num_metric
        assert (config_result.metrics[0].column_name, config_result.metrics[0].column_type,) == (
            first_metric_name,
            first_metric_type,
        )
        assert config_result.metrics[0].metric_aggregate_func == first_metric_agg_func
        assert config_result.metrics[0].metric_type == first_metric_metric_type

    def assertion_diff_in_diff_experiment_settings(
        self,
        config_result,
        num_treatment_unit_ids,
        match_unit_size,
        matching_method,
        matching_start_date,
        matching_end_date,
        experiment_start_date,
        experiment_end_date,
    ):
        assert len(config_result.treatment_unit_ids) == num_treatment_unit_ids
        assert config_result.match_unit_size == match_unit_size
        assert config_result.matching_method == matching_method
        assert config_result.matching_start_date == pd.Timestamp(matching_start_date)
        assert config_result.matching_end_date == pd.Timestamp(matching_end_date)
        assert config_result.experiment_start_date == pd.Timestamp(experiment_start_date)
        assert config_result.experiment_end_date == pd.Timestamp(experiment_end_date)
        assert config_result.matching_method == matching_method

    def test_parse_experiment_group(self):
        experiment_group_column = {
            "column_type": "experiment_group",
            "control_label": "control",
            "variations": ["control", "treatment"],
            "variations_split": [0.5, 0.5],
        }
        experiment_group = parse_experiment_group(experiment_group_column, "group")
        assert experiment_group.column_name == "group"
        assert experiment_group.control.variation_name == "control"
        assert experiment_group.control.variation_split == 0.5
        assert len(experiment_group.treatments) == 1
        assert experiment_group.treatments[0].variation_name == "treatment"
        assert experiment_group.treatments[0].variation_split == 0.5

    def test_multiple_config(self, raw_data, configs):
        config = set_experiment_config(configs["multi_config"])
        assert len(config.metrics) == 2
        assert len(config.experiment_randomize_units) == 1
        assert config.cluster is not None
