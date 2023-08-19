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
from statsmodels.stats.power import zt_ind_solve_power

from causal_platform.src.pipeline.power_calculators.ab_power_calculator import ABPowerCalculator
from causal_platform.tests.data.data_generator import generate_ab_data


class TestPowerCalculator:
    """
    This is designed to test the correctness of power calculation, sample size calculation and
    absolute_treatment_effect calculation by comparing to the standard z-test calculator. For the
    non-clustered data, we do a direct comparison. For the clustered data, in order
    to use standard calculator, we make the clusters size homogeneous and take the cluster average
    as the input to z-test calculator
    """

    @pytest.fixture
    def clustered_data(self):
        df = generate_ab_data(
            group_mean=10,
            group_std=5,
            ar_cor=0.8,
            total_group=700,
            time_length=30,
            treatment_effect=0,
        )
        return df

    @pytest.fixture
    def non_clustered_data(self):
        df = generate_ab_data(
            group_mean=10,
            group_std=5,
            ar_cor=0.8,
            total_group=1,
            time_length=500,
            treatment_effect=0,
        )
        return df

    @pytest.fixture
    def ratio_data(self):
        df = pd.DataFrame({"numerator": np.random.normal(10, 1, 10000), "denominator": [1] * 10000})
        return df

    def test_run_power(self, clustered_data, non_clustered_data):
        """
        compare the power calculation with standard power z-test calculator
        """
        # non-clustered
        config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "absolute_treatment_effect": 2,
                    "sample_size_per_group": 100,
                },
            },
            "experiment_settings": {"type": "ab"},
        }

        ab_power_calculator = ABPowerCalculator(non_clustered_data, config)
        result = ab_power_calculator.run()
        expected = zt_ind_solve_power(effect_size=2 / non_clustered_data["asap"].std(), nobs1=100, alpha=0.05, ratio=1)
        assert np.abs(result["power"].iloc[0] - expected) <= 0.01

        # clustered
        config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "absolute_treatment_effect": 2,
                    "sample_size_per_group": 100,
                },
                "group": {"column_type": "cluster"},
            },
            "experiment_settings": {"type": "ab"},
        }

        ab_power_calculator = ABPowerCalculator(clustered_data, config)
        result = ab_power_calculator.run()
        expected = zt_ind_solve_power(
            effect_size=2 / clustered_data.groupby(["group"]).agg({"asap": "mean"}).reset_index()["asap"].std(),
            nobs1=100,
            alpha=0.05,
            ratio=1,
        )
        assert np.abs(result["power"].iloc[0] - expected) <= 0.01

    def test_run_sample_size(self, clustered_data, non_clustered_data):
        """
        compare the required sample size calculation with standard power z-test calculator
        """
        # non-clustered
        config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "absolute_treatment_effect": 2,
                    "power": 0.8,
                },
            },
            "experiment_settings": {"type": "ab"},
        }
        ab_power_calculator = ABPowerCalculator(non_clustered_data, config)
        result = ab_power_calculator.run()
        expected = zt_ind_solve_power(effect_size=2 / non_clustered_data["asap"].std(), power=0.8, alpha=0.05, ratio=1)
        assert np.abs(result["sample_size_per_group"].iloc[0] - expected) <= 1

        # clustered
        config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "absolute_treatment_effect": 2,
                    "power": 0.8,
                },
                "group": {"column_type": "cluster"},
            },
            "experiment_settings": {"type": "ab"},
        }
        ab_power_calculator = ABPowerCalculator(clustered_data, config)
        result = ab_power_calculator.run()
        expected = zt_ind_solve_power(
            effect_size=2 / clustered_data.groupby(["group"]).agg({"asap": "mean"}).reset_index()["asap"].std(),
            power=0.8,
            alpha=0.05,
            ratio=1,
        )
        assert np.abs(result["sample_size_per_group"].iloc[0] - expected) <= 1

    def test_run_absolute_treatment_effect(self, clustered_data, non_clustered_data):
        """
        compare the required sample size calculation with standard power z-test calculator
        """
        # non-clustered
        config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "power": 0.8,
                    "sample_size_per_group": 100,
                },
            },
            "experiment_settings": {"type": "ab"},
        }
        ab_power_calculator = ABPowerCalculator(non_clustered_data, config)
        result = ab_power_calculator.run()
        sigma = non_clustered_data["asap"].std()
        expected = zt_ind_solve_power(nobs1=100, alpha=0.05, ratio=1, power=0.8) * sigma
        assert np.abs(result["absolute_treatment_effect"].iloc[0] - expected) <= 0.01

        # clustered
        config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "power": 0.8,
                    "sample_size_per_group": 100,
                },
                "group": {"column_type": "cluster"},
            },
            "experiment_settings": {"type": "ab"},
        }
        ab_power_calculator = ABPowerCalculator(clustered_data, config)
        result = ab_power_calculator.run()
        sigma = clustered_data.groupby(["group"]).agg({"asap": "mean"}).reset_index()["asap"].std()
        expected = zt_ind_solve_power(nobs1=100, alpha=0.05, ratio=1, power=0.8) * sigma
        assert np.abs(result["absolute_treatment_effect"].iloc[0] - expected) <= 0.01

    def test_multiple_metrics_power(self, clustered_data):
        df = clustered_data.copy()
        df["amplified_asap"] = df["asap"] * 100
        config = {
            "columns": {
                "asap": [
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "absolute_treatment_effect": 40,
                        "sample_size_per_group": 1000,
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "absolute_treatment_effect": 0,
                        "sample_size_per_group": 1000,
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "absolute_treatment_effect": 0,
                        "sample_size_per_group": 0,
                    },
                ],
                "amplified_asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "absolute_treatment_effect": 40,
                    "sample_size_per_group": 1000,
                },
                "group": {
                    "column_type": "cluster",
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }
        ab_power_calculator = ABPowerCalculator(df, config)
        result = ab_power_calculator.run()
        assert result.shape == (4, 4)
        # when absolute_treatment_effect is zero, power should be equal to 0.05
        assert (
            result[
                (result["metric"] == "asap")
                & (result["sample_size_per_group"] == 0)
                & (result["absolute_treatment_effect"] == 0)
            ]["power"].iloc[0]
            == 0.05
        )
        # given the same absolute_treatment_effect and sample size, data with higher variation should have less power
        assert (
            result[
                (result["metric"] == "asap")
                & (result["sample_size_per_group"] == 1000)
                & (result["absolute_treatment_effect"] == 40)
            ]["power"].iloc[0]
            >= result[
                (result["metric"] == "amplified_asap")
                & (result["sample_size_per_group"] == 1000)
                & (result["absolute_treatment_effect"] == 40)
            ]["power"].iloc[0]
        )

    def test_multiple_metrics_sample_size(self, clustered_data):
        df = clustered_data.copy()
        df["amplified_asap"] = df["asap"] * 100
        config = {
            "columns": {
                "asap": [
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "absolute_treatment_effect": 10,
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "absolute_treatment_effect": 10,
                        "power": 0.5,
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "absolute_treatment_effect": 10,
                        "power": 0,
                    },
                ],
                "amplified_asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "absolute_treatment_effect": 10,
                    "power": 0.5,
                },
                "group": {
                    "column_type": "cluster",
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }
        ab_power_calculator = ABPowerCalculator(df, config)
        result = ab_power_calculator.run()

        # when std(col1) is 100x of std(col2), for the same absolute_treatment_effect,
        # the sample size required for col1 should be 10000x of col2
        assert (
            np.abs(
                result[
                    (result["metric"] == "asap")
                    & (result["power"] == 0.5)
                    & (result["absolute_treatment_effect"] == 10)
                ]["sample_size_per_group"].iloc[0]
                * 100**2
                - result[
                    (result["metric"] == "amplified_asap")
                    & (result["power"] == 0.5)
                    & (result["absolute_treatment_effect"] == 10)
                ]["sample_size_per_group"].iloc[0]
            )
            <= 0.001
        )

    def test_multiple_metrics_absolute_treatment_effect(self, clustered_data):
        df = clustered_data.copy()
        df["amplified_asap"] = df["asap"] * 100
        config = {
            "columns": {
                "asap": [
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "power": 0.5,
                        "sample_size_per_group": 100,
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "power": 0,
                        "sample_size_per_group": 100,
                    },
                    {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "power": 0.8,
                        "sample_size_per_group": 100,
                    },
                ],
                "amplified_asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "power": 0.5,
                    "sample_size_per_group": 100,
                },
                "group": {
                    "column_type": "cluster",
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }
        ab_power_calculator = ABPowerCalculator(df, config)
        result = ab_power_calculator.run()

        # when std(col1) is 100x of std(col2), for the same power],
        # the absolute_treatment_effect for col1 should be 100x of col2
        assert (
            np.abs(
                result[
                    (result["metric"] == "asap") & (result["power"] == 0.5) & (result["sample_size_per_group"] == 100)
                ]["absolute_treatment_effect"].iloc[0]
                * 100
                - result[
                    (result["metric"] == "amplified_asap")
                    & (result["power"] == 0.5)
                    & (result["sample_size_per_group"] == 100)
                ]["absolute_treatment_effect"].iloc[0]
            )
            <= 0.001
        )

    def test_config_input(self, non_clustered_data):
        # test missing element
        config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "absolute_treatment_effect": 2,
                },
            },
            "experiment_settings": {"type": "ab"},
        }

        ab_power_calculator = ABPowerCalculator(non_clustered_data, config)
        with pytest.raises(Exception):
            ab_power_calculator.run()
        # test too many element
        config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "absolute_treatment_effect": 2,
                    "power": 0.8,
                    "sample_size_per_group": 100,
                },
            },
            "experiment_settings": {"type": "ab"},
        }

        ab_power_calculator = ABPowerCalculator(non_clustered_data, config)
        with pytest.raises(Exception):
            ab_power_calculator.run()

    def test_run_metric_stats_calculator(self, ratio_data):
        config = {
            "columns": {
                "metric": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "numerator",
                    "denominator_column": "denominator",
                },
            },
            "experiment_settings": {"type": "ab"},
        }
        ab_power_calculator = ABPowerCalculator(ratio_data, config)
        result = ab_power_calculator.run_metric_stats_calculator()
        assert result["metric"]["standard_deviation"] > 0
        assert result["metric"]["metric_value"] > 0
        assert result["metric"]["sample_size"] == 10000
