"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import copy
import json
import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
import statsmodels.api as sm

from causal_platform.src.models.configuration_model.config import PipelineConfig
from causal_platform.src.models.data_model import DataLoader
from causal_platform.src.pipeline.experiment_pipelines.ab_pipeline import ABPipeline
from causal_platform.src.pipeline.experiment_pipelines.diff_in_diff_pipeline import DiffinDiffPipeline
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.error import InputConfigError
from causal_platform.src.utils.validation_utils import check_data_is_datetime
from causal_platform.tests.data import data_generator
from causal_platform.tests.data.data_generator import generate_ab_data
from causal_platform.tests.unit.data import (
    get_ab_input_with_flicker_and_imbalance,
    get_ab_int_group_input,
    get_preprocess_only_test_input,
    get_quantile_test_input,
    get_ratio_test_input,
    get_real_diff_in_diff_input,
    get_redundant_columns_table_input,
    get_small_sample_diff_in_diff_input,
    get_test_input,
)
from causal_platform.tests.unit.models.analyser_model.test_data import ABTestBase


class TestPipeline(ABTestBase):
    def test_ab_pipeline(self):
        data, config = get_test_input()
        pipeline = ABPipeline(data, config)
        result = pipeline.run()
        assert len(result) == 6

        for _, val in result.items():
            assert len(val) > 0

        # check the date column has been properly converted
        processed_data = pipeline.preprocess_result.processed_data
        assert check_data_is_datetime(processed_data, "date")

    def test_ab_pipeline_with_group_sequential(self):
        data, config = get_test_input()
        config["experiment_settings"]["information_rates"] = [0.1, 0.2, 1.0]
        config["experiment_settings"]["target_sample_size"] = 50
        pipeline = ABPipeline(data, config)
        result = pipeline.run()
        assert len(result) == 6

        for _, val in result.items():
            assert len(val) > 0

    def test_ab_pipeline_with_group_sequential_where_sample_sizes_are_provided(self):
        data, config = get_test_input()
        config["experiment_settings"]["information_rates"] = [0.1, 0.2, 1.0]
        config["experiment_settings"]["target_sample_size"] = 100
        config["experiment_settings"]["current_sample_size"] = 50
        pipeline = ABPipeline(data, config)
        result = pipeline.run()
        assert len(result) == 6

        for _, val in result.items():
            assert len(val) > 0

    def test_ab_pipeline_2(self, ab_config_dict, data):
        pipeline = ABPipeline(data, ab_config_dict)
        result = pipeline.run()
        assert len(result) == 8

        for _, val in result.items():
            assert len(val) > 0

        # test fixed effect demean
        ab_config_dict["experiment_settings"]["fixed_effect_estimator"] = True
        pipeline = ABPipeline(data, ab_config_dict)
        assert pipeline.config.fixed_effect_estimator is True
        result2 = pipeline.run()
        assert len(result2) == 8

        trt1 = result[result["variation_name"] == "treatment"]
        trt2 = result2[(result2["variation_name"] == "treatment")]
        asap_ate1 = trt1[trt1.metric_name == "ASAP"].iloc[0]["average_treatment_effect"]
        asap_ate2 = trt2[trt2.metric_name == "ASAP"].iloc[0]["average_treatment_effect"]
        dat_ate1 = trt1[trt1.metric_name == "DAT"].iloc[0]["average_treatment_effect"]
        dat_ate2 = trt2[trt2.metric_name == "DAT"].iloc[0]["average_treatment_effect"]
        assert round(asap_ate1, 1) == round(asap_ate2, 1)
        assert round(dat_ate1, 1) == round(dat_ate2, 1)
        assert trt1[trt1.metric_name == "ASAP"].iloc[0]["SE"] > 0

    def test_ab_pipeline_data_size_output(self):
        # Test with DF as output
        data, config = get_test_input()
        pipeline = ABPipeline(data, config)
        result = pipeline.run()
        assert len(result) == 6
        assert "data_size" in result
        assert result[result["variation_name"] == "treatment"].iloc[0]["data_size"] == 4

        # Test with dict as output
        result_dict = pipeline.run(output_format="dict")
        assert len(result_dict["metric_results"]) == 2

        metric1_dict = list(result_dict["metric_results"])[0]
        assert (
            metric1_dict["experiment_group_results"][0]["agg_func_results"][0]["control_results"][0]["data_size"] == 4
        )
        assert (
            metric1_dict["experiment_group_results"][0]["agg_func_results"][0]["treatment_results"][0]["data_size"] == 4
        )

    def test_ab_pipline_with_customized_covar(
        self, data, config_without_customized_covar, config_with_customized_covar
    ):
        pipeline_config = PipelineConfig(copy_input_data=False)
        pipeline1 = ABPipeline(data, config_without_customized_covar, pipeline_config)
        assert len(pipeline1.config.metrics[0].covariates) == 1
        assert len(pipeline1.config.metrics[1].covariates) == 1
        assert pipeline1.config.metrics[0].covariates[0].column_name == "submarket_id"
        assert len(pipeline1.config.metrics[0].clusters) == 1
        assert pipeline1.config.metrics[0].clusters[0].column_name == "unit_id"
        assert len(pipeline1.config.metrics[1].clusters) == 1
        assert len(pipeline1.config.metrics[1].clusters) == 1
        pipeline2 = ABPipeline(data, config_with_customized_covar, pipeline_config)
        result2 = pipeline2.run()
        assert len(pipeline2.config.metrics[0].covariates) == 1
        assert pipeline2.config.metrics[0].covariates[0].column_name == "submarket_id"
        assert len(pipeline2.config.metrics[1].covariates) == 0
        assert len(pipeline2.config.metrics[0].clusters) == 1
        assert pipeline2.config.metrics[0].clusters[0].column_name == "unit_id"
        assert len(pipeline2.config.metrics[1].clusters) == 0

        result_with_var_red = smf.ols("asap ~ C(group1) + C(submarket_id)", data=data).fit()
        result_without_var_red = smf.ols("dat ~ C(group1)", data=data).fit()
        assert (
            abs(
                result2.loc[
                    (result2["metric_name"] == "ASAP") & (result2["variation_name"] == "treatment"),
                    "average_treatment_effect",
                ].iloc[0]
                - result_with_var_red.params.loc["C(group1)[T.treatment]"]
            )
            <= 0.001
        )
        assert (
            abs(
                result2.loc[
                    (result2["metric_name"] == "DAT") & (result2["variation_name"] == "treatment"),
                    "average_treatment_effect",
                ].iloc[0]
                - result_without_var_red.params.loc["C(group1)[T.treatment]"]
            )
            <= 0.001
        )

    def test_ab_pipeline_remove_redundant_columns(self):
        data, config = get_redundant_columns_table_input()
        pl = ABPipeline(data, config)
        result = pl.run()
        assert set(pl.data.columns) == {"metric1", "group", "date"}
        assert len(result) == 2

    def test_ab_pipeline_basic_fitter(self):
        df = pd.DataFrame(
            {
                "metric": np.random.normal(100, 1, 1000),
                "exp_group": np.random.choice([0, 1], p=[0.5, 0.5], size=1000),
                "cluster": np.random.choice([0, 1, 2, 3, 5], p=[0.2] * 5, size=1000),
            }
        )
        # average treatment effect
        config = {
            "columns": {
                "exp_group": {
                    "column_type": "experiment_group",
                    "variations": [0, 1],
                    "variation_split": [0.5, 0.5],
                    "control_label": 0,
                },
                "metric": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "fitter_type": "basic",
                },
            },
            "experiment_settings": {"type": "ab"},
        }
        pl = ABPipeline(df, config)
        result = pl.run()
        assert result.shape[0] == 2

        # quantile treatment effect
        config["columns"]["metric"]["metric_aggregate_func"] = "quantile"
        config["columns"]["metric"]["quantile"] = 0.8
        pl = ABPipeline(df, config)
        result = pl.run()
        assert result.shape[0] == 2

        # quantile treatment effect with cluster
        arrays = []
        cluster_mean = np.random.normal(200, 80, 50)
        cluster_shape = [int(round(x)) for x in np.random.normal(500, 10, 50)]
        for i in np.arange(50):
            group = np.random.binomial(1, 0.5)
            groups = np.array([group] * cluster_shape[i]).reshape(-1, 1)
            d = np.random.normal(cluster_mean[i], 10, cluster_shape[i]).reshape(-1, 1)
            c = np.array([i] * cluster_shape[i]).reshape(-1, 1)
            arrays.append(np.concatenate([d, c, groups], axis=1))
        df = pd.DataFrame(np.concatenate(arrays), columns=["metric", "cluster", "exp_group"])
        config["columns"]["cluster"] = {"column_type": "cluster"}
        pl = ABPipeline(df, config)
        result = pl.run()
        assert result.shape[0] == 2

    def test_ab_pipeline_causal_fitter_causal_analysis(self, causal_test_data):
        config = {
            "columns": {
                "metric1": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric1",
                    "denominator_column": "denominator",
                    "applied_covariates": ["cov1"],
                },
                "metric2": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric2",
                    "denominator_column": "denominator",
                    "applied_covariates": ["cov1", "cov2"],
                },
                "metric3": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric3",
                    "denominator_column": "denominator",
                    "applied_covariates": ["cov2"],
                },
                "cov1": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov1",
                    "denominator_column": "denominator",
                },
                "cov2": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov2",
                    "denominator_column": "denominator",
                },
            },
            "experiment_settings": {
                "type": "causal",
            },
        }
        pipeline = ABPipeline(causal_test_data, config)
        pipeline.run()
        covariates_results = pipeline.get_covariate_results()

        coef = covariates_results[
            (covariates_results["covariate_name"] == "cov2") & (covariates_results["metric_name"] == "metric2")
        ].iloc[0]["coefficient"]
        se = covariates_results[
            (covariates_results["covariate_name"] == "cov2") & (covariates_results["metric_name"] == "metric2")
        ].iloc[0]["SE"]

        # we expect the delta-method generates the same output as regression
        model = sm.OLS(causal_test_data["metric2"], sm.add_constant(causal_test_data[["cov1", "cov2"]]))
        expected_results = model.fit()
        expected_coef = expected_results.params[2]
        expected_se = expected_results.bse[2]

        assert abs(coef - expected_coef) <= 0.001
        assert abs(se - expected_se) <= 0.001

    def test_ab_pipeline_causal_fitter_ab_testing(self, causal_test_data):
        config = {
            "columns": {
                "exp_group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "metric1": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric1",
                    "denominator_column": "denominator",
                    "applied_covariates": ["cov1"],
                },
                "metric2": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric2",
                    "denominator_column": "denominator",
                    "applied_covariates": ["cov1", "cov2"],
                },
                "metric3": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric3",
                    "denominator_column": "denominator",
                    "applied_covariates": ["cov2"],
                },
                "cov1": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov1",
                    "denominator_column": "denominator",
                },
                "cov2": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov2",
                    "denominator_column": "denominator",
                },
            },
            "experiment_settings": {
                "type": "causal",
            },
        }
        pipeline = ABPipeline(causal_test_data, config)
        # AB testing experiment group result
        result = pipeline.run()
        # AB testing covariate result
        covariates_results = pipeline.get_covariate_results()
        assert len(result) == 6
        assert len(covariates_results) == 4

    def test_ab_quantile_pipeline(self):
        data, config = get_quantile_test_input()
        pipeline = ABPipeline(data, config)
        result = pipeline.run()
        assert len(result) == 2
        trt = result[result["variation_name"] == "treatment"]
        ate = trt.iloc[0]["average_treatment_effect"]
        assert ate == -2.8

    def test_ab_ratio_metric_pipeline(self):
        data, config = get_ratio_test_input(iterations=100)
        # basic fitter (delta method)
        pipeline = ABPipeline(data, config)
        result = pipeline.run()
        assert len(result) == 2
        trt = result[result["variation_name"] == "treatment"]
        ate = trt.iloc[0]["average_treatment_effect"]
        assert round(ate, 2) == 1.86
        # boostrap fitter
        config["columns"]["a_ratio_metric"]["fitter_type"] = "bootstrap"
        pipeline = ABPipeline(data, config)
        result = pipeline.run()
        trt = result[result["variation_name"] == "treatment"]
        ate = trt.iloc[0]["average_treatment_effect"]
        assert len(result) == 2
        assert round(ate, 2) == 1.86
        # metric with cluster
        data["cluster_id"] = np.random.choice(10, data.shape[0])
        config["columns"]["cluster_id"] = {"column_type": "cluster"}
        with pytest.raises(InputConfigError) as excinfo:
            ABPipeline(data, config)
        assert (
            excinfo.value.value == "a_ratio_metric is a ratio metric that has cluster cluster_id which is not supported"
        )

    def test_ab_ratio_metric_and_covariates_pipeline_json(self):
        data, config = get_ratio_test_input(use_cov=True)
        pipeline = ABPipeline(data, config)
        result = pipeline.run(output_format="dict")

        assert len(result) == 3
        assert len(pipeline.config.metrics[0].covariates) == 1

        json_result = pipeline.run(output_format="json")
        deserialized_dict = json.loads(json_result)
        assert deserialized_dict

    def test_ab_ratio_metric_variance_reduction_non_iterative_pipeline(self):
        df = data_generator.get_multi_ratio_cov_data()
        multi_cov_config = {
            "columns": {
                "exp_group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "metric": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric_n",
                    "denominator_column": "metric_d",
                },
                "cov1": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov1_n",
                    "denominator_column": "cov1_d",
                },
                "cov2": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov2_n",
                    "denominator_column": "cov2_d",
                },
            },
            "experiment_settings": {"type": "ab", "use_iterative_cv_method": False},
        }

        multi_cov_pipeline = ABPipeline(df, multi_cov_config)
        multi_multi_cov_result = multi_cov_pipeline.run()

        # for the noniterative method, we check if the VR optional coefficients matches OLS coefficients with 2 regressors
        coef_regression = []
        model_full = smf.ols("metric_n ~ 1 + cov1_n + cov2_n", data=df).fit()
        coef_regression.append(model_full.params[1])
        coef_regression.append(model_full.params[2])

        coef_delta = []
        coef_delta.append(multi_cov_pipeline.analyser.fitter_dict["metric"].covariates[0].coef)
        coef_delta.append(multi_cov_pipeline.analyser.fitter_dict["metric"].covariates[1].coef)

        # check the VR coefficient is expected
        assert np.abs(coef_regression[0] - coef_delta[0]) <= 1e-4
        assert np.abs(coef_regression[1] - coef_delta[1]) <= 1e-4

        df["adjusted_metric"] = df["metric_n"] - coef_regression[0] * df["cov1_n"] - coef_regression[1] * df["cov2_n"]
        pe = (
            df[df["exp_group"] == "treatment"]["adjusted_metric"].mean()
            - df[df["exp_group"] == "control"]["adjusted_metric"].mean()
        )
        se = np.sqrt(
            np.var(df[df["exp_group"] == "control"]["adjusted_metric"], ddof=1)
            / df[df["exp_group"] == "control"].shape[0]
            + np.var(df[df["exp_group"] == "treatment"]["adjusted_metric"], ddof=1)
            / df[df["exp_group"] == "treatment"].shape[0]
        )

        # check the reduced se is expected
        assert np.abs(se - multi_multi_cov_result.iloc[1]["SE"]) <= 1e-4
        # check the point estimate is expected
        assert np.abs(pe - multi_multi_cov_result.iloc[1]["average_treatment_effect"]) <= 1e-4

    def test_ab_ratio_metric_variance_reduction_pipeline(self):
        """
        Test the correctness of variance reduction with single/multi covariates
        The test validate the regression coefficient, ATE, and S.E associated with ATE
        """
        df = data_generator.get_multi_ratio_cov_data()

        single_cov_config = {
            "columns": {
                "exp_group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "metric": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric_n",
                    "denominator_column": "metric_d",
                },
                "cov1": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov1_n",
                    "denominator_column": "cov1_d",
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }

        single_cov_pipeline = ABPipeline(df, single_cov_config)
        single_cov_result = single_cov_pipeline.run()

        model_full = smf.ols("metric_n ~ 1 + cov1_n", data=df).fit()
        coef_regression = model_full.params[1]
        coef_delta = single_cov_pipeline.analyser.fitter_dict["metric"].covariates[0].coef

        df["adjusted_metric"] = df["metric_n"] - coef_regression * df["cov1_n"]
        pe = (
            df[df["exp_group"] == "treatment"]["adjusted_metric"].mean()
            - df[df["exp_group"] == "control"]["adjusted_metric"].mean()
        )
        se = np.sqrt(
            np.var(df[df["exp_group"] == "control"]["adjusted_metric"], ddof=1)
            / df[df["exp_group"] == "control"].shape[0]
            + np.var(df[df["exp_group"] == "treatment"]["adjusted_metric"], ddof=1)
            / df[df["exp_group"] == "treatment"].shape[0]
        )

        # check the VR coefficient is expected
        assert np.abs(coef_regression - coef_delta) <= 1e-4
        # check the reduced se is expected
        assert np.abs(se - single_cov_result.iloc[1]["SE"]) <= 1e-4
        # check the point estimate is expected
        assert np.abs(pe - single_cov_result.iloc[1]["average_treatment_effect"]) <= 1e-4

        multi_cov_config = {
            "columns": {
                "exp_group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "metric": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric_n",
                    "denominator_column": "metric_d",
                },
                "cov1": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov1_n",
                    "denominator_column": "cov1_d",
                },
                "cov2": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov2_n",
                    "denominator_column": "cov2_d",
                },
            },
            "experiment_settings": {"type": "ab", "use_iterative_cv_method": True},
        }

        multi_cov_pipeline = ABPipeline(df, multi_cov_config)
        multi_multi_cov_result = multi_cov_pipeline.run()

        # for the iterative method, we check if the VR optional coefficients matches 2 steps OLS coefficients
        coef_regression = []
        model_full = smf.ols("metric_n ~ 1 + cov1_n", data=df).fit()
        coef_regression.append(model_full.params[1])
        df["metric_temp"] = df["metric_n"] - coef_regression * df["cov1_n"]
        model_full = smf.ols("metric_temp ~ 1 + cov2_n", data=df).fit()
        coef_regression.append(model_full.params[1])

        coef_delta = []
        coef_delta.append(multi_cov_pipeline.analyser.fitter_dict["metric"].covariates[0].coef)
        coef_delta.append(multi_cov_pipeline.analyser.fitter_dict["metric"].covariates[1].coef)

        # check the VR coefficient is expected
        assert np.abs(coef_regression[0] - coef_delta[0]) <= 1e-4
        assert np.abs(coef_regression[1] - coef_delta[1]) <= 1e-4

        df["adjusted_metric"] = df["metric_n"] - coef_regression[0] * df["cov1_n"] - coef_regression[1] * df["cov2_n"]
        pe = (
            df[df["exp_group"] == "treatment"]["adjusted_metric"].mean()
            - df[df["exp_group"] == "control"]["adjusted_metric"].mean()
        )
        se = np.sqrt(
            np.var(df[df["exp_group"] == "control"]["adjusted_metric"], ddof=1)
            / df[df["exp_group"] == "control"].shape[0]
            + np.var(df[df["exp_group"] == "treatment"]["adjusted_metric"], ddof=1)
            / df[df["exp_group"] == "treatment"].shape[0]
        )

        # check the reduced se is expected
        assert np.abs(se - multi_multi_cov_result.iloc[1]["SE"]) <= 1e-4
        # check the point estimate is expected
        assert np.abs(pe - multi_multi_cov_result.iloc[1]["average_treatment_effect"]) <= 1e-4

    def test_ab_pipeline_multi_covariates_order(self):
        """
        Test the correctness of variance reduction with multi covariates of different orders
        The test validate the regression coefficient, ATE, and S.E associated with ATE
        """

        df = data_generator.get_multi_ratio_cov_data()

        multi_cov_config_a = {
            "columns": {
                "exp_group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "metric": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric_n",
                    "denominator_column": "metric_d",
                    "applied_covariates": ["cov1", "cov2"],
                },
                "cov1": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov1_n",
                    "denominator_column": "cov1_d",
                },
                "cov2": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov2_n",
                    "denominator_column": "cov2_d",
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }

        multi_cov_config_b = {
            "columns": {
                "exp_group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "metric": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric_n",
                    "denominator_column": "metric_d",
                    "applied_covariates": ["cov2", "cov1"],
                },
                "cov1": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov1_n",
                    "denominator_column": "cov1_d",
                },
                "cov2": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "cov2_n",
                    "denominator_column": "cov2_d",
                },
            },
            "experiment_settings": {"type": "ab", "use_iterative_cv_method": True},
        }

        multi_cov_pipeline = ABPipeline(df, multi_cov_config_a)
        multi_multi_cov_result = multi_cov_pipeline.run()
        assert [cov.column_name for cov in multi_cov_pipeline.config.metrics[0].covariates] == [
            "cov1",
            "cov2",
        ]

        multi_cov_pipeline = ABPipeline(df, multi_cov_config_b)
        multi_multi_cov_result = multi_cov_pipeline.run()
        assert [cov.column_name for cov in multi_cov_pipeline.config.metrics[0].covariates] == [
            "cov2",
            "cov1",
        ]

        # for the iterative method, we check if the VR optional coefficients matches 2 steps OLS coefficients
        coef_regression = []
        model_full = smf.ols("metric_n ~ 1 + cov2_n", data=df).fit()
        coef_regression.append(model_full.params[1])
        df["metric_temp"] = df["metric_n"] - coef_regression * df["cov2_n"]
        model_full = smf.ols("metric_temp ~ 1 + cov1_n", data=df).fit()
        coef_regression.append(model_full.params[1])

        coef_delta = []
        coef_delta.append(multi_cov_pipeline.analyser.fitter_dict["metric"].covariates[0].coef)
        coef_delta.append(multi_cov_pipeline.analyser.fitter_dict["metric"].covariates[1].coef)

        # check the VR coefficient is expected
        assert np.abs(coef_regression[0] - coef_delta[0]) <= 1e-4
        assert np.abs(coef_regression[1] - coef_delta[1]) <= 1e-4

        df["adjusted_metric"] = df["metric_n"] - coef_regression[0] * df["cov2_n"] - coef_regression[1] * df["cov1_n"]
        pe = (
            df[df["exp_group"] == "treatment"]["adjusted_metric"].mean()
            - df[df["exp_group"] == "control"]["adjusted_metric"].mean()
        )
        se = np.sqrt(
            np.var(df[df["exp_group"] == "control"]["adjusted_metric"], ddof=1)
            / df[df["exp_group"] == "control"].shape[0]
            + np.var(df[df["exp_group"] == "treatment"]["adjusted_metric"], ddof=1)
            / df[df["exp_group"] == "treatment"].shape[0]
        )

        # check the reduced se is expected
        assert np.abs(se - multi_multi_cov_result.iloc[1]["SE"]) <= 1e-4
        # check the point estimate is expected
        assert np.abs(pe - multi_multi_cov_result.iloc[1]["average_treatment_effect"]) <= 1e-4

    def test_ab_pipeline_zero_variance_covariates(self):
        """
        Test the correctness of variance reduction when there are covariates which have no variance
        The pipeline will need to drop the zero variance covaraites instead of return an Inf
        for the adjusted metric
        """
        df = data_generator.get_multi_ratio_cov_data()
        df["dumb_cov_numerator"] = -1
        df["dumb_cov_denominator"] = -1

        config = {
            "columns": {
                "exp_group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "metric": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric_n",
                    "denominator_column": "metric_d",
                    "applied_covariates": ["cov1", "cov2"],
                },
                "dumb_cov": {
                    "column_type": "covariate",
                    "value_type": "ratio",
                    "numerator_column": "dumb_cov_numerator",
                    "denominator_column": "dumb_cov_denominator",
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }

        config_no_cov = {
            "columns": {
                "exp_group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "metric": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "metric_n",
                    "denominator_column": "metric_d",
                    "applied_covariates": ["cov1", "cov2"],
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }
        pipeline = ABPipeline(df, config)
        result = pipeline.run()

        pipeline_no_cov = ABPipeline(df, config_no_cov)
        expected = pipeline_no_cov.run()
        assert result.iloc[1]["average_treatment_effect"] == expected.iloc[1]["average_treatment_effect"]
        assert result.iloc[1]["SE"] == expected.iloc[1]["SE"]

    def test_ab_pipeline_for_json_ouput(self):
        data, config = get_test_input()
        pipeline = ABPipeline(data, config)
        result = pipeline.run(output_format="dict")
        analysis_result = result[Constants.METRIC_RESULTS]
        assert len(analysis_result) == 2
        metric2_results = analysis_result[1][Constants.EXPERIMENT_GROUP_RESULTS]
        assert len(metric2_results) == 1
        exp_group = metric2_results[0][Constants.AGG_FUNC_RESULTS]
        assert len(exp_group) == 2
        treatment_results = exp_group[0][Constants.TREATMENT_RESULTS][0]
        assert Constants.SE in treatment_results
        assert Constants.P_VALUE in treatment_results
        assert Constants.AVERAGE_TREATMENT_EFFECT in treatment_results
        assert Constants.ABSOLUTE_CONFIDENCE_INTERVAL in treatment_results
        assert Constants.RELATIVE_AVERAGE_TREATMENT_EFFECT in treatment_results
        assert Constants.RELATIVE_CONFIDENCE_INTERVAL in treatment_results
        assert Constants.SEQUENTIAL_P_VALUE in treatment_results
        assert Constants.SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL in treatment_results
        assert Constants.SEQUENTIAL_RELATIVE_CONFIDENCE_INTERVAL in treatment_results
        assert Constants.SEQUENTIAL_RESULT_TYPE in treatment_results
        # check pre-process results
        preprocess_result = result[Constants.PREPROCESS_RESULTS]
        assert preprocess_result[Constants.DOES_FLICKER_EXISTS] is False
        assert preprocess_result[Constants.ARE_BUCKETS_IMBALANCED] is False

        assert treatment_results[Constants.SEQUENTIAL_P_VALUE]
        assert (
            treatment_results[Constants.SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL][0]
            and treatment_results[Constants.SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL][1]
        )
        assert (
            treatment_results[Constants.SEQUENTIAL_RELATIVE_CONFIDENCE_INTERVAL][0]
            and treatment_results[Constants.SEQUENTIAL_RELATIVE_CONFIDENCE_INTERVAL][1]
        )

        json_result = pipeline.run(output_format="json")
        deserialized_dict = json.loads(json_result)
        assert deserialized_dict

        # create a metric who's name is in uppercase
        config["columns"]["Metric1"] = config["columns"]["metric1"]
        config["columns"].pop("metric1")
        pipeline = ABPipeline(data, config)
        result = pipeline.run(output_format="json")
        result = json.loads(result)
        analysis_result = result[Constants.METRIC_RESULTS]
        # assert the original metric name in result is persisted (not lower case)
        analysis_result[0]["metric_name"] == "Metric1"

    def test_ab_pipeline_with_flickers_and_imbalance_for_json_ouput(self):
        data, config = get_ab_input_with_flicker_and_imbalance()
        pipeline_config = PipelineConfig(copy_input_data=False)
        pipeline = ABPipeline(data, config, pipeline_config)
        result = pipeline.run(output_format="dict")
        analysis_result = result[Constants.METRIC_RESULTS]
        assert len(analysis_result) == 2
        metric2_results = analysis_result[1][Constants.EXPERIMENT_GROUP_RESULTS]
        assert len(metric2_results) == 1
        agg_results = metric2_results[0][Constants.AGG_FUNC_RESULTS]
        treatment_results = agg_results[0][Constants.TREATMENT_RESULTS][0]
        assert Constants.SE in treatment_results
        assert Constants.P_VALUE in treatment_results
        assert Constants.AVERAGE_TREATMENT_EFFECT in treatment_results
        assert Constants.ABSOLUTE_CONFIDENCE_INTERVAL in treatment_results
        assert Constants.RELATIVE_AVERAGE_TREATMENT_EFFECT in treatment_results
        assert Constants.RELATIVE_CONFIDENCE_INTERVAL in treatment_results
        assert Constants.SEQUENTIAL_P_VALUE in treatment_results
        assert Constants.SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL in treatment_results
        assert Constants.SEQUENTIAL_RELATIVE_CONFIDENCE_INTERVAL in treatment_results
        # removed two flickers
        processed_data = pipeline.preprocess_result.processed_data
        metric2_group_avg = processed_data.groupby(["group"]).agg({"metric2": "mean"}).reset_index()
        metric2_treatment_effect = metric2_group_avg.iloc[1][1] - metric2_group_avg.iloc[0][1]
        assert (
            np.abs(
                metric2_results[0]["agg_func_results"][0]["treatment_results"][0]["average_treatment_effect"]
                - metric2_treatment_effect
            )
            <= 0.01
        )

        # check pre-process results
        assert len(processed_data) == 19
        preprocess_result = result[Constants.PREPROCESS_RESULTS]
        assert preprocess_result[Constants.DOES_FLICKER_EXISTS] is False
        assert preprocess_result[Constants.ARE_BUCKETS_IMBALANCED] is True

        assert treatment_results[Constants.SEQUENTIAL_P_VALUE]
        assert (
            treatment_results[Constants.SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL][0]
            and treatment_results[Constants.SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL][1]
        )
        assert (
            treatment_results[Constants.SEQUENTIAL_RELATIVE_CONFIDENCE_INTERVAL][0]
            and treatment_results[Constants.SEQUENTIAL_RELATIVE_CONFIDENCE_INTERVAL][1]
        )

        pipeline = ABPipeline(data, config, pipeline_config)
        json_result = pipeline.run(output_format="json")
        deserialized_dict = json.loads(json_result)
        assert deserialized_dict
        # check pre-process results
        preprocess_result = deserialized_dict[Constants.PREPROCESS_RESULTS]
        assert preprocess_result[Constants.DOES_FLICKER_EXISTS] is False
        assert preprocess_result[Constants.ARE_BUCKETS_IMBALANCED] is True

    def test_ab_quantile_pipeline_with_json_ouput(self):
        data, config = get_quantile_test_input()
        pipeline = ABPipeline(data, config)
        result = pipeline.run(output_format="dict")
        assert Constants.METRIC_RESULTS in result
        analysis_result = result[Constants.METRIC_RESULTS]
        assert len(analysis_result) == 1
        assert analysis_result[0][Constants.METRIC_NAME] == "metric1"
        metric_results = analysis_result[0][Constants.EXPERIMENT_GROUP_RESULTS]
        expt_group = metric_results[0][Constants.AGG_FUNC_RESULTS]
        treatment_results = expt_group[0][Constants.TREATMENT_RESULTS][0]
        control_results = expt_group[0][Constants.CONTROL_RESULTS][0]
        average_treatment_effect = treatment_results[Constants.AVERAGE_TREATMENT_EFFECT]
        relative_ate = treatment_results[Constants.RELATIVE_AVERAGE_TREATMENT_EFFECT]
        treatment_value = treatment_results[Constants.METRIC_VALUE]
        control_value = control_results[Constants.METRIC_VALUE]
        assert average_treatment_effect == -2.8
        assert average_treatment_effect / control_value == relative_ate
        assert control_value + average_treatment_effect == treatment_value
        json_result = pipeline.run(output_format="json")
        deserialized_dict = json.loads(json_result)
        assert deserialized_dict

    def test_ab_ratio_metric_pipeline_with_json_output(self):
        data, config = get_ratio_test_input()
        pipeline = ABPipeline(data, config)
        result = pipeline.run(output_format="dict")
        analysis_result = result[Constants.METRIC_RESULTS]
        agg_func_results = analysis_result[0][Constants.EXPERIMENT_GROUP_RESULTS][0][Constants.AGG_FUNC_RESULTS]
        metric_treatment_result = agg_func_results[0][Constants.TREATMENT_RESULTS][0]
        assert round(metric_treatment_result[Constants.AVERAGE_TREATMENT_EFFECT], 2) == 1.86
        assert metric_treatment_result[Constants.METRIC_VALUE] == 4
        json_result = pipeline.run(output_format="json")
        deserialized_dict = json.loads(json_result)
        assert deserialized_dict

    def test_ab_pipeline_preproces_only(self):
        data, config, config_bypass_check = get_preprocess_only_test_input()
        pipeline = ABPipeline(data, config)
        result = pipeline.run(output_format="json")
        result_dict = json.loads(result)
        assert "does_flicker_exists" in result_dict["preprocess_results"]
        assert "are_buckets_imbalanced" in result_dict["preprocess_results"]

        pipeline_bypass_check = ABPipeline(data, config_bypass_check)
        result_bypass_check = pipeline_bypass_check.run(output_format="json")
        result_dict__bypass_check = json.loads(result_bypass_check)
        assert "are_buckets_imbalanced" not in result_dict__bypass_check["preprocess_results"]
        assert "are_buckets_imbalanced" not in result_dict__bypass_check["preprocess_results"]

    def test_ab_pipeline_json_log_output(self):
        data, config = get_test_input()
        config["columns"]["metric2"][1]["fitter_type"] = "regression"
        config["columns"]["group"]["variations_split"] = [0.01, 0.99]
        pipeline = ABPipeline(data, config)
        result = pipeline.run(output_format="json")
        result_dict = json.loads(result)
        assert len(result_dict[Constants.LOG_MESSAGES][Constants.WARNINGS]) == 1

        # no inf exists in result
        data["metric3"] = (data["group"] == "treatment") * 2
        config = {
            "columns": {
                "group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                },
                "metric3": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "log_transform": True,
                    "remove_outlier": True,
                    "check_distribution": False,
                },
            },
            "experiment_settings": {
                "is_check_flickers": True,
                "is_check_imbalance": True,
                "is_check_metric_type": True,
                "type": "ab",
            },
        }
        pipeline = ABPipeline(data, config)
        result = pipeline.run(output_format="json")
        result_dict = json.loads(result)
        treatment_result = result_dict["metric_results"][0]["experiment_group_results"][0]["agg_func_results"][0][
            "treatment_results"
        ][0]

        assert treatment_result["average_treatment_effect"] == 2
        assert treatment_result["relative_average_treatment_effect"] is None
        assert treatment_result["rel_confidence_interval"] == [None] * 2

    def test_diff_in_diff_pipeline(self):
        data, config = get_real_diff_in_diff_input()
        pipeline = DiffinDiffPipeline(data, config)
        matched_markets = pipeline.matching()
        pipeline.plot_matching(control_unit_ids=matched_markets["applied_submarket_id"])
        assert len(matched_markets) == 5
        result = pipeline.run()
        assert result.analysis_result.shape[0] == 2
        assert result.matching_result.shape[0] == 5
        pipeline.plot_treatment_effect()

        # test different formats of date column
        """
        the case when date column is string typed and format is inferred
        the date part of config stay the same:
        "applied_date": {"column_type": "date"}
        """
        modified_data = data.copy()
        modified_data["applied_date"] = modified_data["applied_date"].dt.strftime("%Y-%m-%d")
        modified_config = copy.deepcopy(config)
        str_input_pipeline = DiffinDiffPipeline(modified_data, modified_config)
        str_input_result = str_input_pipeline.run()
        assert len(str_input_result.analysis_result) == len(result.analysis_result)

        """
        when the string format is defined by user
        update the date part of config to:
        "applied_date": {"column_type": "date", "date_format": "%m/%d/%Y, %H:%M:%S"}
        """
        modified_data = data.copy()
        modified_data["applied_date"] = modified_data["applied_date"].dt.strftime("%m/%d/%Y, %H:%M:%S")
        modified_config = copy.deepcopy(config)
        modified_config["columns"]["applied_date"]["date_format"] = "%m/%d/%Y, %H:%M:%S"
        str_input_pipeline = DiffinDiffPipeline(modified_data, modified_config)
        str_input_result = str_input_pipeline.run()
        assert len(str_input_result.analysis_result) == len(result.analysis_result)

        # test custom control units
        data, config = get_real_diff_in_diff_input()
        pipeline = DiffinDiffPipeline(data, config, control_unit_ids=[5, 10, 20, 11, 12, 13])
        result = pipeline.run()
        assert result.matching_result.shape[0] == 5

        # test small sample
        (
            data,
            config_no_adjust,
            config_with_adjust,
        ) = get_small_sample_diff_in_diff_input()
        pipeline = DiffinDiffPipeline(data, config_no_adjust)
        result_without_adjust = pipeline.run()
        pipeline = DiffinDiffPipeline(data, config_with_adjust)
        result_with_adjust = pipeline.run()

        # with adjust for small sample using t, result is more conservative
        # i.e. smaller p-value and narrower CI
        assert (
            result_without_adjust.analysis_result["p_value"].iloc[1]
            <= result_with_adjust.analysis_result["p_value"].iloc[1]
        )
        assert (
            result_with_adjust.analysis_result["confidence_interval"].iloc[1][0]
            < result_without_adjust.analysis_result["confidence_interval"].iloc[1][0]
            < result_without_adjust.analysis_result["confidence_interval"].iloc[1][1]
            < result_with_adjust.analysis_result["confidence_interval"].iloc[1][1]
        )

        # test dict output
        data, config = get_real_diff_in_diff_input()
        pipeline = DiffinDiffPipeline(data, config)
        result = pipeline.run(output_format="dict")
        assert len(result.matching_result[Constants.JSON_MATCHING_RESULT]) > 0
        assert len(result.analysis_result[Constants.METRIC_RESULTS]) > 0

        # test json output
        data, config = get_real_diff_in_diff_input()
        pipeline = DiffinDiffPipeline(data, config)
        result = pipeline.run(output_format="json")
        result_dict = json.loads(result)
        assert len(result_dict[Constants.JSON_MATCHING_RESULT]) > 0
        assert result_dict[Constants.JSON_MATCHING_METHOD] == "correlation"
        assert result_dict[Constants.JSON_MATCHING_COLUMN_NAME] == "applied_submarket_id"
        assert len(result_dict[Constants.METRIC_RESULTS]) > 0

    def test_validation_variation_dtype(self):
        data, config = get_ab_int_group_input()
        pipeline = ABPipeline(data, config)
        result = pipeline.run()
        exp_group = pipeline.config.experiment_groups[0]
        assert type(exp_group.control.variation_name) == float
        assert result.shape[0] == 2
        data["group"] = data["group"].astype(int)
        config["group"] = {
            "column_type": "experiment_group",
            "control_label": 0.0,
            "variations": [0.0, 1.0],
        }
        pipeline = ABPipeline(data, config)
        exp_group = pipeline.config.experiment_groups[0]
        result = pipeline.run()
        assert type(exp_group.control.variation_name) == int
        assert result.shape[0] == 2

    def test_ab_pipeline_with_basic_fitter(self, ab_test_data, ab_test_config):
        """
        test without covariates w/ cluster
        """
        pipeline = ABPipeline(ab_test_data, ab_test_config)
        result = pipeline.run()
        assert result.shape[0] == 12
        asap = result[(result.metric_name == "asap") & (result.variation_name == 1)]
        asap_control = result[(result.metric_name == "asap") & (result.variation_name == 0)]
        asap_control["sample_size"].iloc[0] == 892
        asap_control["sample_size"].iloc[1] == 892
        assert round(asap["SE"].iloc[0], 2) == round(asap["SE"].iloc[1], 2)
        assert asap["sample_size"].iloc[0] == 852
        assert asap["sample_size"].iloc[1] == 852
        assert round(asap["p_value"].iloc[0], 2) == round(asap["p_value"].iloc[1], 2)
        assert round(asap["average_treatment_effect"].iloc[0], 2) == round(asap["average_treatment_effect"].iloc[1], 2)
        """
        test without covariates w/o cluster
        """
        ab_test_config["columns"].pop("bucket_key")
        pipeline = ABPipeline(ab_test_data, ab_test_config)
        result = pipeline.run()
        assert result.shape[0] == 12
        asap_control = result[(result.metric_name == "asap") & (result.variation_name == 0)]
        asap_control["sample_size"].iloc[0] == 1017
        asap_control["sample_size"].iloc[1] == 1017
        asap = result[(result.metric_name == "asap") & (result.variation_name == 1)]
        assert round(asap["SE"].iloc[0], 2) == round(asap["SE"].iloc[1], 2)
        assert asap["sample_size"].iloc[0] == 983
        assert asap["sample_size"].iloc[1] == 983
        assert round(asap["p_value"].iloc[0], 2) == round(asap["p_value"].iloc[1], 2)
        assert round(asap["average_treatment_effect"].iloc[0], 2) == round(asap["average_treatment_effect"].iloc[1], 2)
        """
        test with continuous covariates
        """
        # continuous covariate
        df = generate_ab_data(
            group_mean=10,
            group_std=5,
            ar_cor=0.8,
            total_group=700,
            time_length=30,
            treatment_effect=1,
        )

        # construct a covariate
        df["d2r"] = np.random.normal(40, 20, df.shape[0])
        df["asap"] = df["asap"] + df["d2r"]

        # adjust d2r to become orthogonal with centralized exp_group
        df.loc[df.shape[0] - 1, "d2r"] = (
            df.loc[df.shape[0] - 1, "d2r"]
            + df[df["is_treatment_group"] == 1]["d2r"].sum()
            - df[df["is_treatment_group"] == 0]["d2r"].sum()
        )

        smf_fitter_config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                },
                "d2r": {"column_type": "covariate", "value_type": "numerical"},
                "is_treatment_group": {
                    "column_type": "experiment_group",
                    "control_label": 0,
                    "variations": [0, 1],
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }

        delta_config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "fitter_type": "basic",
                },
                "d2r": {"column_type": "covariate", "value_type": "numerical"},
                "is_treatment_group": {
                    "column_type": "experiment_group",
                    "control_label": 0,
                    "variations": [0, 1],
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }

        delta_pipeline = ABPipeline(df, delta_config)
        delta_result = delta_pipeline.run()

        smf_pipeline = ABPipeline(df, smf_fitter_config)
        smf_result = smf_pipeline.run()

        # the treatment effects are supposed to exactly the same after orthogonalization
        # the SE should be almost the same up to the difference of degree of freedom adjustment
        assert (
            np.abs(delta_result.iloc[1]["average_treatment_effect"] - smf_result.iloc[1]["average_treatment_effect"])
            <= 1e-5
        )
        assert np.abs(delta_result.iloc[1]["SE"] - smf_result.iloc[1]["SE"]) <= 0.01
        assert np.abs(delta_result.iloc[1]["p_value"] - smf_result.iloc[1]["p_value"]) <= 0.0001

        """
        test with categorical and continuous covariates
        """

        # construct a covariate
        plant_list = ["clover", "lily", "lavender"] * (int(df.shape[0] / 3))
        df["plant"] = plant_list
        df["asap"] = (
            df["asap"] + (df["plant"] == "clover") * 4 - (df["plant"] == "clover") * 3 + (df["plant"] == "clover") * 7
        )

        smf_config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                },
                "plant": {"column_type": "covariate", "value_type": "categorical"},
                "d2r": {"column_type": "covariate", "value_type": "numerical"},
                "is_treatment_group": {
                    "column_type": "experiment_group",
                    "control_label": 0,
                    "variations": [0, 1],
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }

        delta_config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                    "fitter_type": "basic",
                },
                "plant": {"column_type": "covariate", "value_type": "categorical"},
                "d2r": {"column_type": "covariate", "value_type": "numerical"},
                "is_treatment_group": {
                    "column_type": "experiment_group",
                    "control_label": 0,
                    "variations": [0, 1],
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }

        delta_pipeline = ABPipeline(df, delta_config)
        delta_result = delta_pipeline.run()

        smf_pipeline = ABPipeline(df, smf_config)
        smf_result = smf_pipeline.run()

        assert (
            np.abs(delta_result.iloc[1]["average_treatment_effect"] - smf_result.iloc[1]["average_treatment_effect"])
            <= 1e-4
        )
        assert np.abs(delta_result.iloc[1]["SE"] - smf_result.iloc[1]["SE"]) <= 1e-2
        assert np.abs(delta_result.iloc[1]["p_value"] - smf_result.iloc[1]["p_value"]) <= 1e-4

    def test_ab_pipeline_interaction_json(self):
        asap = np.random.normal(100, 10, 10000)
        exp_group_1 = np.random.choice([0, 1], p=[0.5, 0.5], size=10000)
        exp_group_2 = np.random.choice([0, 1], p=[0.5, 0.5], size=10000)
        df = pd.DataFrame({"exp_group_1": exp_group_1, "exp_group_2": exp_group_2, "asap": asap})
        config = {
            "columns": {
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                },
                "exp_group_1": {
                    "column_type": "experiment_group",
                    "variations": [0, 1],
                    "variation_splits": [0.5, 0.5],
                    "control_label": 0,
                },
                "exp_group_2": {
                    "column_type": "experiment_group",
                    "variations": [0, 1],
                    "variation_splits": [0.5, 0.5],
                    "control_label": 0,
                },
            },
            "experiment_settings": {"type": "ab", "interaction": "True"},
        }
        pl = ABPipeline(df, config)
        json_result = pl.run(output_format="json")
        result = json.loads(json_result)
        metric_result = result["metric_results"][0]["experiment_group_results"]
        metric_result[0]["agg_func_results"][0]["control_results"][0]
        metric_result[0]["agg_func_results"][0]["treatment_results"][0]
        metric_result[1]["agg_func_results"][0]["control_results"][0]
        metric_result[1]["agg_func_results"][0]["treatment_results"][0]
        assert len(metric_result[2]["agg_func_results"][0]["control_results"]) == 0
        exp_group_12_trt = metric_result[2]["agg_func_results"][0]["treatment_results"][0]
        assert exp_group_12_trt["value"] is None
        assert exp_group_12_trt["sample_size"] is None
        assert exp_group_12_trt["p_value"]
        assert exp_group_12_trt["average_treatment_effect"]
        assert exp_group_12_trt["relative_average_treatment_effect"]
        assert exp_group_12_trt["abs_confidence_interval"]
        assert exp_group_12_trt["rel_confidence_interval"]

    def test_missing_values_input(self, data_with_missing_value):
        config = {
            "columns": {
                "is_treatment_group": {
                    "column_type": "experiment_group",
                    "control_label": 0,
                    "variations": [0, 1],
                    "variations_split": [0.5, 0.5],
                },
                "asap": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                },
                "ar_error": {
                    "column_type": "covariate",
                    "value_type": "numerical",
                },
                "d2r": {
                    "column_type": "metric",
                    "metric_type": "continuous",
                    "metric_aggregate_func": "mean",
                },
                "group": {
                    "column_type": "cluster",
                },
            },
            "experiment_settings": {
                "type": "ab",
            },
        }
        pipeline = ABPipeline(data_with_missing_value, config)
        result = pipeline.run()
        assert result.shape == (2, 11)

    def test_sequential_testing_should_run_successfully(self, ab_test_data, ab_test_sequential_config):
        pipeline = ABPipeline(ab_test_data, ab_test_sequential_config)
        result = pipeline.run(output_format="json")
        result_dict = json.loads(result)
        pipeline.config.metrics[0].sequential_testing_tau == 1.2
        treatment_result = result_dict["metric_results"][0]["experiment_group_results"][0]["agg_func_results"][0][
            "treatment_results"
        ][0]
        assert treatment_result[Constants.SEQUENTIAL_P_VALUE] is not None
        assert treatment_result[Constants.SEQUENTIAL_P_VALUE] <= 1
        assert treatment_result[Constants.SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL] is not None

    def test_group_sequential_testing_should_run_successfully(self, ab_test_data, ab_test_sequential_config):
        ab_test_sequential_config["experiment_settings"]["information_rates"] = [0.1, 0.5, 1.0]
        ab_test_sequential_config["experiment_settings"]["target_sample_size"] = 20000
        pipeline = ABPipeline(ab_test_data, ab_test_sequential_config)
        result = pipeline.run(output_format="json")
        result_dict = json.loads(result)
        treatment_result = result_dict["metric_results"][0]["experiment_group_results"][0]["agg_func_results"][0][
            "treatment_results"
        ][0]
        assert treatment_result[Constants.SEQUENTIAL_P_VALUE] is not None
        assert treatment_result[Constants.SEQUENTIAL_P_VALUE] <= 1
        assert treatment_result[Constants.SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL] is not None

    def test_lowmem_pipeline_with_basic_fitter(self, ab_test_data, ab_test_config):
        with tempfile.TemporaryDirectory() as temp_dir:
            """
            test without covariates w/ cluster
            """
            folder = pathlib.Path(temp_dir)
            ab_test_data.iloc[: ab_test_data.shape[0] // 2, :].to_parquet(folder / "raw_data.parquet")
            ab_test_data.iloc[ab_test_data.shape[0] // 2 :, :].to_parquet(folder / "raw_data2.parquet")
            ab_test_config["columns"]["bucket_key"] = [
                {"column_type": "cluster"},
                {"column_type": "experiment_randomize_unit"},
            ]
            data_loader = DataLoader(data_folder=folder)
            pipeline = ABPipeline(data_loader, ab_test_config)
            result = pipeline.run()
            assert result.shape[0] == 12
            asap = result[(result.metric_name == "asap") & (result.variation_name == 1)]
            assert round(asap["SE"].iloc[0], 2) == round(asap["SE"].iloc[1], 2)
            assert round(asap["p_value"].iloc[0], 2) == round(asap["p_value"].iloc[1], 2)
            assert round(asap["average_treatment_effect"].iloc[0], 2) == round(
                asap["average_treatment_effect"].iloc[1], 2
            )

            """
            test without covariates w/o cluster
            """
            ab_test_config["columns"].pop("bucket_key")
            pipeline = ABPipeline(data_loader, ab_test_config)
            result = pipeline.run()
            assert result.shape[0] == 12
            asap = result[(result.metric_name == "asap") & (result.variation_name == 1)]
            assert round(asap["SE"].iloc[0], 2) == round(asap["SE"].iloc[1], 2)
            assert round(asap["p_value"].iloc[0], 2) == round(asap["p_value"].iloc[1], 2)
            assert round(asap["average_treatment_effect"].iloc[0], 2) == round(
                asap["average_treatment_effect"].iloc[1], 2
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            """
            test with continuous covariates
            """
            # continuous covariate
            df = generate_ab_data(
                group_mean=10,
                group_std=5,
                ar_cor=0.8,
                total_group=700,
                time_length=30,
                treatment_effect=1,
            )

            # construct a covariate
            df["d2r"] = np.random.normal(40, 20, df.shape[0])
            df["asap"] = df["asap"] + df["d2r"]

            # adjust d2r to become orthogonal with centralized exp_group
            df.loc[df.shape[0] - 1, "d2r"] = (
                df.loc[df.shape[0] - 1, "d2r"]
                + df[df["is_treatment_group"] == 1]["d2r"].sum()
                - df[df["is_treatment_group"] == 0]["d2r"].sum()
            )

            # stash data away in a parquet file and store in data-loader format.
            folder = pathlib.Path(temp_dir)
            df.iloc[: df.shape[0] // 2, :].to_parquet(folder / "raw_data.parquet")
            df.iloc[df.shape[0] // 2 :, :].to_parquet(folder / "raw_data2.parquet")
            data_loader = DataLoader(data_folder=folder)

            smf_fitter_config = {
                "columns": {
                    "asap": {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                    },
                    "d2r": {"column_type": "covariate", "value_type": "numerical"},
                    "is_treatment_group": {
                        "column_type": "experiment_group",
                        "control_label": 0,
                        "variations": [0, 1],
                    },
                },
                "experiment_settings": {
                    "type": "ab",
                },
            }

            lowmem_config = {
                "columns": {
                    "asap": {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "fitter_type": "basic",
                    },
                    "d2r": {"column_type": "covariate", "value_type": "numerical"},
                    "is_treatment_group": {
                        "column_type": "experiment_group",
                        "control_label": 0,
                        "variations": [0, 1],
                    },
                },
                "experiment_settings": {
                    "type": "ab",
                },
            }

            lowmem_pipeline = ABPipeline(data_loader, lowmem_config)
            lowmem_result = lowmem_pipeline.run()

            smf_pipeline = ABPipeline(df, smf_fitter_config)
            smf_result = smf_pipeline.run()

            # the treatment effects are supposed to exactly the same after orthogonalization
            # the SE should be almost the same up to the difference of degree of freedom adjustment
            assert (
                np.abs(
                    lowmem_result.iloc[1]["average_treatment_effect"] - smf_result.iloc[1]["average_treatment_effect"]
                )
                <= 1e-5
            )
            assert np.abs(lowmem_result.iloc[1]["SE"] - smf_result.iloc[1]["SE"]) <= 0.01
            assert np.abs(lowmem_result.iloc[1]["p_value"] - smf_result.iloc[1]["p_value"]) <= 0.0001

        with tempfile.TemporaryDirectory() as temp_dir:
            """
            test with categorical and continuous covariates
            """

            # construct a covariate
            plant_list = ["clover", "lily", "lavender"] * (int(df.shape[0] / 3))
            df["plant"] = plant_list
            df["asap"] = (
                df["asap"]
                + (df["plant"] == "clover") * 4
                - (df["plant"] == "clover") * 3
                + (df["plant"] == "clover") * 7
            )

            # stash data away in a parquet file and store in data-loader format.
            folder = pathlib.Path(temp_dir)
            df.iloc[: df.shape[0] // 2, :].to_parquet(folder / "raw_data.parquet")
            df.iloc[df.shape[0] // 2 :, :].to_parquet(folder / "raw_data2.parquet")
            data_loader = DataLoader(data_folder=folder)

            smf_config = {
                "columns": {
                    "asap": {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                    },
                    "plant": {"column_type": "covariate", "value_type": "categorical"},
                    "d2r": {"column_type": "covariate", "value_type": "numerical"},
                    "is_treatment_group": {
                        "column_type": "experiment_group",
                        "control_label": 0,
                        "variations": [0, 1],
                    },
                },
                "experiment_settings": {
                    "type": "ab",
                },
            }

            lowmem_config = {
                "columns": {
                    "asap": {
                        "column_type": "metric",
                        "metric_type": "continuous",
                        "metric_aggregate_func": "mean",
                        "fitter_type": "basic",
                    },
                    "plant": {"column_type": "covariate", "value_type": "categorical"},
                    "d2r": {"column_type": "covariate", "value_type": "numerical"},
                    "is_treatment_group": {
                        "column_type": "experiment_group",
                        "control_label": 0,
                        "variations": [0, 1],
                    },
                },
                "experiment_settings": {
                    "type": "ab",
                },
            }

            lowmem_pipeline = ABPipeline(data_loader, lowmem_config)
            lowmem_result = lowmem_pipeline.run()

            smf_pipeline = ABPipeline(df, smf_config)
            smf_result = smf_pipeline.run()

            assert (
                np.abs(
                    lowmem_result.iloc[1]["average_treatment_effect"] - smf_result.iloc[1]["average_treatment_effect"]
                )
                <= 1e-4
            )
            assert np.abs(lowmem_result.iloc[1]["SE"] - smf_result.iloc[1]["SE"]) <= 1e-2
            assert np.abs(lowmem_result.iloc[1]["p_value"] - smf_result.iloc[1]["p_value"]) <= 1e-4

    def test_lowmem_quantile_pipeline(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            data, config = get_quantile_test_input()
            folder = pathlib.Path(temp_dir)
            data.iloc[: data.shape[0] // 2, :].to_parquet(folder / "raw_data.parquet")
            data.iloc[data.shape[0] // 2 :, :].to_parquet(folder / "raw_data2.parquet")
            data_loader = DataLoader(data_folder=folder)

            pipeline = ABPipeline(data_loader, config)
            result = pipeline.run()
            assert len(result) == 2
            trt = result[result["variation_name"] == "treatment"]
            ate = trt.iloc[0]["average_treatment_effect"]
            assert ate == -2.8

    def test_lowmem_ratio_metric_pipeline(self):

        with tempfile.TemporaryDirectory() as temp_dir:
            data, config = get_ratio_test_input(iterations=100)
            folder = pathlib.Path(temp_dir)
            data.iloc[: data.shape[0] // 2, :].to_parquet(folder / "raw_data.parquet")
            data.iloc[data.shape[0] // 2 :, :].to_parquet(folder / "raw_data2.parquet")
            data_loader = DataLoader(data_folder=folder)

            # basic fitter (delta method)
            pipeline = ABPipeline(data_loader, config)
            result = pipeline.run()
            assert len(result) == 2
            trt = result[result["variation_name"] == "treatment"]
            ate = trt.iloc[0]["average_treatment_effect"]
            assert round(ate, 2) == 1.86

            # boostrap fitter
            config["columns"]["a_ratio_metric"]["fitter_type"] = "bootstrap"
            pipeline = ABPipeline(data_loader, config)
            result = pipeline.run()
            trt = result[result["variation_name"] == "treatment"]
            ate = trt.iloc[0]["average_treatment_effect"]
            assert len(result) == 2
            assert round(ate, 2) == 1.86
