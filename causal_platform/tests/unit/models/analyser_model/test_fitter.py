"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import math

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from causal_platform.src.models.analyser_model.fitters.basic_fitter import BasicFitter
from causal_platform.src.models.analyser_model.fitters.bootstrap_fitter import (
    QuantileBootstrapFitter,
    RatioBootstrapFitter,
)
from causal_platform.src.models.analyser_model.fitters.fitter import SMFFitter
from causal_platform.src.models.configuration_model.base_objects import (
    Column,
    ColumnType,
    Covariate,
    CovariateType,
    ExperimentGroup,
    ExperimentVariation,
    Metric,
    MetricAggregateFunc,
    MetricType,
)
from causal_platform.src.utils.config_utils import set_experiment_config
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.delta_method import get_delta_ratio_covariance
from causal_platform.src.utils.error import InputConfigError, InputDataError
from causal_platform.tests.unit.models.analyser_model.test_data import ABTestBase


class TestSMFFitter(ABTestBase):
    @pytest.fixture
    def fitter(self, data, metrics, experiment_groups, covariates, interactions, cluster_column):
        data.columns = map(str.lower, data.columns)
        fitter = SMFFitter(
            data,
            metrics[0],
            experiment_groups,
            covariates,
            interactions,
            cluster_column,
        )
        return fitter

    def test_fitter_with_interaction(self, data, metrics, experiment_groups, covariates, cluster_column):
        data.columns = map(str.lower, data.columns)
        for metric in metrics:
            fitter_crse = SMFFitter(
                data,
                metric,
                experiment_groups,
                covariates,
                cluster=cluster_column,
                interaction=True,
            )
            fitter_ols = SMFFitter(data, metric, experiment_groups, covariates, cluster=None, interaction=True)
            fitter_crse.fit()
            fitter_ols.fit()

            result_crse = fitter_crse.get_analysis_results()
            result_ols = fitter_ols.get_analysis_results()

            assert result_crse[0].estimated_treatment_effect == result_ols[0].estimated_treatment_effect

            result_iteraction_crse = fitter_crse.get_two_way_interaction_results()
            result_iteraction_ols = fitter_crse.get_two_way_interaction_results()
            assert [r.estimated_treatment_effect for r in result_iteraction_crse] == [
                r.estimated_treatment_effect for r in result_iteraction_ols
            ]

    def test_fitter_wo_interaction(self, data, metrics, experiment_groups, covariates, cluster_column):
        data.columns = map(str.lower, data.columns)
        for metric in metrics:
            fitter_crse = SMFFitter(
                data,
                metric,
                experiment_groups,
                covariates,
                cluster=cluster_column,
                interaction=False,
            )
            fitter_ols = SMFFitter(data, metric, experiment_groups, covariates, interaction=False)

            fitter_crse.fit()
            fitter_ols.fit()

            result_crse = fitter_crse.get_analysis_results()
            result_ols = fitter_ols.get_analysis_results()

            assert result_crse[0].estimated_treatment_effect == result_ols[0].estimated_treatment_effect

    def test_get_ols_model_result_key_list(
        self,
        covariates_map,
        experiment_groups_map,
        fitter,
    ):
        assert fitter._get_ols_model_result_key_list(covariates_map["flf"]) == [("flf", None)]
        assert fitter._get_ols_model_result_key_list(covariates_map["sp_id"]) == [
            ("sp_id[T.16]", 16),
            ("sp_id[T.20]", 20),
            ("sp_id[T.401]", 401),
            ("sp_id[T.2109]", 2109),
        ]

        assert fitter._get_ols_model_result_key_list(experiment_groups_map["GROUP1"]) == [
            (
                "C(group1, Treatment('control'))[T.treatment]",
                experiment_groups_map["GROUP1"].treatments[0],
            )
        ]

        assert fitter._get_ols_model_result_key_list(experiment_groups_map["GROUP2"]) == [
            ("C(group2, Treatment(0))[T.1]", experiment_groups_map["GROUP2"].treatments[0])
        ]

    def test_generate_formula_from_columns(self, data, metrics, experiment_groups, covariates, fitter):
        fitter = SMFFitter(data, metrics[1], experiment_groups, covariates)
        assert (
            fitter._generate_formula_from_columns()
            == "dat ~ C(group1, Treatment('control')) + C(group2, Treatment(0)) + flf + C(sp_id)"
        )
        fitter = SMFFitter(data, metrics[1], experiment_groups, covariates, interaction=True)
        assert (
            fitter._generate_formula_from_columns()
            == "dat ~ C(group1, Treatment('control'))*C(group2, Treatment(0)) + flf + C(sp_id)"
        )

    def test_get_formula_for_column(self, fitter):
        metric = Metric(
            "metric",
            MetricType.continuous,
            MetricAggregateFunc.mean,
            False,
            False,
            False,
        )
        assert fitter._get_formula_element_from_column(metric) == "metric"

        exp_grp = ExperimentGroup("exp_grp", control=ExperimentVariation("control", 0.5))
        assert fitter._get_formula_element_from_column(exp_grp) == "C(exp_grp, Treatment('control'))"

        cat_cov = Covariate("cat_cov", CovariateType.categorial)
        assert fitter._get_formula_element_from_column(cat_cov) == "C(cat_cov)"

        numerical_cov = Covariate("numerical_cov", CovariateType.numerical)
        assert fitter._get_formula_element_from_column(numerical_cov) == "numerical_cov"

    def test_get_formula_for_interaction(self, fitter):
        exp_grps = tuple(
            ExperimentGroup("exp_grp_{}".format(i), ExperimentVariation("control_{}".format(i), 0.5)) for i in range(2)
        )
        assert fitter._get_formula_element_from_interaction(exp_grps) == "{}*{}".format(
            "C(exp_grp_0, Treatment('control_0'))",
            "C(exp_grp_1, Treatment('control_1'))",
        )

        cat_covs = tuple(Covariate("cat_cov_{}".format(i), CovariateType.categorial) for i in range(2))
        assert fitter._get_formula_element_from_interaction(cat_covs) == "{}*{}".format("C(cat_cov_0)", "C(cat_cov_1)")

        numerical_covs = tuple(Covariate("numerical_cov_{}".format(i), CovariateType.numerical) for i in range(2))
        assert fitter._get_formula_element_from_interaction(numerical_covs) == "{}*{}".format(
            "numerical_cov_0", "numerical_cov_1"
        )

    def test_demean_fixed_effect(
        self,
        data,
        metrics,
        experiment_groups,
        experiment_group1,
        experiment_group2,
        covariates,
        interactions,
        fitter,
    ):
        from causal_platform.src.utils.logger import logger

        logger.info(covariates)
        # experiment group with integer value
        fitter = SMFFitter(
            data,
            metrics[0],
            [experiment_group2],
            covariates,
            fixed_effect_estimator=True,
        )
        assert fitter.formula == "asap_{0} ~ group2_1_{0} + flf_{0}".format(
            Constants.FIXED_EFFECT_DEMEAN_COLUMN_POSTFIX
        )
        fitter.fit()
        analysis_results = fitter.get_analysis_results()
        assert len(analysis_results) == 1
        assert analysis_results[0].experiment_group.column_name == "group2"

        # experiment group with str values
        fitter = SMFFitter(
            data,
            metrics[0],
            [experiment_group1],
            covariates,
            fixed_effect_estimator=True,
        )
        assert fitter.formula == "asap_{0} ~ group1_treatment_{0} + flf_{0}".format(
            Constants.FIXED_EFFECT_DEMEAN_COLUMN_POSTFIX
        )
        fitter.fit()
        analysis_results = fitter.get_analysis_results()
        assert len(analysis_results) == 1
        assert analysis_results[0].experiment_group.column_name == "group1"

        # assert error when use demean with interactions
        with pytest.raises(InputConfigError) as err:
            fitter = SMFFitter(
                data,
                metrics[0],
                [experiment_group1],
                covariates,
                interactions,
                fixed_effect_estimator=True,
            )
        assert "interaction effect" in str(err.value)


class TestQuantileBootstrapFitter(ABTestBase):
    @pytest.fixture
    def config(self, data, quantile_config):
        config = set_experiment_config(quantile_config)
        return config

    def test_bootstrap_t(self, data, config):
        data.columns = map(str.lower, data.columns)
        metric = Metric(
            "ASAP",
            MetricType.continuous,
            MetricAggregateFunc.quantile,
            False,
            False,
            False,
            quantile=0.95,
        )
        fitter = QuantileBootstrapFitter(
            data=data,
            metric=metric,
            experiment_groups=config.experiment_groups,
            cluster=config.cluster,
            method=Constants.BOOTSTRAP_T,
            iteration=2,
            bootstrap_size=10,
        )
        fitter.fit()
        analysis_result = fitter.get_analysis_results()
        assert len(analysis_result) == 1

    def test_bootstrap_se(self, data, config):
        data.columns = map(str.lower, data.columns)
        metric = Metric(
            "ASAP",
            MetricType.continuous,
            MetricAggregateFunc.quantile,
            False,
            False,
            False,
            quantile=0.95,
        )
        fitter = QuantileBootstrapFitter(
            data=data,
            metric=metric,
            experiment_groups=config.experiment_groups,
            cluster=config.cluster,
            method=Constants.BOOTSTRAP_SE,
            iteration=10,
            bootstrap_size=10,
        )
        fitter.fit()
        analysis_result = fitter.get_analysis_results()
        assert len(analysis_result) == 1


class TestRatioBootstrapFitter(ABTestBase):
    def get_config(self):
        raw_data = [
            ["2019-01-01", 1, 0, "treatment", "ab_c"],
            ["2019-01-02", 1, 0, "control", "ab_c"],
            ["2019-01-02", 1, 0, "treatment", "ab_b"],
            ["2019-01-02", 1, 0, "control", "ab_a"],
            ["2019-01-02", 1, 0, "control", "ab_c"],
            ["2019-01-02", 1, 0, "treatment", "ab_a"],
            ["2019-01-02", 1, 0, "treatment", "ab_a"],
            ["2019-01-02", 1, 0, "control", "ab_a"],
            ["2019-01-02", 2, 0, "control", "ab_c"],
            ["2019-01-02", 1, 0, "control", "ab_c"],
        ]

        data = pd.DataFrame(raw_data, columns=["date", "numerator", "denominator", "group", "cluster"])
        data.date = pd.to_datetime(data.date)

        config_dict = {
            "columns": {
                "group": {
                    "column_type": "experiment_group",
                    "control_label": "control",
                    "variations": ["control", "treatment"],
                    "variations_split": [0.5, 0.5],
                },
                "date": {"column_type": "date"},
                "a_ratio_metric": {
                    "column_type": "metric",
                    "metric_type": "ratio",
                    "numerator_column": "numerator",
                    "denominator_column": "denominator",
                },
            },
            "experiment_settings": {
                "is_check_imbalance": False,
                "is_check_flickers": False,
                "is_check_metric_type": False,
                "type": "ab",
            },
        }
        config = set_experiment_config(config_dict)
        return data, config

    def test_bootstrap_t(self):
        data, config = self.get_config()
        # test ratio denominator value is zero
        with pytest.raises(InputDataError):
            fitter = RatioBootstrapFitter(
                data=data,
                metric=config.metrics[0],
                experiment_groups=config.experiment_groups,
                cluster=config.cluster,
                method=Constants.BOOTSTRAP_T,
                iteration=2,
                bootstrap_size=5,
            )
            fitter.fit()

    def test_bootstrap_se(self):
        data, config = self.get_config()
        # test ratio denominator value is zero
        with pytest.raises(Exception):
            fitter = RatioBootstrapFitter(
                data=data,
                metric=config.metrics[0],
                experiment_groups=config.experiment_groups,
                cluster=config.cluster,
                method=Constants.BOOTSTRAP_SE,
                iteration=5,
                bootstrap_size=15,
            )
            fitter.fit()


class TestBasicFitter(ABTestBase):
    def test_continuous_metric(self, data, metrics, experiment_groups, covariates, interactions, cluster_column):
        asap_metric = metrics[0]
        exp_group = experiment_groups[0]
        data.columns = [col.lower() for col in data.columns]
        basic_fitter = BasicFitter(data, asap_metric, [exp_group])
        basic_result = basic_fitter.get_analysis_results()
        assert len(basic_result) == 1

        smf_fitter = SMFFitter(data, asap_metric, [exp_group])
        smf_fitter.fit()
        assert basic_fitter.data.shape[0] == smf_fitter.data.shape[0]
        assert basic_fitter.data.shape[1] <= smf_fitter.data.shape[1]
        assert basic_fitter.data.iloc[10]["asap"] == smf_fitter.data.iloc[10]["asap"]
        smf_results = smf_fitter.get_analysis_results()
        assert round(basic_result[0].estimated_treatment_effect, 2) == round(
            smf_results[0].estimated_treatment_effect, 2
        )

    def test_ratio_metric(self, data, metrics, experiment_groups):
        ratio_metric = Metric(
            column_name="ratio",
            metric_type=MetricType.ratio,
            metric_aggregate_func=MetricAggregateFunc.mean,
            numerator_column=Column(column_name="asap", column_type=ColumnType.metric),
            denominator_column=Column(column_name="dat", column_type=ColumnType.metric),
        )
        exp_group = experiment_groups[0]
        data.columns = [col.lower() for col in data.columns]
        fitter = BasicFitter(data, ratio_metric, [exp_group])
        result = fitter.get_analysis_results()
        assert len(result) == 1

    def test_cluster_ratio_metric(self, data, metrics, experiment_groups, cluster_column):
        ratio_metric = Metric(
            column_name="ratio",
            metric_type=MetricType.ratio,
            metric_aggregate_func=MetricAggregateFunc.mean,
            numerator_column=Column(column_name="asap", column_type=ColumnType.metric),
            denominator_column=Column(column_name="dat", column_type=ColumnType.metric),
        )
        exp_group = experiment_groups[0]
        data.columns = [col.lower() for col in data.columns]
        fitter = BasicFitter(data, ratio_metric, [exp_group], cluster_column)
        result = fitter.get_analysis_results()
        assert len(result) == 1

    def test_quantile_metric(self, data, experiment_groups, cluster_column):
        asap_metric = Metric("ASAP", MetricType.continuous, MetricAggregateFunc.quantile, quantile=0.6)
        exp_group = experiment_groups[0]
        data.columns = [col.lower() for col in data.columns]
        basic_fitter = BasicFitter(data, asap_metric, [exp_group])
        basic_result = basic_fitter.get_analysis_results()
        assert len(basic_result) == 1

        basic_fitter = BasicFitter(data, asap_metric, [exp_group], cluster=cluster_column)
        basic_result = basic_fitter.get_analysis_results()
        assert len(basic_result) == 1

    def test_cluster_error(self, data, experiment_groups):
        asap_metric = Metric("ASAP", MetricType.continuous, MetricAggregateFunc.mean)
        cluster = Column("unit_id", column_type=ColumnType.cluster)
        exp_group = experiment_groups[0]
        data.columns = [col.lower() for col in data.columns]
        basic_fitter = BasicFitter(data, asap_metric, [exp_group], cluster=cluster)
        basic_result = basic_fitter.get_analysis_results()
        assert len(basic_result) == 1

    def test_no_value_error(self, data_wo_control, experiment_groups):
        asap_metric = Metric("ASAP", MetricType.continuous, MetricAggregateFunc.mean)
        cluster = Column("unit_id", column_type=ColumnType.cluster)
        exp_group = experiment_groups[0]
        data_wo_control.columns = [col.lower() for col in data_wo_control.columns]
        basic_fitter = BasicFitter(data_wo_control, asap_metric, [exp_group], cluster=cluster)
        with pytest.raises(Exception):
            basic_fitter.get_analysis_results()

    def test_basic_fitter_interaction(self, ab_test_data: pd.DataFrame):
        ab_test_data["exp_group2"] = np.random.choice([0, 1], p=[0.5, 0.5], size=ab_test_data.shape[0])
        asap_metric = Metric("asap", MetricType.continuous, MetricAggregateFunc.mean)
        exp_group_1 = ExperimentGroup("exp_group", ExperimentVariation(0, 0.5), [ExperimentVariation(1, 0.5)])
        exp_group_2 = ExperimentGroup("exp_group2", ExperimentVariation(0, 0.5), [ExperimentVariation(1, 0.5)])
        # with interaction
        basic_fitter = BasicFitter(ab_test_data, asap_metric, [exp_group_1, exp_group_2], interaction=True)
        basic_fitter.fit()
        results_basic = basic_fitter.get_analysis_results()
        assert len(results_basic) == 3
        smf_fitter = SMFFitter(ab_test_data, asap_metric, [exp_group_1, exp_group_2], interaction=True)
        smf_fitter.fit()
        results_smf = smf_fitter.get_analysis_results()
        assert len(results_smf) == 3
        for i in np.arange(3):
            assert results_basic[i].estimated_treatment_effect - results_smf[i].estimated_treatment_effect < 0.001
            assert results_basic[i].se - results_smf[i].se < 0.2
            assert results_basic[i].metric_treatment_value == results_smf[i].metric_treatment_value
            assert results_basic[i].metric_control_value == results_smf[i].metric_control_value

        # no interaction
        basic_fitter = BasicFitter(ab_test_data, asap_metric, [exp_group_1, exp_group_2], interaction=False)
        basic_fitter.fit()
        results_basic = basic_fitter.get_analysis_results()
        assert len(results_basic) == 2
        smf_fitter = SMFFitter(ab_test_data, asap_metric, [exp_group_1, exp_group_2], interaction=False)
        smf_fitter.fit()
        results_smf = smf_fitter.get_analysis_results()
        assert len(results_smf) == 2
        for i in np.arange(2):
            assert results_basic[i].estimated_treatment_effect - results_smf[i].estimated_treatment_effect < 0.1
            assert results_basic[i].se - results_smf[i].se < 0.2
            assert results_basic[i].metric_treatment_value == results_smf[i].metric_treatment_value
            assert results_basic[i].metric_control_value == results_smf[i].metric_control_value
        # multiple variations
        ab_test_data["exp_group2"] = np.random.choice(
            [0, 1, 2, 3], p=[0.25, 0.25, 0.25, 0.25], size=ab_test_data.shape[0]
        )
        exp_group_2 = ExperimentGroup(
            "exp_group2",
            ExperimentVariation(0, 0.25),
            [
                ExperimentVariation(1, 0.25),
                ExperimentVariation(2, 0.25),
                ExperimentVariation(3, 0.25),
            ],
        )
        basic_fitter = BasicFitter(ab_test_data, asap_metric, [exp_group_1, exp_group_2], interaction=True)
        basic_fitter.fit()
        results_basic = basic_fitter.get_analysis_results()
        assert len(results_basic) == 7
        smf_fitter = SMFFitter(ab_test_data, asap_metric, [exp_group_1, exp_group_2], interaction=True)
        smf_fitter.fit()
        results_smf = smf_fitter.get_analysis_results()
        assert len(results_smf) == 7
        for i in np.arange(7):
            assert results_basic[i].estimated_treatment_effect - results_smf[i].estimated_treatment_effect < 0.01
            assert results_basic[i].se - results_smf[i].se < 0.5
            assert results_basic[i].metric_treatment_value == results_smf[i].metric_treatment_value
            assert results_basic[i].metric_control_value == results_smf[i].metric_control_value

    def test_delta_ratio_covariance(self, ab_test_data):
        cluster = Column("bucket_key", column_type=ColumnType.cluster)
        exp_group = ExperimentGroup("exp_group", ExperimentVariation(0, 0.5), [ExperimentVariation(1, 0.5)])
        ratio_metric = Metric(
            "ratio",
            metric_aggregate_func=MetricAggregateFunc.mean,
            metric_type=MetricType.ratio,
            numerator_column=Column("asap", ColumnType.metric),
            denominator_column=Column("dat", ColumnType.metric),
        )
        covariates_list = [
            Covariate(
                "tester",
                value_type=CovariateType.ratio,
                numerator_column=Column("asap", ColumnType.covariate),
                denominator_column=Column("pred_asap", ColumnType.covariate),
            )
        ]
        fitter = BasicFitter(ab_test_data, ratio_metric, [exp_group], cluster=cluster, covariates=covariates_list)
        fitter.fit()
        beta1 = fitter.processed_ratio_covariates[0].coef

        # calculate variance using other equation
        def alternative_ratio_variance_calculation(n, d):
            k = d.shape[0]
            cov_mat = np.cov(d, n, ddof=1)
            var_n_bar = cov_mat[1, 1] / k
            var_d_bar = cov_mat[0, 0] / k
            mu_n = n.mean()
            mu_d = d.mean()
            cov_nd_bar = cov_mat[0, 1] / k
            return ((mu_n / mu_d) ** 2) * (
                (var_n_bar / mu_n**2) - (2 * cov_nd_bar / (mu_n * mu_d)) + (var_d_bar / mu_d**2)
            )

        covariate_var_calculated_w_alternate_eq = alternative_ratio_variance_calculation(
            ab_test_data[fitter.processed_ratio_covariates[0].numerator_column.column_name],
            ab_test_data[fitter.processed_ratio_covariates[0].denominator_column.column_name],
        )

        metric_covariate_covar = get_delta_ratio_covariance(
            ab_test_data[fitter.metric.numerator_column.column_name],
            ab_test_data[fitter.metric.denominator_column.column_name],
            ab_test_data[fitter.processed_ratio_covariates[0].numerator_column.column_name],
            ab_test_data[fitter.processed_ratio_covariates[0].denominator_column.column_name],
        )
        beta2 = float(metric_covariate_covar / covariate_var_calculated_w_alternate_eq)

        # check whether two ways of calculating variance is same (thru comparing beta value)
        if fitter.use_iterative_cv_method:
            assert round(beta1, 4) == round(beta2, 4)

    def test_variance_reduction(self, ab_test_data):
        # categorical covariate
        asap_metric = Metric("asap", MetricType.continuous, MetricAggregateFunc.mean)
        cluster = Column("bucket_key", column_type=ColumnType.cluster)
        covariates = [Covariate("submarket_id", value_type=CovariateType.categorial)]
        exp_group = ExperimentGroup("exp_group", ExperimentVariation(0, 0.5), [ExperimentVariation(1, 0.5)])
        ab_test_data.columns = [col.lower() for col in ab_test_data.columns]
        basic_fitter = BasicFitter(ab_test_data, asap_metric, [exp_group], cluster=cluster, covariates=covariates)
        basic_fitter.fit()
        basic_result = basic_fitter.get_analysis_results()
        assert len(basic_result) == 1
        assert not np.isnan(basic_result[0].estimated_treatment_effect)
        assert not np.isnan(basic_result[0].p_value)
        # numerical covariate
        covariates = [Covariate("pred_asap", value_type=CovariateType.numerical)]
        asap_metric = Metric("asap", MetricType.continuous, MetricAggregateFunc.mean)
        basic_fitter = BasicFitter(ab_test_data, asap_metric, [exp_group], cluster=cluster, covariates=covariates)
        basic_fitter.fit()
        basic_result = basic_fitter.get_analysis_results()
        assert len(basic_result) == 1
        assert not np.isnan(basic_result[0].estimated_treatment_effect)
        assert not np.isnan(basic_result[0].p_value)
        # numerical covariate + categorical covariate
        covariates = [
            Covariate("pred_asap", value_type=CovariateType.numerical),
            Covariate("submarket_id", value_type=CovariateType.categorial),
        ]
        asap_metric = Metric("asap", MetricType.continuous, MetricAggregateFunc.mean)
        basic_fitter = BasicFitter(ab_test_data, asap_metric, [exp_group], cluster=cluster, covariates=covariates)
        basic_fitter.fit()
        basic_result = basic_fitter.get_analysis_results()
        assert len(basic_result) == 1
        assert not np.isnan(basic_result[0].estimated_treatment_effect)
        assert not np.isnan(basic_result[0].p_value)

        # numerical ratio metric
        ratio_metric = Metric(
            "ratio",
            metric_aggregate_func=MetricAggregateFunc.mean,
            metric_type=MetricType.ratio,
            numerator_column=Column("asap", ColumnType.metric),
            denominator_column=Column("dat", ColumnType.metric),
        )
        covariates_list = [
            Covariate(
                "tester",
                value_type=CovariateType.ratio,
                numerator_column=Column("asap", ColumnType.covariate),
                denominator_column=Column("pred_asap", ColumnType.covariate),
            )
        ]

        fitter = BasicFitter(ab_test_data, ratio_metric, [exp_group], cluster=cluster, covariates=covariates_list)
        fitter.fit()
        result = fitter.get_analysis_results()
        assert len(result) == 1
        assert len(fitter.processed_ratio_covariates) == 1  # ratio covariates extracted
        assert fitter.processed_ratio_covariates[0].coef is not None  # beta is actually set
        assert not np.isnan(result[0].estimated_treatment_effect)
        assert not np.isnan(result[0].p_value)

        # test smf (linear reg) with extracted beta of ratio
        ab_test_data["intercept"] = np.ones(ab_test_data.shape[0])
        ab_test_data["ratio_numerator_cov"] = np.random.normal(10, 1, ab_test_data.shape[0])
        basic_ratio_metric = Metric(
            "basic_ratio",
            metric_aggregate_func=MetricAggregateFunc.mean,
            metric_type=MetricType.ratio,
            numerator_column=Column("asap", ColumnType.metric),
            denominator_column=Column("intercept", ColumnType.metric),
        )

        covariates = [
            Covariate(
                "ratio_cov",
                CovariateType.ratio,
                numerator_column=Column("ratio_numerator_cov", ColumnType.covariate),
                denominator_column=Column("intercept", ColumnType.covariate),
            )
        ]

        fitter = BasicFitter(ab_test_data, basic_ratio_metric, [exp_group], cluster=cluster, covariates=covariates)
        fitter.fit()
        basic_result = basic_fitter.get_analysis_results()

        beta = fitter.processed_ratio_covariates[0].coef

        # extract optimal value with linear regression
        model = smf.ols("asap ~ ratio_numerator_cov + intercept", data=ab_test_data).fit()
        lin_reg_optim = model.params[1]

        assert abs(lin_reg_optim - beta) < 1e-4  # value from calculated beta and lin reg. should be same

        # test smf and basic equality
        ab_test_data["num_cov"] = np.random.normal(10, 1, ab_test_data.shape[0])
        ab_test_data["cat_cov"] = np.random.choice(
            [1, 2, 3, 4, 5], p=[0.2, 0.2, 0.2, 0.2, 0.2], size=ab_test_data.shape[0]
        )
        asap_metric = Metric("asap", MetricType.continuous, MetricAggregateFunc.mean)
        covariates = [
            Covariate("num_cov", CovariateType.numerical),
            Covariate("cat_cov", CovariateType.categorial),
        ]
        smf_fitter = SMFFitter(ab_test_data, asap_metric, [exp_group], cluster=cluster, covariates=covariates)
        smf_fitter.fit()
        smf_result = smf_fitter.get_analysis_results()
        basic_fitter = BasicFitter(ab_test_data, asap_metric, [exp_group], cluster=cluster, covariates=covariates)
        basic_fitter.fit()
        basic_result = basic_fitter.get_analysis_results()
        assert basic_result[0].estimated_treatment_effect - smf_result[0].estimated_treatment_effect < 0.1
        assert basic_result[0].p_value - smf_result[0].p_value < 0.1

    def test_get_covariate_results(self, causal_test_data):
        ratio_metric = Metric(
            "metric",
            metric_aggregate_func=MetricAggregateFunc.mean,
            metric_type=MetricType.ratio,
            numerator_column=Column("metric1", ColumnType.metric),
            denominator_column=Column("denominator", ColumnType.metric),
        )
        covariates_list = [
            Covariate(
                "cov1",
                value_type=CovariateType.ratio,
                numerator_column=Column("cov1", ColumnType.covariate),
                denominator_column=Column("denominator", ColumnType.covariate),
            )
        ]

        exp_group = ExperimentGroup(
            "exp_group", ExperimentVariation("control", 0.5), [ExperimentVariation("treatment", 0.5)]
        )
        basic_fitter = BasicFitter(
            causal_test_data, ratio_metric, [exp_group], covariates=covariates_list, use_iterative_cv_method=False
        )
        basic_fitter.fit()
        covariates_result = basic_fitter.get_covariate_results()
        assert abs(covariates_result[0].estimated_coefficient - (-2.959)) <= 0.01
        assert abs(covariates_result[0].se - 0.980) <= 0.01

    def test_msprt_p_value_is_conservative(self, ab_test_data):
        import scipy as sp
        from scipy.stats import norm

        asap_metric = Metric("asap", MetricType.continuous, MetricAggregateFunc.mean)
        cluster = Column("bucket_key", column_type=ColumnType.cluster)
        covariates = [Covariate("submarket_id", value_type=CovariateType.categorial)]
        exp_group = ExperimentGroup("exp_group", ExperimentVariation(0, 0.5), [ExperimentVariation(1, 0.5)])
        ab_test_data.columns = [col.lower() for col in ab_test_data.columns]
        basic_fitter = BasicFitter(ab_test_data, asap_metric, [exp_group], cluster=cluster, covariates=covariates)
        basic_fitter.fit()
        basic_result = basic_fitter.get_analysis_results()
        assert len(basic_result) == 1
        assert not np.isnan(basic_result[0].estimated_treatment_effect)
        assert not np.isnan(basic_result[0].p_value)

        x = norm.rvs(loc=0, scale=1.0, size=1000)

        p_fixed, _, _ = basic_fitter._compute_p_value_and_conf_int(np.mean(x), sp.stats.sem(x), is_fixed_horizon=True)
        p_seq, _, _ = basic_fitter._compute_p_value_and_conf_int(np.mean(x), sp.stats.sem(x), is_fixed_horizon=False)
        assert p_fixed <= p_seq

    def test_sequential_testing_with_tau_input(self, ab_test_data):
        asap_metric = Metric("asap", MetricType.continuous, MetricAggregateFunc.mean, sequential_testing_tau=1)
        cluster = Column("bucket_key", column_type=ColumnType.cluster)
        covariates = [Covariate("submarket_id", value_type=CovariateType.categorial)]
        exp_group = ExperimentGroup("exp_group", ExperimentVariation(0, 0.5), [ExperimentVariation(1, 0.5)])
        ab_test_data.columns = [col.lower() for col in ab_test_data.columns]
        basic_fitter = BasicFitter(ab_test_data, asap_metric, [exp_group], cluster=cluster, covariates=covariates)
        basic_fitter.fit()
        basic_results = basic_fitter.get_analysis_results()
        pval = basic_results[0].sequential_p_value
        confint0 = basic_results[0].sequential_confidence_interval_left
        confint1 = basic_results[0].sequential_confidence_interval_right
        assert pval and not math.isinf(pval)
        assert confint0 and not math.isinf(confint0)
        assert confint1 and not math.isinf(confint1)

        # should default tau to 1 and run if inputed tau = 0
        asap_metric = Metric("asap", MetricType.continuous, MetricAggregateFunc.mean, sequential_testing_tau=0)
        basic_fitter = BasicFitter(ab_test_data, asap_metric, [exp_group], cluster=cluster, covariates=covariates)
        basic_fitter.fit()
        basic_results = basic_fitter.get_analysis_results()
        pval = basic_results[0].sequential_p_value
        confint0 = basic_results[0].sequential_confidence_interval_left
        confint1 = basic_results[0].sequential_confidence_interval_right
        assert pval and not math.isinf(pval)
        assert confint0 and not math.isinf(confint0)
        assert confint1 and not math.isinf(confint1)
