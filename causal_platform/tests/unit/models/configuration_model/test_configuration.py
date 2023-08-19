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
    CheckImbalanceMethod,
    Column,
    ColumnType,
    Covariate,
    CovariateType,
    ExperimentGroup,
    ExperimentType,
    ExperimentVariation,
    Metric,
    MetricAggregateFunc,
    MetricType,
)
from causal_platform.src.models.configuration_model.config import AbConfig, BaseConfig
from causal_platform.src.pipeline.experiment_pipelines.ab_pipeline import ABPipeline
from causal_platform.src.pipeline.experiment_pipelines.diff_in_diff_pipeline import DiffinDiffPipeline
from causal_platform.src.utils.config_utils import set_experiment_config
from causal_platform.src.utils.error import InputConfigError, InputDataError
from causal_platform.tests.unit.data import (
    get_ab_int_group_input,
    get_config_with_customized_covariate,
    get_imbalance_check_input,
    get_ratio_test_input,
    get_real_diff_in_diff_input,
    get_test_input,
)


class TestConfiguration:
    def test_config_experiment_type(self):
        data, config = get_test_input()
        config["experiment_settings"]["type"] = "other"

        with pytest.raises(InputConfigError) as err:
            ABPipeline(data, config)
        assert "Experiment type 'other' is not valid" in str(err.value)

    def test_config_metric_type(self):
        data, config = get_test_input()
        config["columns"]["metric1"]["metric_type"] = "other"

        with pytest.raises(InputConfigError) as err:
            ABPipeline(data, config)
        assert "'other' is not a valid MetricType" in str(err.value)

    def test_config_covariate_type(self):
        data, config = get_test_input(use_covariate=True)
        config["columns"]["covariate1"]["value_type"] = "other"
        with pytest.raises(InputConfigError) as err:
            ABPipeline(data, config)
        assert "'other' is not a valid CovariateType" in str(err.value)

    def test_config_agg_func(self):
        data, config = get_test_input()
        config["columns"]["metric1"]["metric_aggregate_func"] = "other"
        with pytest.raises(InputConfigError) as err:
            ABPipeline(data, config)
        assert "'other' is not a valid MetricAggregateFunc" in str(err.value)

    def test_basic_config_init_metric_existence_success(self):
        # data, config = get_test_data()
        # set_experiment_config(data, config)
        config = BaseConfig(
            metrics=[
                Metric(
                    "metric",
                    MetricType.continuous,
                    MetricAggregateFunc.mean,
                    False,
                    False,
                    False,
                )
            ],
            covariates=[],
            experiment_type=ExperimentType.ab,
            experiment_randomize_units=[],
        )
        config._validate_config()

    def test_basic_config_init_metric_and_covariate_success(self):
        covariates_ratio = [
            Covariate(
                "covariate",
                CovariateType.ratio,
                numerator_column=Column(column_name="cov_num", column_type=ColumnType.ratio_covariate_component),
                denominator_column=Column(column_name="cov_den", column_type=ColumnType.ratio_covariate_component),
            )
        ]

        covariates_non_ratio = [
            Covariate(
                "covariate",
                CovariateType.numerical,
            )
        ]

        # ratio metric, ratio covariate
        config = BaseConfig(
            metrics=[
                Metric(
                    "metric",
                    MetricType.ratio,
                    MetricAggregateFunc.mean,
                    covariates=covariates_ratio,
                    numerator_column=Column(column_name="met_num", column_type=ColumnType.ratio_metric_component),
                    denominator_column=Column(column_name="met_den", column_type=ColumnType.ratio_metric_component),
                )
            ],
            covariates=covariates_ratio,
            experiment_type=ExperimentType.ab,
            experiment_randomize_units=[],
        )
        config._validate_config()

        # non-ratio metric, non-ratio covariate

        config = BaseConfig(
            metrics=[
                Metric(
                    "metric",
                    MetricType.continuous,
                    MetricAggregateFunc.mean,
                    covariates=covariates_non_ratio,
                )
            ],
            covariates=covariates_non_ratio,
            experiment_type=ExperimentType.ab,
            experiment_randomize_units=[],
        )
        config._validate_config()

    def test_basic_config_init_ratio_metric_non_ratio_covariate_raise_error(self):
        covariates = [
            Covariate(
                "covariate",
                CovariateType.ratio,
                numerator_column=Column(column_name="cov_num", column_type=ColumnType.ratio_covariate_component),
                denominator_column=Column(column_name="cov_den", column_type=ColumnType.ratio_covariate_component),
            ),
            Covariate(
                "covariate",
                CovariateType.numerical,
            ),
        ]

        # test ratio metric, non-ratio covariate
        with pytest.raises(InputConfigError) as err:
            config = BaseConfig(
                metrics=[
                    Metric(
                        "metric",
                        MetricType.ratio,
                        MetricAggregateFunc.mean,
                        covariates=covariates,
                        numerator_column=Column(column_name="met_num", column_type=ColumnType.ratio_metric_component),
                        denominator_column=Column(column_name="met_den", column_type=ColumnType.ratio_metric_component),
                    )
                ],
                covariates=covariates,
                experiment_type=ExperimentType.ab,
                experiment_randomize_units=[],
            )
            config._validate_config()
        assert "is a ratio metric that has covariate {} that is not also ratio type".format(
            covariates[1].column_name
        ) in str(err.value)

        # test non-ratio metric, ratio covariate
        with pytest.raises(InputConfigError) as err:
            config = BaseConfig(
                metrics=[
                    Metric(
                        "metric",
                        MetricType.continuous,
                        MetricAggregateFunc.mean,
                        covariates=covariates,
                    )
                ],
                covariates=covariates,
                experiment_type=ExperimentType.ab,
                experiment_randomize_units=[],
            )
            config._validate_config()
        assert "is not a ratio metric but has a covariate {} that is a ratio type".format(
            covariates[0].column_name
        ) in str(err.value)

    def test_config_with_customized_covar(self):
        (
            config_with_correct_applied_metrics,
            config_with_applied_metrics_covar_error,
            config_without_applied_metrics,
        ) = get_config_with_customized_covariate()

        assert set_experiment_config(config_with_correct_applied_metrics)

        with pytest.raises(InputConfigError) as err:
            set_experiment_config(config_with_applied_metrics_covar_error)
        assert "covariate pred_asap has an unrecognized metric name that is trying to apply" in str(err.value)

        assert set_experiment_config(config_without_applied_metrics)

    def test_diff_in_diff_config_success(self):
        data, config = get_real_diff_in_diff_input()
        DiffinDiffPipeline(data, config)

    def test_diff_in_diff_config_more_than_one_metric_error(self):
        with pytest.raises(InputConfigError):
            data, config = get_real_diff_in_diff_input()
            config["columns"]["2nd_metric"] = {
                "column_type": "metric",
                "metric_type": "continuous",
                "metric_aggregate_func": "mean",
            }
            DiffinDiffPipeline(data, config)

        with pytest.raises(InputConfigError):
            data, config = get_real_diff_in_diff_input()
            config["experiment_settings"]["matching_weights"] = [0.5, 0.6]
            DiffinDiffPipeline(data, config)

    def test_diff_in_diff_config_invalid_matching_weights_error(self):
        with pytest.raises(InputConfigError):
            data, config = get_real_diff_in_diff_input()
            config["experiment_settings"]["matching_weights"] = [-1]
            DiffinDiffPipeline(data, config)

        with pytest.raises(InputConfigError):
            data, config = get_real_diff_in_diff_input()
            config["experiment_settings"]["matching_weights"] = [0.5, 0.6]
            DiffinDiffPipeline(data, config)

    def test_diff_in_diff_config_matching_weights_error(self):
        # missing matching weights column
        with pytest.raises(InputDataError):
            data, config = get_real_diff_in_diff_input()
            config["experiment_settings"]["matching_columns"] = ["does_not_exist"]
            DiffinDiffPipeline(data, config)

    def test_diff_in_diff_config_date_column_not_date_error(self):
        with pytest.raises(InputConfigError):
            data, config = get_real_diff_in_diff_input()
            data["applied_date"] = 1
            DiffinDiffPipeline(data, config)

    def test_diff_in_diff_config_dates_out_of_order_error(self):
        with pytest.raises(InputConfigError):
            data, config = get_real_diff_in_diff_input()
            config["experiment_settings"]["matching_start_date"] = "2019-09-16"
            DiffinDiffPipeline(data, config)

        with pytest.raises(InputConfigError):
            data, config = get_real_diff_in_diff_input()
            config["experiment_settings"]["matching_end_date"] = "2019-09-20"
            DiffinDiffPipeline(data, config)

        with pytest.raises(InputConfigError):
            data, config = get_real_diff_in_diff_input()
            config["experiment_settings"]["experiment_start_date"] = "2019-09-22"
            DiffinDiffPipeline(data, config)

    def test_diff_in_diff_config_invalid_treatment_unit_count_error(self):
        with pytest.raises(InputDataError):
            data, config = get_real_diff_in_diff_input()
            config["experiment_settings"]["match_unit_size"] = 1000
            DiffinDiffPipeline(data, config)

        with pytest.raises(InputDataError):
            data, config = get_real_diff_in_diff_input()
            config["experiment_settings"]["exclude_unit_ids"] = range(100)
            DiffinDiffPipeline(data, config)

    def test_diff_in_diff_config_missing_data_error(self):
        # no data in matching period
        with pytest.raises(InputDataError):
            data, config = get_real_diff_in_diff_input()
            exp_start = config["experiment_settings"]["experiment_start_date"]
            data = data[data["applied_date"] >= exp_start]
            DiffinDiffPipeline(data, config)

    def test_diff_in_diff_config_invalid_matching_method(self):
        # no data in matching period
        with pytest.raises(InputConfigError):
            data, config = get_real_diff_in_diff_input()
            config["experiment_settings"]["matching_method"] = "invalid_method"
            DiffinDiffPipeline(data, config)

    def test_ratio_metric_config(self):
        # numerator/denominator not in data
        data, config_dict = get_ratio_test_input()
        data = data.drop("metric1", axis=1)
        pipeline = ABPipeline(data, config_dict)
        result = pipeline.run(output_format="dict")
        assert (
            "Denominator column 'metric1' for ratio metric 'a_ratio_metric' does not exist"
            in result["log_messages"]["errors"][0]
        )
        # successfully run through
        data, config_dict = get_ratio_test_input()
        config = set_experiment_config(config_dict)
        assert config.metrics[0].column_name == "a_ratio_metric"
        assert config.metrics[0].metric_type.name == "ratio"
        assert config.metrics[0].numerator_column.column_name == "metric2"
        assert config.metrics[0].denominator_column.column_name == "metric1"
        assert len(config.covariates) == 0
        assert config.metrics[0].covariates == []

        # not provide numerator or denominator
        with pytest.raises(InputConfigError) as err:
            Metric(
                column_name="ratio_metric",
                metric_type=MetricType.ratio,
                metric_aggregate_func=MetricAggregateFunc.mean,
            )
        assert "Please provide numerator and denominator columns" in str(err.value)

        with pytest.raises(InputConfigError) as err:
            Metric(
                column_name="ratio_metric",
                metric_type=MetricType.ratio,
                numerator_column=Column(
                    column_name="numerator",
                    column_type=ColumnType.ratio_metric_component,
                ),
                metric_aggregate_func=MetricAggregateFunc.mean,
            )
        assert "provide numerator" in str(err.value)

        with pytest.raises(InputConfigError) as err:
            data, config_dict = get_ratio_test_input()
            config_dict["columns"]["a_ratio_metric"] = {
                "column_type": "metric",
                "metric_type": "ratio",
            }
            config = set_experiment_config(config_dict)
        assert "A required field 'numerator_column' not found" in str(err.value)

    def test_ratio_covariate_config(self):
        # successfully run through
        data, config_dict = get_ratio_test_input(use_cov=True)
        config = set_experiment_config(config_dict)
        assert len(config.covariates) == 1
        assert len(config.metrics[0].covariates) == 1
        assert config.covariates[0].column_name == "a_ratio_covariate"
        assert config.covariates[0].value_type.name == "ratio"
        assert config.covariates[0].numerator_column.column_name == "covariate1"
        assert config.covariates[0].denominator_column.column_name == "covariate2"
        assert config.metrics[0].column_name == "a_ratio_metric"
        assert config.metrics[0].metric_type.name == "ratio"
        assert config.metrics[0].numerator_column.column_name == "metric2"
        assert config.metrics[0].denominator_column.column_name == "metric1"
        # not provide numerator or denominator
        with pytest.raises(InputConfigError) as err:
            Covariate(
                column_name="ratio_covariate",
                value_type=CovariateType.ratio,
            )
        assert "Please provide numerator and denominator columns" in str(err.value)

        with pytest.raises(InputConfigError) as err:
            Covariate(
                column_name="ratio_covariate",
                value_type=CovariateType.ratio,
                numerator_column=Column(
                    column_name="numerator",
                    column_type=ColumnType.ratio_covariate_component,
                ),
            )
        assert "provide numerator" in str(err.value)

        with pytest.raises(InputConfigError) as err:
            data, config_dict = get_ratio_test_input(use_cov=True)
            config_dict["columns"]["a_ratio_covars"] = {
                "column_type": "covariate",
                "value_type": "ratio",
            }
            config = set_experiment_config(config_dict)
        assert "A required field 'numerator_column' not found" in str(err.value)


class TestABConfiguration:
    def test_ab_config_init_valid_date_success(self):
        # data, config = get_test_data()
        # set_experiment_config(data, config)
        raw_data = [
            ["2019-01-01", 0.34, "treatment", "ddd"],
            ["2019-01-02", 0.13, "control", "ccc"],
        ]
        data = pd.DataFrame(raw_data, columns=["date", "metric", "group", "delivery"])
        data.date = pd.to_datetime(data.date)
        config = AbConfig(
            experiment_groups=[
                ExperimentGroup(
                    "group",
                    ExperimentVariation("control", 0.5),
                    [ExperimentVariation("treatment", 0.5)],
                )
            ],
            date=Column("date", ColumnType.date),
            metrics=[
                Metric(
                    "metric",
                    MetricType.continuous,
                    MetricAggregateFunc.mean,
                    False,
                    False,
                    False,
                )
            ],
            covariates=[],
            experiment_type=ExperimentType.ab,
            experiment_randomize_units=[],
        )
        config._validate_ab_config()

    def test_ab_config_init_valid_experiment_groups_success(self):
        # data, config = get_test_data()
        # set_experiment_config(data, config)
        raw_data = [
            ["2019-01-01", 0.34, "treatment", "ddd"],
            ["2019-01-02", 0.13, "control", "ccc"],
        ]
        data = pd.DataFrame(raw_data, columns=["date", "metric", "group", "delivery"])
        data.date = pd.to_datetime(data.date)
        config = AbConfig(
            experiment_groups=[
                ExperimentGroup(
                    "group",
                    ExperimentVariation("control", 0.5),
                    [ExperimentVariation("treatment", 0.5)],
                )
            ],
            date=Column("date", ColumnType.date),
            metrics=[
                Metric(
                    "metric",
                    MetricType.continuous,
                    MetricAggregateFunc.mean,
                    False,
                    False,
                    False,
                )
            ],
            covariates=[],
            experiment_type=ExperimentType.ab,
            experiment_randomize_units=[],
        )
        config._validate_ab_config()

    def test_ab_config_init_without_experiment_groups_raise_error(self):
        raw_data = [["2019-01-01", 0.34], ["2019-01-02", 0.13]]
        data = pd.DataFrame(raw_data, columns=["date", "metric"])
        config = AbConfig(
            experiment_groups=[
                ExperimentGroup(
                    "group",
                    ExperimentVariation("control", 0.5),
                    [ExperimentVariation("treatment", 0.5)],
                )
            ],
            metrics=[
                Metric(
                    "metric",
                    MetricType.continuous,
                    MetricAggregateFunc.mean,
                    False,
                    False,
                    False,
                )
            ],
            covariates=[],
            experiment_type=ExperimentType.ab,
            experiment_randomize_units=[],
        )
        with pytest.raises(InputDataError) as err:
            pl = ABPipeline(data, config)
            pl._validate()
        assert "Experiment_group column '{}' does not exist in data".format(
            config.experiment_groups[0].column_name in str(err.value)
        )

    def test_ab_config_init_less_than_two_experiment_groups_raise_error(self):
        with pytest.raises(InputDataError) as err:
            raw_data = [
                ["2019-01-01", 0.34, "treatment"],
                ["2019-01-02", 0.13, "treatment"],
            ]
            data = pd.DataFrame(raw_data, columns=["date", "metric", "group"])
            data.date = pd.to_datetime(data.date)
            config = AbConfig(
                experiment_groups=[ExperimentGroup("group")],
                metrics=[
                    Metric(
                        "metric",
                        MetricType.continuous,
                        MetricAggregateFunc.mean,
                        False,
                        False,
                        False,
                    )
                ],
                covariates=[],
                experiment_type=ExperimentType.ab,
                experiment_randomize_units=[],
            )
            pl = ABPipeline(data, config)
            pl._validate()
        assert "There are less than 2 experiment variations in the column" in str(err.value)

    def test_ab_config_init_experiment_variation_different_from_data_raise_error(self):
        # test variation not exist
        with pytest.raises(InputDataError) as err:
            raw_data = [
                ["2019-01-01", 0.34, "treatment"],
                ["2019-01-02", 0.13, "treatment"],
                ["2019-01-02", 0.13, "control"],
                ["2019-01-02", 0.13, "treatment2"],
            ]
            data = pd.DataFrame(raw_data, columns=["date", "metric", "group"])
            config = AbConfig(
                experiment_groups=[
                    ExperimentGroup(
                        "group",
                        ExperimentVariation("con", 0.5),
                        [ExperimentVariation("treatment", 0.5)],
                    )
                ],
                metrics=[
                    Metric(
                        "metric",
                        MetricType.continuous,
                        MetricAggregateFunc.mean,
                        False,
                        False,
                        False,
                    )
                ],
                covariates=[],
                experiment_type=ExperimentType.ab,
                experiment_randomize_units=[],
            )
            pl = ABPipeline(data, config)
            pl._validate_column_existence_and_type()
        assert '"con" does not exist in the experiment group column' in str(err.value)

        # test subset of variations
        config.experiment_groups = [
            ExperimentGroup(
                "group",
                ExperimentVariation("control", 0.5),
                [ExperimentVariation("treatment", 0.5)],
            )
        ]
        pl = ABPipeline(data, config)
        pl._validate_column_existence_and_type()

    def test_ab_config_init_experiment_variation_not_add_to_one_raise_error(self):
        with pytest.raises(InputConfigError) as err:
            raw_data = [
                ["2019-01-01", 0.34, "treatment"],
                ["2019-01-02", 0.13, "treatment"],
            ]
            data = pd.DataFrame(raw_data, columns=["date", "metric", "group"])
            data.date = pd.to_datetime(data.date)
            config = AbConfig(
                experiment_groups=[
                    ExperimentGroup(
                        "group",
                        ExperimentVariation("control", 0.5),
                        [ExperimentVariation("treatment", 0.4)],
                    )
                ],
                date=Column("date", ColumnType.date),
                metrics=[
                    Metric(
                        "metric",
                        MetricType.ratio,
                        MetricAggregateFunc.mean,
                        False,
                        False,
                        False,
                    )
                ],
                covariates=[],
                experiment_type=ExperimentType.ab,
                experiment_randomize_units=[],
            )
            config._validate_ab_config()
        assert "Sum of variation split must equal 1!" in str(err.value)

    def test_ab_config_experiment_group_check_imbalance(self):
        data, config, _ = get_imbalance_check_input()
        ABconfig = set_experiment_config(config)
        assert ABconfig.check_imbalance_method == CheckImbalanceMethod.binomial

    def test_validation_remove_metric(self):
        data, config = get_ab_int_group_input()
        config["columns"]["metric_not_exist"] = ({"column_type": "metric", "metric_type": "continuous"},)
        config["columns"]["metric_not_exist2"] = ({"column_type": "metric", "metric_type": "continuous"},)
        pl = ABPipeline(data, config)
        assert len(pl.config.metrics) == 1

    def test_valid_sequential_parameters(self):
        data, config = get_ab_int_group_input()
        config["experiment_settings"]["information_rates"] = [0.1, 0.2, 1.0]
        config["experiment_settings"]["target_sample_size"] = 100
        ab_pl = ABPipeline(data, config)
        assert len(ab_pl.config.information_rates) == 3
        assert ab_pl.config.target_sample_size == 100

    def test_validation_for_sequential_parameters_will_throw_errors(self):
        with pytest.raises(InputConfigError):
            # not monotonic
            data, config = get_ab_int_group_input()
            config["experiment_settings"]["information_rates"] = [0.2, 0.1, 1.0]
            config["target_sample_size"] = 100
            ABPipeline(data, config)

        with pytest.raises(InputConfigError):
            # out of range
            data, config = get_ab_int_group_input()
            config["experiment_settings"]["information_rates"] = [0.2, 0.1, 5.0]
            config["target_sample_size"] = 100
            ABPipeline(data, config)

    # tests throwing errors for both numerical, categorical, and ratio that don't exist in data
    def test_validation_data_error_covariates(self):
        data, config = get_ab_int_group_input()

        config["columns"]["covariate_not_exist"] = ({"column_type": "covariate", "value_type": "numerical"},)

        ab_pl = ABPipeline(data, config)
        assert len(ab_pl.config.covariates) == 0

        # check non-existent ratio covariate
        data, config = get_ab_int_group_input(use_cov=True)
        config["columns"]["covariate_not_exist"] = (
            {
                "column_type": "covariate",
                "value_type": "ratio",
                "numerator_column": "covariate_not_exist",
                "denominator_column": "covariate_not_exist",
            },
        )

        pl = ABPipeline(data, config)
        assert len(pl.config.covariates) == 0
        assert len(pl.config.metrics[0].covariates) == 0  # ratio covariate invalid bc denominator

        # check invalid ratio covariates num/den types
        data, config = get_ab_int_group_input(use_cov=True)
        config["columns"]["ratio_num_categorical"] = (
            {
                "column_type": "covariate",
                "value_type": "ratio",
                "numerator_column": "cluster",
                "denominator_column": "cov1",
            },
        )

        pipeline = ABPipeline(data, config)
        result = pipeline.run()
        assert len(result) == 2
        assert len(pipeline.config.covariates) == 0  # ratio covariate invalid bc numerator
        assert len(pipeline.config.metrics[0].covariates) == 0  # ratio covariate invalid bc denominator

        # ratio covariate w categorical den
        config["columns"]["ratio_den_categorical"] = (
            {
                "column_type": "covariate",
                "value_type": "ratio",
                "numerator_column": "cov1",
                "denominator_column": "cluster",
            },
        )
        pipeline = ABPipeline(data, config)
        result = pipeline.run()
        assert len(result) == 2
        assert len(pipeline.config.metrics[0].covariates) == 0  # ratio covariate invalid bc denominator
        assert len(pipeline.config.covariates) == 0  # ratio covariate invalid bc denominator

    def test_ab_config_missing_data_error(self):
        # no data in matching period
        with pytest.raises(InputDataError):
            data, config = get_ab_int_group_input()
            data = data[data["date"] == "2022-08-15"]
            ABPipeline(data, config)

    def test_ab_config_no_metric_warning(self):
        data, config = get_ab_int_group_input()
        config["columns"]["metric1"]["column_type"] = "cluster"
        pipeline = ABPipeline(data, config)
        assert len(pipeline.message_collection.overall_messages) == 1
