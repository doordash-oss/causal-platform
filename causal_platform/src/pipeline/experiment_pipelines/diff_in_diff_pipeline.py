"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import json
from typing import Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from causal_platform.src.models.analyser_model.diff_in_diff_analyser import DiffinDiffAnalyser
from causal_platform.src.models.configuration_model.base_objects import ColumnType
from causal_platform.src.models.configuration_model.config import PipelineConfig, TBaseConfig
from causal_platform.src.models.result_model.result import (
    DiDOutputResult,
    DiffinDiffPreprocessPipelineResult,
)
from causal_platform.src.pipeline.experiment_pipelines.experiment_pipeline import BaseExperimentPipeline
from causal_platform.src.pipeline.preprocess_pipelines.diff_in_diff_preprocess_pipeline import (
    DiffinDiffPreprocessPipeline,
)
from causal_platform.src.utils.common_utils import NumpyEncoder
from causal_platform.src.utils.config_utils import set_experiment_config
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.diff_in_diff.calculation import standardize_and_calculate_weighted_sum
from causal_platform.src.utils.diff_in_diff.plotting import (
    plot_diff_in_diff_treatment_effect,
    plot_matching_parallel_lines,
    prep_matching_plot_data,
    prep_treatment_effect_plot_data,
)
from causal_platform.src.utils.diff_in_diff.prep_data import get_data_between_start_end_date
from causal_platform.src.utils.error import InputConfigError, InputDataError
from causal_platform.src.utils.logger import logger
from causal_platform.src.utils.validation_utils import check_column_in_data, check_column_is_type


class DiffinDiffPipeline(BaseExperimentPipeline):
    def __init__(
        self,
        data: pd.DataFrame,
        config: Union[Dict, TBaseConfig],
        pipeline_config: PipelineConfig = None,
        control_unit_ids: Optional[List[int]] = None,
        logger_type: Optional[str] = None,
        timer_callback: Optional[Callable] = None,
    ):
        if isinstance(config, dict):
            config = set_experiment_config(config)

        super().__init__(
            data,
            config,
            DiffinDiffPreprocessPipeline(config=config, control_unit_ids=control_unit_ids),
            DiffinDiffAnalyser,
            pipeline_config,
            logger_type,
            timer_callback,
        )

        self.control_unit_ids = control_unit_ids
        self.preprocess_result = DiffinDiffPreprocessPipelineResult(processed_data=self.data)

    def _validate(self):
        logger.info("Start validating DiffinDiff Pipeline...")
        super()._validate()
        self._validate_diff_in_diff_column_existence_and_type()
        self._validate_diff_in_diff_data()
        logger.info("Finished DiffinDiff Pipeline validation!")

    def matching(self) -> pd.DataFrame:
        """expose market matching function so that user can choose to only
        run this to get matched market before the experiment start.
        """
        self.preprocess_result: DiffinDiffPreprocessPipelineResult = self.preprocess_pipeline.matching(self.data)
        return self.preprocess_result.matched_units_output(self.config.experiment_randomize_units[0])

    def plot_matching(
        self,
        control_unit_ids: Optional[List[int]] = None,
        y_metric: Optional[str] = None,
        figsize: Tuple = Constants.DID_PLOT_DEFAULT_FIGURE_SIZE,
        title: Optional[str] = Constants.DID_PLOT_DEFAULT_MATCHING_PLOT_TITLE,
        xlabel: Optional[str] = Constants.DID_PLOT_DEFAULT_X_LABEL,
        ylabel: Optional[str] = Constants.DID_PLOT_DEFAULT_Y_LABEL,
    ):
        # TODO(caixia): matching_aggregate_func only use np.sum for now
        # but we can add this to config so that user can choose
        if y_metric is None:
            y_metric = Constants.WEIGHTED_SUM_COLUMN_NAME
            self.__check_and_add_weighted_sum_column_to_data()
        else:
            if y_metric not in self.preprocess_result.processed_data.columns:
                raise InputDataError("{} not in data!".format(y_metric))

        if control_unit_ids is None:
            if self.preprocess_result.matched_unit_ids is None:
                raise InputConfigError("Please provide the unit ids of control group.")
            else:
                control_unit_ids = self.preprocess_result.matched_unit_ids

        (
            matching_treatment_aggregate_metric_series,
            matching_control_aggregate_metric_series,
        ) = prep_matching_plot_data(
            self.preprocess_result.processed_data,
            self.config,
            control_unit_ids,
            y_metric,
            matching_aggregate_func=Constants.DIFF_IN_DIFF_MATCHING_AGGREGATE_FUNC,
        )

        plot_matching_parallel_lines(
            matching_treatment_aggregate_metric_series,
            matching_control_aggregate_metric_series,
            control_unit_ids,
            self.config.treatment_unit_ids,
            figsize,
            title,
            xlabel,
            ylabel,
        )

    def __check_and_add_weighted_sum_column_to_data(self):
        if Constants.WEIGHTED_SUM_COLUMN_NAME not in self.data.columns:
            self.preprocess_result.processed_data[
                Constants.WEIGHTED_SUM_COLUMN_NAME
            ] = standardize_and_calculate_weighted_sum(
                self.preprocess_result.processed_data[self.config.matching_columns].to_numpy(),
                self.config,
            )

    def plot_treatment_effect(
        self,
        figsize: Tuple[int, int] = Constants.DID_PLOT_DEFAULT_FIGURE_SIZE,
        title: str = Constants.DID_PLOT_DEFAULT_TREATMENT_EFFECT_PLOT_TITLE,
        xlabel: str = Constants.DID_PLOT_DEFAULT_X_LABEL,
        ylabel: str = Constants.DID_PLOT_DEFAULT_Y_LABEL,
    ):
        if self.preprocess_result is None:
            raise InputConfigError("Must execute run function of the pipeline before plotting!")
        plot_data = prep_treatment_effect_plot_data(
            self.preprocess_result.processed_data,
            self.config,
            self.config.metrics[0].column_name,
            Constants.DIFF_IN_DIFF_MATCHING_AGGREGATE_FUNC,
        )
        plot_diff_in_diff_treatment_effect(plot_data, self.config, figsize, title, xlabel, ylabel)

    def run(
        self,
        output_format="dataframe",
        analyser_kwargs=None,
        serializer_kwargs=None,
    ) -> Union[DiDOutputResult, str]:

        use_format = "dict" if output_format == "json" else output_format
        analysis_result = super().run(output_format=use_format)
        matching_result = self.preprocess_result.matched_units_output(
            self.config.experiment_randomize_units[0], output_format=use_format
        )

        if output_format == "json":
            # merge analysis_result and matching_result into one dict
            analysis_result[Constants.JSON_MATCHING_RESULT] = matching_result[Constants.JSON_MATCHING_RESULT]
            analysis_result[Constants.JSON_MATCHING_METHOD] = matching_result[Constants.JSON_MATCHING_METHOD]
            analysis_result[Constants.JSON_MATCHING_COLUMN_NAME] = matching_result[Constants.JSON_MATCHING_COLUMN_NAME]
            return json.dumps(analysis_result, cls=NumpyEncoder)
        else:
            did_output_result = DiDOutputResult(analysis_result=analysis_result, matching_result=matching_result)
            return did_output_result

    def _validate_diff_in_diff_column_existence_and_type(self):
        # validate each randomize unit column exists
        for column in self.config.experiment_randomize_units:
            if not check_column_in_data(self.data, column.column_name):
                raise InputDataError("Randomize Unit column '{}' does not exist in data".format(column.column_name))

        # validate that the date column exists
        if not check_column_in_data(self.data, self.config.date.column_name):
            raise InputDataError("Date column '{}' does not exist in data".format(self.config.date.column_name))

        # validate that the matching columns exist
        for matching_column in self.config.matching_columns:
            if not check_column_in_data(self.data, matching_column):
                raise InputDataError("Matching column '{}' does not exist in data".format(matching_column))

    def _validate_diff_in_diff_data(self):
        # validate that the match end > match start
        if self.config.matching_start_date >= self.config.matching_end_date:
            raise InputConfigError("Matching end date must be after matching start date")

        # validate that the exp start > match end
        if self.config.matching_end_date >= self.config.experiment_start_date:
            raise InputConfigError("Experiment start date must be after matching end date")

        # validate that the exp end > exp start
        if self.config.experiment_start_date >= self.config.experiment_end_date:
            raise InputConfigError("Experiment end date must be after experiment start date")

        # validate date column values are of type datetime
        if not check_column_is_type(self.data, self.config.date, ColumnType.date):
            raise InputConfigError("Date column '{}' has non datetime values".format(self.config.date.column_name))

        matching_data = get_data_between_start_end_date(
            self.data,
            self.config.date.column_name,
            self.config.matching_start_date,
            self.config.matching_end_date,
        )

        # validate that there is data in the match period
        if len(matching_data) == 0:
            raise InputDataError(
                "No data found in matching period '{} - {}'".format(
                    self.config.matching_start_date, self.config.matching_end_date
                )
            )

        # validate that there are enough unique unit ids
        experiment_unit_column = self.config.get_diff_in_diff_experiment_unit_column()
        unique_treatment_units = list(self.data[experiment_unit_column.column_name].unique())
        required_units = len(self.config.treatment_unit_ids)
        required_units += self.config.match_unit_size
        required_units += len(self.config.exclude_unit_ids)

        if required_units > len(unique_treatment_units):
            err_msg = """Not enough unique units to pick '{}' control units
            for each treatment unit. At least '{}' required. Found '{}'."""
            raise InputDataError(
                err_msg.format(
                    self.config.match_unit_size,
                    required_units,
                    len(unique_treatment_units),
                )
            )
