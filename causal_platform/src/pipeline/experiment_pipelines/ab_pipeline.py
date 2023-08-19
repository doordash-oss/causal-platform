"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

from typing import Callable, Dict, Optional, Union, Any

import pandas as pd

from causal_platform.src.models.analyser_model.ab_analyser import AbAnalyser
from causal_platform.src.models.configuration_model.config import PipelineConfig, TBaseConfig
from causal_platform.src.models.data_model import DataLoader
from causal_platform.src.pipeline.experiment_pipelines.experiment_pipeline import BaseExperimentPipeline
from causal_platform.src.pipeline.preprocess_pipelines.ab_preprocess_pipeline import AbPreprocessPipeline
from causal_platform.src.utils.config_utils import set_experiment_config
from causal_platform.src.utils.error import InputDataError
from causal_platform.src.utils.logger import logger
from causal_platform.src.utils.validation_utils import (
    check_column_in_data,
    check_data_is_datetime,
    check_data_is_object_type,
)


class ABPipeline(BaseExperimentPipeline):
    def __init__(
        self,
        data: Union[pd.DataFrame, DataLoader],
        config: Union[Dict, TBaseConfig],
        pipeline_config: Optional[PipelineConfig] = None,
        logger_type: Optional[str] = None,
        timer_callback: Optional[Callable] = None,
    ):
        if isinstance(config, dict):
            config = set_experiment_config(config)

        super().__init__(
            data,
            config,
            AbPreprocessPipeline(config=config),
            AbAnalyser,
            pipeline_config,
            logger_type,
            timer_callback,
        )

    def _validate(self):
        logger.info("Start validating AB Pipeline...")
        super()._validate()
        self._validate_ab_column_existence_and_type()
        self._validate_ab_data()
        logger.info("Finished AB Pipeline validation!")

    def _validate_ab_column_existence_and_type(self):
        # validate experiment group existence
        for experiment_group in self.config.experiment_groups:
            if not check_column_in_data(self.data, experiment_group.column_name):
                raise InputDataError(
                    "Experiment_group column '{}' does not exist in data".format(experiment_group.column_name)
                )
        # Validate the date field
        if self.config.date and (not check_data_is_object_type(self.data, self.config.date.column_name)):
            if not check_data_is_datetime(self.data, self.config.date.column_name):
                raise InputDataError("Date columns '{}' is not valid".format(self.config.date.column_name))

    def _validate_ab_data(self):
        # Check num of groups in data
        for experiment_group in self.config.experiment_groups:
            unique_variation_arr = self.data[experiment_group.column_name].dropna().unique()
            unique_variation_set = set(unique_variation_arr)
            if len(unique_variation_set) < 2:
                raise InputDataError(
                    "There are less than 2 experiment variations in the column '{}'".format(
                        experiment_group.column_name
                    )
                )
            # Ensure experiment variation exists in the data
            for variant in experiment_group.all_variation_names:
                if variant not in unique_variation_set:
                    raise InputDataError(
                        f"""
                        "{variant}" does not exist in the experiment group column "{experiment_group.column_name}" of the data. Please check the sql query and config to ensure all variantions as specified in the config exist in the data.
                        """
                    )

            # Convert dtype to allow float 1.0 in the data but int 1 in the config
            variation_dtype = unique_variation_arr.dtype
            control_name = experiment_group.control.variation_name
            control_dtype = type(control_name)
            if variation_dtype in [int, float] and variation_dtype != control_dtype:
                func = int if variation_dtype == int else float
                experiment_group.control.variation_name = func(control_name)
                for treatment in experiment_group.treatments:
                    treatment.variation_name = func(treatment.variation_name)

    def get_covariate_results(
        self, output_format: str = "dataframe", **kwargs
    ) -> Union[Dict[str, Any], str, pd.DataFrame]:
        self.analyser: AbAnalyser
        covariate_results = self.analyser.get_covariate_results()
        if output_format == "dict":
            result = self._populate_dict_covariate_result(
                covariate_results,
            )
        elif output_format == "json":
            result = self._populate_json_covariate_result(
                covariate_results,
            )
        else:
            result = self._populate_dataframe_analysis_result(covariate_results)
        return result
