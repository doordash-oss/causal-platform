"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import json
from typing import Any, Callable, Dict, List, Optional, Type, Union

import pandas as pd

from causal_platform.src.models.analyser_model.analyser import BaseAnalyser, TBaseAnalyser
from causal_platform.src.models.configuration_model.base_objects import (
    CovariateType,
    ExperimentType,
    MetricType,
)
from causal_platform.src.models.configuration_model.config import PipelineConfig, TBaseConfig
from causal_platform.src.models.message.message import Message, MessageCollection, Source, Status
from causal_platform.src.models.result_model.result import AnalysisResult, BasePreprocessPipelineResult, CovariateResult
from causal_platform.src.pipeline.preprocess_pipelines.preprocess_pipeline import BasePreprocessPipeline
from causal_platform.src.utils.common_utils import (
    DataLoader,
    NumpyEncoder,
    convert_data_to_proper_types,
    convert_table_column_to_lower_case,
)
from causal_platform.src.utils.config_utils import set_experiment_config
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.dashab_timer import time_profiler
from causal_platform.src.utils.error import InputDataError
from causal_platform.src.utils.experiment.result_utils import (
    get_metric_result_dict_list,
    get_covariate_result_dict_list,
)
from causal_platform.src.utils.logger import logger
from causal_platform.src.utils.validation_utils import check_column_in_data, check_column_is_type


class BaseExperimentPipeline:
    def __init__(
        self,
        data: Union[pd.DataFrame, DataLoader],
        config: Union[Dict, TBaseConfig],
        preprocess_pipeline: BasePreprocessPipeline,
        analyser_class: Type[BaseAnalyser],
        pipeline_config: Optional[PipelineConfig] = None,
        logger_type: Optional[str] = None,
        timer_callback: Optional[Callable] = None,
    ):
        # Prepare a default configuration if none is provided
        self.pipeline_config = PipelineConfig() if pipeline_config is None else pipeline_config
        logger.reset_logger_with_type(logger_type)
        if isinstance(config, dict):
            self.config = set_experiment_config(config)
        else:
            self.config = config

        # Setup attributes
        self.data = data.copy() if self.pipeline_config.copy_input_data else data
        self.timer_callback = timer_callback
        self.preprocess_pipeline = preprocess_pipeline
        self.analyser_class = analyser_class
        self.analyser: Optional[TBaseAnalyser] = None
        self.message_collection: MessageCollection = MessageCollection()

        # Do any cleaning/validation
        self._clean_data_for_preprocessing()
        self._validate()

        # self.preprocess_result will be overridden when preprocessing is called
        self.preprocess_result = BasePreprocessPipelineResult(processed_data=self.data)

    def _validate(self):
        self._validate_data_size()
        self._validate_column_existence_and_type()
        if len(self.config.metrics) == 0:
            self.message_collection.add_overall_message(
                Message(
                    source=Source.analysis,
                    title="Validation",
                    description="""Skip the metric analysis since no metric pass the validation!""",
                    status=Status.skip,
                )
            )

    @time_profiler(process_name="preprocessor")
    def _preprocessor(self, data: Union[pd.DataFrame, DataLoader], **kwargs) -> BasePreprocessPipelineResult:
        result = self.preprocess_pipeline.run(data=data)
        self.message_collection.combine(self.preprocess_pipeline.message_collection)
        return result

    @time_profiler(process_name="analyser")
    def _analyser(self, preprocess_result: BasePreprocessPipelineResult, **kwargs) -> List[AnalysisResult]:
        self.analyser = self.analyser_class(self.config, preprocess_result)
        result = self.analyser.run()
        self.message_collection.combine(self.analyser.message_collection)
        return result

    def _populate_dataframe_analysis_result(
        self, analysis_result_list: List[Union[AnalysisResult, CovariateResult]]
    ) -> pd.DataFrame:
        """function to transform list of AnalysisResult objects into a dictionary return to user

        Arguments:
            analysis_results {List[AnalysisResult]} -- list of result from analyser

        Returns:
            [pd.DataFrame] -- Result of all the metrics in a dataframe of shape m * k * 2
                where m is number of metric, k is number of result (i.e. p-value)
        """
        result_array = []
        for analysis_result in analysis_result_list:
            result_array.extend(analysis_result.result_dicts_for_dataframe_output)

        return pd.DataFrame(result_array)

    def _transform_hierarchy_dict_to_dict(
        self, analysis_result_dict: Dict[str, AnalysisResult]
    ) -> List[Dict[str, Any]]:
        """function to transform list of AnalysisResult objects into a dictionary (json)
        for the experiment platform

        Arguments:
            analysis_result_dict {Dict} -- list of result from analyser

        Returns:
            analysis_result_json_dict_list [List[Dict]] -- Result of list of analysis results
            in json format required by the experiment platform
        """
        metric_results = []
        for metric_name, metric_result in analysis_result_dict.items():
            output_experiment_group_results = []
            for experiment_group_name, experiment_group_result in metric_result.items():
                agg_func_results = []
                for agg_func_name, agg_func_result in experiment_group_result.items():
                    output_keys = {
                        Constants.VARIATION,
                        Constants.AVERAGE_TREATMENT_EFFECT,
                        Constants.RELATIVE_AVERAGE_TREATMENT_EFFECT,
                        Constants.P_VALUE,
                        Constants.METRIC_VALUE,
                        Constants.ABSOLUTE_CONFIDENCE_INTERVAL,
                        Constants.RELATIVE_CONFIDENCE_INTERVAL,
                        Constants.SAMPLE_SIZE,
                        Constants.DATA_SIZE,
                        Constants.SE,
                        Constants.SEQUENTIAL_P_VALUE,
                        Constants.SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL,
                        Constants.SEQUENTIAL_RELATIVE_CONFIDENCE_INTERVAL,
                        Constants.SEQUENTIAL_RESULT_TYPE,
                    }

                    treatment_results, control_results = [], []
                    for _, variation_result in agg_func_result.get(Constants.CONTROL_RESULTS, {}).items():
                        control_results.append({k: v for k, v in variation_result.items() if k in output_keys})

                    for _, variation_result in agg_func_result[Constants.TREATMENT_RESULTS].items():
                        treatment_results.append({k: v for k, v in variation_result.items() if k in output_keys})

                    output_agg_func_result = {
                        Constants.AGG_FUNC_NAME: agg_func_name,
                        Constants.CONTROL_RESULTS: control_results,
                        Constants.TREATMENT_RESULTS: treatment_results,
                    }
                    agg_func_results.append(output_agg_func_result)
                output_experiment_group_result = {
                    Constants.EXPERIMENT_GROUP_NAME: experiment_group_name,
                    Constants.AGG_FUNC_RESULTS: agg_func_results,
                }
                output_experiment_group_results.append(output_experiment_group_result)
            output_metric_result = {
                Constants.METRIC_NAME: metric_name,
                Constants.EXPERIMENT_GROUP_RESULTS: output_experiment_group_results,
            }
            metric_results.append(output_metric_result)
        return metric_results

    def _populate_dict_covariate_result(self, covariate_result_list: List[CovariateResult]) -> Dict[str, Any]:
        covariate_result_dict_list = get_covariate_result_dict_list(covariate_result_list)
        output_dict = {Constants.COVARIATE_RESULTS: covariate_result_dict_list}
        return output_dict

    def _populate_json_covariate_result(
        self,
        analysis_result_list: List[CovariateResult],
    ) -> str:
        dict_result = self._populate_dict_covariate_result(analysis_result_list)
        json_result = json.dumps(dict_result, cls=NumpyEncoder)
        return json_result

    def _populate_json_analysis_result(
        self,
        analysis_result_list: List[AnalysisResult],
        preprocess_result: BasePreprocessPipelineResult,
    ) -> str:
        """
        function to transform list of AnalysisResult into json format
        :param analysis_result_list:
        :return: json
        """
        dict_result = self._populate_dict_analysis_result(analysis_result_list, preprocess_result)
        json_result = json.dumps(dict_result, cls=NumpyEncoder)
        return json_result

    def _populate_dict_analysis_result(
        self,
        analysis_result_list: List[AnalysisResult],
        preprocess_result: BasePreprocessPipelineResult,
    ) -> Dict[str, Any]:
        """
        function to transform list of AnalysisResult objects into a dictionary
        for the experiment platform
        """
        metric_result_dict = get_metric_result_dict_list(analysis_result_list)

        output_dict = {Constants.METRIC_RESULTS: metric_result_dict}

        # add preprocessing results
        if self.config.experiment_type == ExperimentType.ab:
            preprocess_dict = {}
            if preprocess_result.does_flicker_exists is not None:
                preprocess_dict[Constants.DOES_FLICKER_EXISTS] = preprocess_result.does_flicker_exists
            if preprocess_result.are_buckets_imbalanced is not None:
                preprocess_dict[Constants.ARE_BUCKETS_IMBALANCED] = preprocess_result.are_buckets_imbalanced
            output_dict[Constants.PREPROCESS_RESULTS] = preprocess_dict

        # add error and warning messages from Message collection
        output_dict[Constants.LOG_MESSAGES] = {}
        messages_description_list_by_type = self.message_collection.get_messages_description_list()
        output_dict[Constants.LOG_MESSAGES][Constants.ERRORS] = messages_description_list_by_type.get(Status.fail)
        output_dict[Constants.LOG_MESSAGES][Constants.WARNINGS] = messages_description_list_by_type.get(Status.warn)
        return output_dict

    def run(self, output_format: str = "dataframe") -> Union[pd.DataFrame, dict, str]:
        """user interface function for the pipeline

        Returns:
            [Dict] -- return a dictionary which contains the result of each metric
        """

        kwargs = {"timer_callback": self.timer_callback}

        # Run any preprocessing on the data
        self.preprocess_result = self._preprocessor(self.data, **kwargs)

        # Execute the analysis
        analysis_result_list = self._analyser(self.preprocess_result, **kwargs)

        # Serialize the result
        return self._serialize_analysis_result(output_format, analysis_result_list, **kwargs)

    @time_profiler(process_name="serialization")
    def _serialize_analysis_result(
        self, output_format: str, analysis_result_list: List[AnalysisResult], **kwargs
    ) -> Union[Dict[str, Any], str, pd.DataFrame]:
        if output_format == "dict":
            result = self._populate_dict_analysis_result(analysis_result_list, self.preprocess_result)
        elif output_format == "json":
            result = self._populate_json_analysis_result(analysis_result_list, self.preprocess_result)
        else:
            result = self._populate_dataframe_analysis_result(analysis_result_list)
        return result

    def run_under_simulation(self, analyser_kwargs=None) -> List[AnalysisResult]:
        """interface for simulation. The difference between user interface function
           is that this function return a List[AnalysisResult] object instead of Dict

        Returns:
            [List[AnalysisResult]] -- list of AnalysisResult object.
        """

        # Protect against None inputs
        analyser_kwargs = {} if analyser_kwargs is None else analyser_kwargs

        preprocess_result: BasePreprocessPipelineResult = self._preprocessor(self.data)
        analysis_result_list: List[AnalysisResult] = self._analyser(preprocess_result, **analyser_kwargs)
        return analysis_result_list

    def _validate_column_existence_and_type(self):
        invalid_metrics = []
        invalid_covariates = []
        for metric in self.config.metrics:
            # check existence
            if metric.metric_type == MetricType.ratio:
                if self.config.experiment_type not in [ExperimentType.ab, ExperimentType.causal]:
                    self.message_collection.add_metric_message(
                        metric=metric.column_name,
                        message=Message(
                            source=Source.validation,
                            status=Status.fail,
                            title="Ratio metrics only support for AB analysis",
                            description=f"Metric '{metric.column_name}' is a ratio metric. We only support ratio metric \
                            for A/B experiment",
                        ),
                    )

                    invalid_metrics.append(metric)
                    continue
                # data existence
                if not check_column_in_data(self.data, metric.numerator_column.column_name):

                    self.message_collection.add_metric_message(
                        metric=metric.column_name,
                        message=Message(
                            source=Source.validation,
                            status=Status.fail,
                            title="Missing numerator column for ratio metric",
                            description=f"""
                            Numerator column '{metric.numerator_column.column_name}'
                            for ratio metric '{metric.column_name}' does not exist
                            """,
                        ),
                    )

                    invalid_metrics.append(metric)
                    continue

                if not check_column_in_data(self.data, metric.denominator_column.column_name):
                    self.message_collection.add_metric_message(
                        metric=metric.column_name,
                        message=Message(
                            source=Source.validation,
                            status=Status.fail,
                            title="Missing denominator column for ratio metric",
                            description=f"""
                            Denominator column '{metric.denominator_column.column_name}' for ratio metric '{metric.column_name}' does not exist
                            """,
                        ),
                    )

                    invalid_metrics.append(metric)
                    continue
                # data type

                if not (
                    check_column_is_type(self.data, metric.numerator_column, MetricType.continuous)
                    or check_column_is_type(self.data, metric.numerator_column, MetricType.proportional)
                ):
                    self.message_collection.add_metric_message(
                        metric=metric.column_name,
                        message=Message(
                            source=Source.validation,
                            status=Status.fail,
                            title="""
                            Ratio metrics require numerator and denominator columns to be continuous or proportional
                            """,
                            description=f"""
                            Metric {metric.column_name} and numerator {metric.numerator_column.column_name} is
                            of the wrong type
                            """,
                        ),
                    )

                    invalid_metrics.append(metric)
                    continue

                if not (
                    check_column_is_type(self.data, metric.denominator_column, MetricType.continuous)
                    or check_column_is_type(self.data, metric.denominator_column, MetricType.proportional)
                ):
                    self.message_collection.add_metric_message(
                        metric=metric.column_name,
                        message=Message(
                            source=Source.validation,
                            status=Status.fail,
                            title="""
                            Ratio metrics require numerator and denominator columns to be continuous or proportional
                            """,
                            description=f"""
                            Metric {metric.column_name} and denominator {metric.denominator_column.column_name}
                            is of the wrong type
                            """,
                        ),
                    )

                    invalid_metrics.append(metric)
                    continue
            else:
                if not check_column_in_data(self.data, metric.column_name):
                    self.message_collection.add_metric_message(
                        metric=metric.column_name,
                        message=Message(
                            source=Source.validation,
                            status=Status.fail,
                            title=f"Metric column {metric.column_name} does not exist",
                            description=f"Metric column {metric.column_name} does not exist in data",
                        ),
                    )

                    invalid_metrics.append(metric)
                    continue

            # check data type
            if not check_column_is_type(self.data, metric, metric.metric_type):
                self.message_collection.add_metric_message(
                    metric=metric.column_name,
                    message=Message(
                        source=Source.validation,
                        status=Status.fail,
                        description=f"Metric column {metric.column_name} must be type {metric.metric_type}",
                    ),
                )
                invalid_metrics.append(metric)
                continue

        self.config.metrics = [metric for metric in self.config.metrics if metric not in invalid_metrics]

        for covariate in self.config.all_distinct_covariates:
            if covariate.value_type is CovariateType.ratio:
                # data existence
                if not check_column_in_data(self.data, covariate.numerator_column.column_name):
                    self.message_collection.add_overall_message(
                        message=Message(
                            source=Source.validation,
                            status=Status.fail,
                            description=f"Ratio covariate numerator column {covariate.numerator_column.column_name} does not exist in data",
                        ),
                    )
                    invalid_covariates.append(covariate)
                    continue

                if not check_column_in_data(self.data, covariate.denominator_column.column_name):
                    self.message_collection.add_overall_message(
                        message=Message(
                            source=Source.validation,
                            status=Status.fail,
                            description=f"Ratio covariate denominator column {covariate.denominator_column.column_name} does not exist in data",
                        ),
                    )
                    invalid_covariates.append(covariate)
                    continue

                # check data types covariates
                if not (
                    check_column_is_type(self.data, covariate.numerator_column, MetricType.continuous)
                    or check_column_is_type(self.data, covariate.numerator_column, MetricType.proportional)
                ):
                    self.message_collection.add_overall_message(
                        message=Message(
                            source=Source.validation,
                            status=Status.fail,
                            title="""Ratio covariates require numerator and denominator columns to be numerical """,
                            description=f"""Covariate {covariate.column_name} and numerator {covariate.numerator_column.column_name} is of the wrong type """,
                        ),
                    )

                    invalid_covariates.append(covariate)
                    continue

                if not (
                    check_column_is_type(self.data, covariate.denominator_column, MetricType.continuous)
                    or check_column_is_type(self.data, covariate.denominator_column, MetricType.proportional)
                ):
                    self.message_collection.add_overall_message(
                        message=Message(
                            source=Source.validation,
                            status=Status.fail,
                            title="""
                                                Ratio covariates require numerator and denominator columns to be numerical
                                                """,
                            description=f"""
                                                Covariate {covariate.column_name} and denominator {covariate.denominator_column.column_name}
                                                is of the wrong type
                                                """,
                        ),
                    )

                    invalid_covariates.append(covariate)
                    continue

            else:
                if not check_column_in_data(self.data, covariate.column_name):
                    self.message_collection.add_overall_message(
                        message=Message(
                            source=Source.validation,
                            status=Status.fail,
                            description=f"Covariate column {covariate.column_name} does not exist in data",
                        ),
                    )
                    invalid_covariates.append(covariate)
                    continue

        self.config.covariates = [
            covariate for covariate in self.config.covariates if covariate not in invalid_covariates
        ]

        self._filter_invalid_covariates_in_each_metric(invalid_covariates)

    def _validate_data_size(self):
        if self.data.shape[0] == 0:
            metric_names = ",".join(list(map(lambda metric: metric.column_name, self.config.metrics)))
            raise InputDataError(f"The input data is empty for these metric(s): {metric_names}")

    def _filter_invalid_covariates_in_each_metric(self, invalid_covariates):
        # since metric.covariates may include covariates that are not in the valid covariates, we must remove them too
        for metric in self.config.metrics:
            for covariate in invalid_covariates:
                if covariate in metric.covariates:
                    metric.covariates.remove(covariate)

    def _clean_data_for_preprocessing(self):
        """
        Perform data cleansing for preprocessing pipeline
        The purpose is to perform all non-functionality operations. Now including:
            convert column name to small case,
            data type conversion,
            date column conversion to datetime,
        The purpose is not to perform data distribution check or transformations,
        which should be included in PreProcessingPipeline
        :return: processed data
        """
        # step 1: convert table name
        convert_table_column_to_lower_case(self.data)

        # step 2: remove unused columns
        self._remove_unused_columns()

        # step 3: convert data type for metrics, covariates and date
        convert_data_to_proper_types(self.data, self.config.metrics, self.config.date, self.config.covariates)

    def _remove_unused_columns(self):
        required_cols = self.config.get_required_columns_names()
        columns_to_keep = [c for c in required_cols if c in self.data.columns]

        if isinstance(self.data, pd.DataFrame):
            self.data = self.data[columns_to_keep]

        elif isinstance(self.data, DataLoader):
            pass  # no need to drop columns when saving to parquet
