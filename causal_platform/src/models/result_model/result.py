"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import pandas as pd

from causal_platform.src.models.configuration_model.base_objects import (
    ExperimentGroup,
    ExperimentVariation,
    HypothesisTesting,
    Metric,
    SequentialResultType,
    TColumn,
    Covariate,
)
from causal_platform.src.models.data_model import DataLoader
from causal_platform.src.models.message.message import Message, MessageCollection, Source, Status
from causal_platform.src.utils.common_utils import NumpyEncoder, isnan
from causal_platform.src.utils.constants import Constants


class DiDMatchUnit:
    def __init__(self, unit_id: int, matching_score: float, score_name: str = None):
        self.unit_id = unit_id
        self.matching_score = matching_score
        self.score_name = score_name


class PreprocessResult:
    def __init__(
        self,
        processed_data: Union[pd.DataFrame, DataLoader],
        is_process_needed: Optional[bool] = None,
        matched_units: Optional[List[DiDMatchUnit]] = None,
        does_flicker_exists: Optional[bool] = None,
        are_buckets_imbalanced: Optional[bool] = None,
    ):
        self.processed_data = processed_data
        self.is_process_needed = is_process_needed

        # for diff-in-diff only
        self.matched_units = matched_units

        # To output the result of flicker test for AB tests
        self.does_flicker_exists = does_flicker_exists
        self.are_buckets_imbalanced = are_buckets_imbalanced

        # These messages describe what happened during the processing
        self.message_collection: MessageCollection = MessageCollection()


class BasePreprocessPipelineResult:
    def __init__(self, processed_data: Union[pd.DataFrame, DataLoader]):
        self.processed_data = processed_data

        # These messages describe what happened during the processing
        self.message_collection: MessageCollection = MessageCollection()


class AbPreprocessPipelineResult(BasePreprocessPipelineResult):
    def __init__(
        self,
        processed_data: Union[pd.DataFrame, DataLoader],
        test_to_use: Dict[str, HypothesisTesting] = None,
        does_flicker_exists: Optional[bool] = None,
        are_buckets_imbalanced: Optional[bool] = None,
    ):
        super().__init__(processed_data)
        self.test_to_use = test_to_use
        self.does_flicker_exists = does_flicker_exists
        self.are_buckets_imbalanced = are_buckets_imbalanced


class DiffinDiffPreprocessPipelineResult(BasePreprocessPipelineResult):
    def __init__(
        self,
        processed_data: pd.DataFrame,
        matched_control_units: List[DiDMatchUnit] = None,
    ):
        super().__init__(processed_data)
        self.matched_control_units = matched_control_units

    @property
    def matched_unit_ids(self) -> List[int]:
        unit_ids = [unit.unit_id for unit in self.matched_control_units]
        return unit_ids

    @property
    def matched_unit_matching_scores(self) -> List[float]:
        scores = [unit.matching_score for unit in self.matched_control_units]
        return scores

    def matched_units_output(self, experiment_randomize_unit: TColumn, output_format="dataframe"):
        matching_column_name = experiment_randomize_unit.column_name
        matching_method = self.matched_control_units[0].score_name
        matched_units = [
            {
                Constants.JSON_MATCHING_ID: unit.unit_id,
                Constants.JSON_MATCHING_SCORE: unit.matching_score,
            }
            for unit in self.matched_control_units
        ]
        matched_units_dict = {
            Constants.JSON_MATCHING_RESULT: matched_units,
            Constants.JSON_MATCHING_METHOD: matching_method,
            Constants.JSON_MATCHING_COLUMN_NAME: matching_column_name,
        }

        if output_format == "dataframe":
            matched_units_df = pd.DataFrame(matched_units)
            matched_units_df.columns = [matching_column_name, matching_method]
            return matched_units_df.sort_values(by=matching_method, ascending=False)
        elif output_format == "dict":
            return matched_units_dict
        elif output_format == "json":
            return json.dumps(matched_units_dict, cls=NumpyEncoder)


@dataclass
class CovariateResult:
    covariate: Covariate
    metric: Metric
    estimated_coefficient: float
    p_value: float
    confidence_interval_left: float
    confidence_interval_right: float
    data_size: float
    se: float

    @property
    def result_dict(self):
        return OrderedDict(
            {
                Constants.COVARIATE_NAME: self.covariate.column_name,
                Constants.METRIC_NAME: self.metric.original_column_name,
                Constants.COEFFICIENT: self.estimated_coefficient,
                Constants.P_VALUE: self.p_value,
                Constants.CONFIDENCE_INTERVAL: [
                    self.confidence_interval_left,
                    self.confidence_interval_right,
                ],
                Constants.DATA_SIZE: self.data_size,
                Constants.SE: self.se,
            }
        )

    @property
    def result_dicts_for_dataframe_output(
        self,
    ) -> List[Any]:
        return [self.result_dict]


class AnalysisResult:
    simulation_column_index = {
        "estimated_treatment_effect": 0,
        "p_value": 1,
        "confidence_interval_left": 2,
        "confidence_interval_right": 3,
        "experiment_group_variation": 4,
        "experiment_group": 5,
    }

    def __init__(
        self,
        *,
        metric: Metric,
        estimated_treatment_effect: float,
        p_value: float,
        confidence_interval_left: float,
        confidence_interval_right: float,
        experiment_group: Optional[Union[ExperimentGroup, Tuple[ExperimentGroup, ...]]],
        experiment_group_variation: Optional[Union[ExperimentVariation, Tuple[ExperimentVariation, ...]]],
        se: float,
        metric_treatment_value: Optional[float],
        metric_control_value: Optional[float],
        metric_treatment_sample_size: Optional[int],
        metric_control_sample_size: Optional[int],
        metric_treatment_data_size: Optional[int],
        metric_control_data_size: Optional[int],
        is_sequential_result_valid: bool = False,
        sequential_p_value: Optional[float] = None,
        sequential_confidence_interval_left: Optional[float] = None,
        sequential_confidence_interval_right: Optional[float] = None,
        sequential_result_type: Optional[SequentialResultType] = None,
        is_interaction_result: bool = False,
    ):
        self.metric = metric
        # TODO: change this to plural
        self.experiment_group = experiment_group
        self.experiment_group_variation = experiment_group_variation
        self.metric_treatment_value = metric_treatment_value
        self.metric_control_value = metric_control_value
        self.metric_treatment_sample_size = metric_treatment_sample_size
        self.metric_treatment_data_size = metric_treatment_data_size
        self.metric_control_sample_size = metric_control_sample_size
        self.metric_control_data_size = metric_control_data_size
        self.estimated_treatment_effect = estimated_treatment_effect
        self.p_value = p_value
        self.confidence_interval_left = confidence_interval_left
        self.confidence_interval_right = confidence_interval_right
        self.se = se

        """
        NOTE: The is_sequential_result flag determines if we should return the values for sequential testing.
        We currently only support sequential p-values for the BasicFitter. This means other fitters will return a None.
        So we need to set this flag in order to make sure we require non-null values for BasicFitter results.
        """
        self.is_sequential_result = is_sequential_result_valid
        self.sequential_p_value = sequential_p_value
        self.sequential_confidence_interval_left = sequential_confidence_interval_left
        self.sequential_confidence_interval_right = sequential_confidence_interval_right
        self.sequential_result_type = sequential_result_type

        self.message_collection = MessageCollection()
        # TODO: interaction result will need to be in a separate result object, using this flag
        # temporarily to skip some checks for interaction result
        self.is_interaction_result = is_interaction_result

    @property
    def experiment_groups_name(self):
        if isinstance(self.experiment_group, tuple):
            return "(" + ",".join(e.column_name for e in self.experiment_group) + ")"
        else:
            return self.experiment_group.column_name

    @property
    def experiment_groups_control_variation_name(self):
        if isinstance(self.experiment_group, tuple):
            return "(" + ",".join(str(e.control.variation_name) for e in self.experiment_group) + ")"
        else:
            return self.experiment_group.control.variation_name

    @property
    def experiment_groups_treatment_variation_name(self) -> str:
        if isinstance(self.experiment_group, tuple):
            return "(" + ",".join(str(e.variation_name) for e in self.experiment_group_variation) + ")"
        else:
            return self.experiment_group_variation.variation_name

    def validate(self) -> bool:
        errors = []
        # allow treatment value and control value to be null
        # as it's needed to add interaction result
        if isnan(self.estimated_treatment_effect):
            errors.append(f"Result for {self.metric.column_name} is skipped because Estimated Treatment Effect is NaN")

        if isnan(self.p_value):
            errors.append(f"Result for {self.metric.column_name} is skipped because P Value is NaN")

        if isnan(self.confidence_interval_left) or isnan(self.confidence_interval_right):
            errors.append(f"Result for {self.metric.column_name} is skipped because Confidence Interval is NaN")

        if isnan(self.se):
            errors.append(f"Result for {self.metric.column_name} is skipped because Standard Error is NaN")

        if isnan(self.metric_treatment_value):
            errors.append(f"Result for {self.metric.column_name} is skipped because Metric Treatment Value is NaN")

        if isnan(self.metric_control_value):
            errors.append(f"Result for {self.metric.column_name} is skipped because Metric Control Value is NaN")

        if self.is_sequential_result and isnan(self.sequential_p_value):
            errors.append(f"Result for {self.metric.column_name} is skipped because Sequential P Value is NaN")

        if self.is_sequential_result and (
            isnan(self.sequential_confidence_interval_left) or isnan(self.sequential_confidence_interval_right)
        ):
            errors.append(
                f"Result for {self.metric.column_name} is skipped because Sequential Confidence Interval is NaN"
            )

        if len(errors) > 0:
            self.message_collection.add_metric_message(
                self.metric.column_name,
                Message(
                    source=Source.analysis,
                    status=Status.fail,
                    title="Analysis result is invalid",
                    description="\n".join(errors),
                ),
            )

        return len(errors) == 0

    @property
    def relative_treatment_effect(self):
        if self.metric_control_value is None:
            return None
        if self.metric_control_value == 0:
            self.message_collection.add_metric_message(
                self.metric.column_name,
                Message(
                    source=Source.analysis,
                    status=Status.fail,
                    title="Analysis result is invalid",
                    description=f"""
                    Relative treatment effect for metric "{self.metric.column_name}" cannot be calculated \
                    because the control metric value is zero.
                    """,
                ),
            )
            return None
        return self.estimated_treatment_effect / self.metric_control_value

    def _get_relative_value(self, baseline: Optional[float], lift: float) -> Optional[float]:
        if lift is None or baseline is None or baseline == 0:
            return None
        return lift / baseline

    @property
    def relative_confidence_interval_left(self):
        return self._get_relative_value(self.metric_control_value, self.confidence_interval_left)

    @property
    def relative_confidence_interval_right(self):
        return self._get_relative_value(self.metric_control_value, self.confidence_interval_right)

    @property
    def sequential_relative_confidence_interval_left(self):
        return self._get_relative_value(self.metric_control_value, self.sequential_confidence_interval_left)

    @property
    def sequential_relative_confidence_interval_right(self):
        return self._get_relative_value(self.metric_control_value, self.sequential_confidence_interval_right)

    @property
    def result_dicts_for_dataframe_output(
        self,
    ) -> List[Dict[str, Union[Optional[str], Optional[float], List[Optional[float]]]]]:
        treatment_dict = OrderedDict(
            {
                Constants.METRIC_NAME: self.metric.original_column_name,
                Constants.EXPERIMENT_GROUP_NAME: self.experiment_groups_name,
                Constants.VARIATION: self.experiment_groups_treatment_variation_name,
                Constants.METRIC_VALUE: self.metric_treatment_value,
                Constants.AVERAGE_TREATMENT_EFFECT: self.estimated_treatment_effect,
                Constants.RELATIVE_AVERAGE_TREATMENT_EFFECT: self.relative_treatment_effect,
                Constants.P_VALUE: self.p_value,
                Constants.CONFIDENCE_INTERVAL: [
                    self.confidence_interval_left,
                    self.confidence_interval_right,
                ],
                Constants.SAMPLE_SIZE: self.metric_treatment_sample_size,
                Constants.DATA_SIZE: self.metric_treatment_data_size,
                Constants.SE: self.se,
            }
        )
        control_dict = OrderedDict(
            {
                Constants.METRIC_NAME: self.metric.original_column_name,
                Constants.EXPERIMENT_GROUP_NAME: self.experiment_groups_name,
                Constants.VARIATION: self.experiment_groups_control_variation_name,
                Constants.METRIC_VALUE: self.metric_control_value,
                Constants.AVERAGE_TREATMENT_EFFECT: np.nan,
                Constants.RELATIVE_AVERAGE_TREATMENT_EFFECT: np.nan,
                Constants.P_VALUE: np.nan,
                Constants.CONFIDENCE_INTERVAL: np.nan,
                Constants.SAMPLE_SIZE: self.metric_control_sample_size,
                Constants.DATA_SIZE: self.metric_control_data_size,
                Constants.SE: np.nan,
            }
        )
        return [control_dict, treatment_dict]

    @property
    def control_result_dict(self) -> Dict[str, Union[Optional[str], Optional[float]]]:
        metric_value = None if self.is_interaction_result else self.metric_control_value
        sample_size = None if self.is_interaction_result else self.metric_control_sample_size
        return {
            Constants.METRIC_NAME: self.metric.original_column_name,
            Constants.AGG_FUNC_NAME: self.metric.metric_aggregate_func.name,
            Constants.EXPERIMENT_GROUP_NAME: self.experiment_groups_name,
            Constants.VARIATION: self.experiment_groups_control_variation_name,
            Constants.METRIC_VALUE: metric_value,
            Constants.SAMPLE_SIZE: sample_size,
            Constants.DATA_SIZE: self.metric_control_data_size,
        }

    @property
    def treatment_result_dict(
        self,
    ) -> Dict[
        str,
        Union[
            Optional[
                Union[
                    str,
                    int,
                    float,
                    Optional[Union[ExperimentVariation, Tuple[ExperimentVariation, ...]]],
                    List[Optional[float]],
                ]
            ]
        ],
    ]:
        if self.is_interaction_result:
            metric_value = None
            sample_size = None
        else:
            metric_value = self.metric_treatment_value
            sample_size = self.metric_treatment_sample_size

        result_dict = {
            Constants.METRIC_NAME: self.metric.original_column_name,
            Constants.AGG_FUNC_NAME: self.metric.metric_aggregate_func.name,
            Constants.EXPERIMENT_GROUP_NAME: self.experiment_groups_name,
            Constants.VARIATION: self.experiment_groups_treatment_variation_name,
            Constants.AVERAGE_TREATMENT_EFFECT: self.estimated_treatment_effect,
            Constants.P_VALUE: self.p_value,
            Constants.METRIC_VALUE: metric_value,
            Constants.ABSOLUTE_CONFIDENCE_INTERVAL: [
                self.confidence_interval_left,
                self.confidence_interval_right,
            ],
            Constants.RELATIVE_CONFIDENCE_INTERVAL: [
                self.relative_confidence_interval_left,
                self.relative_confidence_interval_right,
            ],
            Constants.RELATIVE_AVERAGE_TREATMENT_EFFECT: self.relative_treatment_effect,
            Constants.SAMPLE_SIZE: sample_size,
            Constants.DATA_SIZE: self.metric_treatment_data_size,
            Constants.SE: self.se,
        }

        if not any(
            [
                self.sequential_p_value is None,
                self.sequential_confidence_interval_left is None,
                self.sequential_confidence_interval_right is None,
                self.sequential_relative_confidence_interval_left is None,
                self.sequential_relative_confidence_interval_right is None,
            ]
        ):
            result_dict.update(
                {
                    Constants.SEQUENTIAL_P_VALUE: self.sequential_p_value,
                    Constants.SEQUENTIAL_RESULT_TYPE: self.sequential_result_type,
                    Constants.SEQUENTIAL_ABSOLUTE_CONFIDENCE_INTERVAL: [
                        self.sequential_confidence_interval_left,
                        self.sequential_confidence_interval_right,
                    ],
                    Constants.SEQUENTIAL_RELATIVE_CONFIDENCE_INTERVAL: [
                        self.sequential_relative_confidence_interval_left,
                        self.sequential_relative_confidence_interval_right,
                    ],
                }
            )
        return result_dict

    def convert_result_to_list(self) -> List:
        return [
            self.metric.original_column_name,
            self.estimated_treatment_effect,
            self.p_value,
            self.confidence_interval_left,
            self.confidence_interval_right,
            self.experiment_group,
            self.experiment_groups_treatment_variation_name,
        ]

    def convert_result_for_simulation(self) -> List:
        return [
            self.estimated_treatment_effect,
            self.p_value,
            self.confidence_interval_left,
            self.confidence_interval_right,
        ]


class SimulationResult:
    def __init__(self, power: float, within_ci: float):
        self.power = power
        self.within_ci = within_ci


class DiDOutputResult:
    def __init__(self, analysis_result, matching_result):
        self.matching_result = matching_result
        self.analysis_result = analysis_result
