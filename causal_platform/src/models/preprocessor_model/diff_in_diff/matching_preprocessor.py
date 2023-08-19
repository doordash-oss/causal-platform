"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import heapq
from typing import List, Optional

import pandas as pd

from causal_platform.src.models.configuration_model.config import DiDConfig
from causal_platform.src.models.message.enums import Source, Status
from causal_platform.src.models.message.message import Message
from causal_platform.src.models.preprocessor_model.base import BasePreprocessor
from causal_platform.src.models.result_model.result import DiDMatchUnit, PreprocessResult
from causal_platform.src.utils.constants import Constants
from causal_platform.src.utils.diff_in_diff.calculation import (
    calculate_distance,
    standardize_and_calculate_weighted_sum,
)
from causal_platform.src.utils.diff_in_diff.prep_data import (
    get_aggregate_metric_in_unit_ids,
    get_data_between_start_end_date,
    get_unit_candidates,
)
from causal_platform.src.utils.logger import logger

MATCHING_TITLE = "Matching transformation"


class MatchingPreprocessor(BasePreprocessor):
    def __init__(self, config: DiDConfig, control_unit_ids: Optional[List[str]] = None):
        self.config = config
        self.control_unit_ids = control_unit_ids

    def process(self, data: pd.DataFrame) -> PreprocessResult:
        """function to find matched market

        Arguments:
            data {pd.DataFrame} -- input data
            config {DiDConfig} -- config class

        Returns:
            PreprocessResult -- result
        """

        min_distance = []
        data = data.copy()
        preprocess_result = PreprocessResult(processed_data=data)

        data[Constants.WEIGHTED_SUM_COLUMN_NAME] = standardize_and_calculate_weighted_sum(
            data[self.config.matching_columns].to_numpy(), self.config
        )

        matching_data = get_data_between_start_end_date(
            data,
            self.config.date.column_name,
            self.config.matching_start_date,
            self.config.matching_end_date,
        )

        treatment_aggregated_data = get_aggregate_metric_in_unit_ids(
            matching_data,
            self.config.date.column_name,
            self.config.experiment_randomize_units[0].column_name,
            self.config.treatment_unit_ids,
            Constants.WEIGHTED_SUM_COLUMN_NAME,
            Constants.DIFF_IN_DIFF_MATCHING_AGGREGATE_FUNC,
        )
        if self.control_unit_ids:
            unit_id_candidates = self.control_unit_ids
            match_unit_size = len(unit_id_candidates)
        else:
            unit_id_candidates = get_unit_candidates(
                matching_data,
                self.config.experiment_randomize_units[0].column_name,
                self.config.exclude_unit_ids,
                self.config.treatment_unit_ids,
            )
            match_unit_size = self.config.match_unit_size

        if len(unit_id_candidates) == 0:

            description = f"Found zero candidates for control group! Please make sure there is at least {self.config.match_unit_size} candidates for matching"

            preprocess_result.message_collection.add_overall_message(
                Message(
                    source=Source.transformation,
                    status=Status.fail,
                    title=MATCHING_TITLE,
                    description=description,
                )
            )
        else:
            logger.info("Found {} unit candidates for control group!".format(len(unit_id_candidates)))

        logger.info("Start to search units candidates for best match..")
        for unit_id in unit_id_candidates:
            control_aggregated_data = get_aggregate_metric_in_unit_ids(
                matching_data,
                self.config.date.column_name,
                self.config.experiment_randomize_units[0].column_name,
                [unit_id],
                Constants.WEIGHTED_SUM_COLUMN_NAME,
                Constants.DIFF_IN_DIFF_MATCHING_AGGREGATE_FUNC,
            )

            if treatment_aggregated_data.shape[0] == control_aggregated_data.shape[0]:
                # if market doesn't have the same length of date as treatment market, drop it
                distance = calculate_distance(treatment_aggregated_data, control_aggregated_data, self.config)

                if len(min_distance) < match_unit_size:
                    heapq.heappush(min_distance, (-distance, unit_id))
                else:
                    heapq.heappushpop(min_distance, (-distance, unit_id))

        control_units = [
            DiDMatchUnit(
                unit_id=distance_unit_pair[1],
                matching_score=distance_unit_pair[0],
                score_name=self.config.matching_method.value,
            )
            for distance_unit_pair in min_distance
        ]

        preprocess_result.message_collection.add_overall_message(
            Message(
                source=Source.transformation,
                status=Status.success if len(control_units) > 0 else Status.fail,
                title=MATCHING_TITLE,
                description=f"Search finished! Found {len(control_units)} control units.",
            )
        )

        preprocess_result.processed_data = data
        preprocess_result.matched_units = control_units

        return preprocess_result
