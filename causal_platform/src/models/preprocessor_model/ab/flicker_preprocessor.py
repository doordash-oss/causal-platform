"""
  Copyright 2023 DoorDash, Inc.

  Licensed under the Apache License, Version 2.0 (the License);
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""

import collections
from typing import List, Optional

import pandas as pd

from causal_platform.src.models.configuration_model.base_objects import Cluster, Column, ExperimentGroup
from causal_platform.src.models.message.message import Message, Source, Status
from causal_platform.src.models.preprocessor_model.base import BasePreprocessor
from causal_platform.src.models.result_model.result import PreprocessResult
from causal_platform.src.utils.logger import logger

FLICKER_TITLE = "Flicker validation"


class FlickerPreprocessor(BasePreprocessor):
    def __init__(
        self,
        experiment_groups: List[ExperimentGroup],
        experiment_randomize_units: Optional[List[Column]],
        cluster: Cluster,
        is_remove_flickers: bool = False,
    ):
        self.experiment_groups = experiment_groups
        self.experiment_randomize_units = experiment_randomize_units
        self.cluster = cluster
        self.is_remove_flickers = is_remove_flickers

    def process(self, data: pd.DataFrame) -> PreprocessResult:
        preprocess_result = PreprocessResult(data)

        randomize_units = [randomize_unit.column_name for randomize_unit in self.experiment_randomize_units]
        if len(randomize_units) > 0:
            units = randomize_units
            logger.info(f"Running flicker test on experiment_randomize_unit {randomize_units}")
        elif self.cluster:
            units = self.cluster.column_name
            logger.info(f"Running flicker test on cluster {self.cluster.column_name}")
        else:
            description = "Skipping flicker check as cluster or experiment_randomize_unit is not provided in the config"
            preprocess_result.message_collection.add_overall_message(
                Message(
                    source=Source.validation,
                    status=Status.skip,
                    title=FLICKER_TITLE,
                    description=description,
                )
            )
            return preprocess_result

        # Track which groups have flickers
        group_flickers = {}
        FlickerData = collections.namedtuple("FlickerData", ["fraction", "numerator", "denominator"])

        # Calculate the flickers as the (# of units with multiple treatment groups) / (# of units) for each experiment_group
        for group in self.experiment_groups:
            flicker_units_count, total_units_count = self.process_flicker_given_units(data, group, units)
            flicker_fraction = flicker_units_count / total_units_count

            if flicker_fraction >= 0.001 and flicker_units_count >= 10:
                group_flickers[group.column_name] = FlickerData(
                    fraction=flicker_fraction,
                    numerator=flicker_units_count,
                    denominator=total_units_count,
                )
        preprocess_result.does_flicker_exists = len(group_flickers) > 0

        flicker_message = Message(source=Source.validation, status=Status.success, title=FLICKER_TITLE, description="")

        if len(group_flickers) > 0:
            descriptions = []
            for group_name, flicker_data in group_flickers.items():
                text = f"""Flicker Test Failed: {flicker_data.numerator} entities ({flicker_data.fraction:.2%}) were exposed to 2 or more variants."""
                descriptions.append(text)
            if self.is_remove_flickers:
                descriptions.append(
                    "We have removed entities with flicker from the experiment analysis and we recommend reviewing the experiment configuration to check for errors."
                )
            flicker_message.status = Status.fail
            flicker_message.description = "\n".join(descriptions)

        preprocess_result.message_collection.add_overall_message(flicker_message)
        return preprocess_result

    def process_flicker_given_units(self, data: pd.DataFrame, experiment_group: ExperimentGroup, units: List[str]):
        """
        :param data: data to drop flickers in place
        :param experiment_group: experiment group
        :param units: list of units used as the group key to check flickers
        :return: (number of flicker units, number of total units)
        """

        def drop_flickers_inplace(flickers_units_index, data_units_index, data):
            IS_FLICKER_TEMP_COLUMN = "is_flicker_temp_col"

            data[IS_FLICKER_TEMP_COLUMN] = data_units_index.isin(flickers_units_index)

            # conduct an inplace drop instead of reassignment to save memory
            data.drop(data[data[IS_FLICKER_TEMP_COLUMN]].index, inplace=True)
            data.drop(IS_FLICKER_TEMP_COLUMN, axis=1, inplace=True)

        # Find number of treatment groups per user
        group_cnt = (
            data[data[experiment_group.column_name].notnull()].groupby(units)[experiment_group.column_name].nunique()
        )

        # Find the flickered users
        flickers_units_index = group_cnt[group_cnt > 1].index
        data_units_index = data.set_index(units).index
        if self.is_remove_flickers:
            drop_flickers_inplace(flickers_units_index, data_units_index, data)
        return len(flickers_units_index), len(group_cnt)
